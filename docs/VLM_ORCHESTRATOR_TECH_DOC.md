# 大模型（VLM）驱动的机器人多任务调度算法技术解析

> 对象：`src/lerobot/act_vlm_orchestrator_v2`（ACT + VLM 编排器）与 `src/lerobot/pi05_vlm_orchestrator`（PI0.5 + VLM 编排器）
> 用途：算法分享 / 技术评审
> 关键词：VLM 场景调度、窗口投票共识、静止检测重评估、多策略模型切换、单模型任务文本切换、RTC 实时分块

---

## 目录

1. [背景与问题定义](#1-背景与问题定义)
2. [总体架构](#2-总体架构)
3. [共享调度层：核心算法机制详解](#3-共享调度层核心算法机制详解)
4. [ACT 编排器：多模型切换方案](#4-act-编排器多模型切换方案)
5. [PI0.5 编排器：单模型任务文本切换方案](#5-pi05-编排器单模型任务文本切换方案)
6. [两种方案深度对比](#6-两种方案深度对比)
7. [底层执行链路的工程细节（两者共用）](#7-底层执行链路的工程细节两者共用)
8. [关键技术问题与解决经验（实战记录）](#8-关键技术问题与解决经验实战记录)
9. [相关文件索引](#9-相关文件索引)

---

## 1. 背景与问题定义

### 1.1 业务场景：包裹分拣流水线

一条固定的长程操作流程被拆分为 4 个子任务，场景之间由**包裹物理位置 + 面单标签可见性**两个状态量唯一决定：

| VLM 分类 | 场景描述 | 执行策略 | 动作 |
|---------|---------|---------|------|
| `subtask1` | 包裹在桌面 | Task_A | 抓取包裹 -> 放入托盘/扫码台 |
| `subtask2` | 包裹在托盘，标签不可见（朝下） | Task_B | 翻转包裹使标签朝上 |
| `subtask3` | 包裹在托盘，标签可见（朝上） | Task_C | 抓取包裹 -> 放入盒子 |
| `subtask4` | 包裹在盒子（桌面/托盘均无包裹） | standby | 待机等待场景复原（人工放回新包裹） |

这是一个**天然的有限状态机（FSM）**，但状态不能预先编程，必须**实时从视觉观测推断**——这就是引入 VLM 的原因。

### 1.2 技术挑战

| 挑战 | 解决方案 |
|------|---------|
| 端到端 VLA 模型一次只能学一个技能包，无法自主长程规划 | 采用「VLM 高层调度 + 技能策略底层执行」的分层架构 |
| VLM（2B 小模型微调）单帧判断有噪声，会误判 | 窗口投票（5 帧取 ≥3 共识）+ 连拍机制 |
| 策略模型不知道"任务做完了"，会无限循环或保持不动 | 静止检测触发 VLM 重评估（保险机制） |
| 任务切换时的动作衔接跳变、模型状态残留 | 切换时清缓存 + 重置策略状态 + 平滑器过渡 |
| 低成本机械臂（Piper）伺服抖动 / 夹爪信号弱 | 保持锁定、夹爪强制闭合、分段增益等工程补偿 |

### 1.3 两个方案的一句话概括

- **ACT + VLM 编排器（v2）**：VLM 调度**多个独立训练的 ACT 策略模型**（Task_A/B/C 各一份权重），切换任务 = 换激活的模型。
- **PI0.5 + VLM 编排器**：VLM 调度**同一个多任务 VLA 模型（PI0.5）**，切换任务 = 只换 task 文本 + 重置推理状态（毫秒级）。

两者的**调度层（VLM + 投票 + 静止检测 + 状态机）完全同构**，差异集中在「策略层」与「推理执行链路」。

---

## 2. 总体架构

### 2.1 架构图

```
┌────────────────────────────────────────────────────────────────────┐
│                       VLM 多任务编排器                              │
│                                                                    │
│   ┌────────────┐    ┌──────────┐    ┌────────────┐    ┌─────────┐ │
│   │  Camera    │───▶│   VLM    │───▶│  Window    │───▶│  任务/  │ │
│   │  Manager   │    │  Client  │    │  Voter     │    │ 模型切换 │ │
│   │ (帧缓存)   │    │ (vLLM)   │    │ (5帧≥3票) │    │         │ │
│   └────────────┘    └──────────┘    └────────────┘    └────┬────┘ │
│          ▲                 ▲                          ┌──────▼────┐ │
│   ┌──────┴──────┐   ┌──────┴───────┐                  │  策略执行  │ │
│   │ Stationary  │   │  主循环状态机  │                  │  ┌──────┐ │ │
│   │  Detector   │──▶│ (Orchestrator)│────────────────▶│  │ ACT  │ │ │
│   │ (2s静止触发) │   └──────────────┘                  │  │ 多模型│ │ │
│   └─────────────┘         │    ▲                      │  └──────┘ │ │
│                           │    └── SafetyMonitor ──▶  │  ┌──────┐ │ │
│                           ▲        (跳变急停)          │  │PI0.5 │ │ │
│                    ┌──────┴───────┐                   │  │单模型 │ │ │
│                    │ Piper 机械臂  │◀──────────────────│  └──────┘ │ │
│                    │ (CAN 总线)   │    平滑/夹爪过滤    └──────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
相机帧 ──┬──▶ VLM 场景分类 (subtask1~4) ──▶ 投票 ──▶ 共识 ──▶ 切换模型/任务文本
        └──▶ 策略模型观测 (state + top + wrist) ──▶ 动作 chunk ──▶ 平滑 ──▶ 安全检查 ──▶ 机械臂

关节反馈 ──▶ 静止检测 ──(静止≥2s)──▶ 触发 VLM 连拍重评估
```

### 2.3 模块清单（两套编排器文件结构对仗）

| 模块 | ACT 版 | PI0.5 版 | 职责 |
|------|--------|----------|------|
| 编排主类 | `orchestrator.py` (`ACTVLMOrchestrator`, 1611 行) | `orchestrator.py` (`PIVLMOrchestrator`, 1849 行) | 主循环状态机、连拍、切换、平滑、发送 |
| 模型管理 | `act_model_manager.py` (`ACTModelManager`, 572 行) | `pi_model_manager.py` (`PIModelManager`, 845 行) | 模型加载/切换/推理（PI 版含后台线程 + RTC） |
| VLM 客户端 | `vlm_client.py` (419 行) | 同左（仅 import 路径不同） | vLLM HTTP 调用 + 响应解析 |
| 投票器 | `window_voter.py` (243 行) | 同左 | 滑动窗口共识 |
| 静止检测 | `stationary_detector.py` (336 行) | 同左 | 两段式静止判定 |
| 相机管理 | `camera_manager.py` (368 行) | 同左 | 统一帧缓存 |
| 配置 | `config.py` (272 行) | `config.py` (308 行) | 全部超参集中管理 |
| 诊断日志 | （内嵌于编排器） | `diag_logger.py` | 推理/执行指标 CSV |
| 入口 | `run_orchestrator.py` + `start_orchestrator.sh` | 同左 | argparse -> config -> 运行 |

---

## 3. 共享调度层：核心算法机制详解

> 本章是两套编排器共用的「大模型调度算法」核心，也是分享的重点。文件以 ACT 版为参照（PI 版逻辑一致）。

### 3.1 VLM 场景分类客户端（`vlm_client.py`）

#### 3.1.1 VLM 的选择与部署

- 模型：**Qwen3-VL-2B 微调版**（`qwen3_vl_2b_parcel_subtask_merged`），部署在 **vLLM** 上，暴露 OpenAI 兼容的 `/v1/chat/completions` 接口。
- 编排器通过 HTTP（`requests.Session`）调用，`temperature=0`（确定性输出）、`max_tokens=128`、`timeout=8s`。
- 相比在主进程里直接加载 VLM，vLLM 独立进程 + HTTP 的好处：显存隔离（不与策略模型抢 GPU）、崩溃隔离、可水平扩展。

#### 3.1.2 Prompt 设计（与微调数据逐字对齐）

```python
VLM_PROMPT = (
    "<image><image>你是一个包裹抓取任务的视觉调度器。\n\n"
    "【相机信息】\n"
    "- 第一张图：顶部全局相机，固定在工作台正上方，提供俯视全局视图，用于判断包裹位置（桌面/托盘/盒子）和快递单标签是否朝上可见\n"
    "- 第二张图：腕部近景相机，安装在机械臂夹爪末端，提供包裹近距离特写，用于辅助确认包裹位置与快递单标签可见性\n\n"
    "【场景物体】\n"
    "- 包裹：灰白色包装袋/棕色纸箱，表面可能贴有白色快递单（含条码、运单文字），可能位于桌面、托盘或盒子内\n"
    "- 托盘：深色圆形，位置固定在工作台上\n"
    "- 盒子：矩形纸箱，位置固定在工作台上\n\n"
    "【判断规则】\n"
    "1. 先用第一张图（顶部俯视）判断包裹位置：桌面 / 托盘 / 盒子\n"
    "2. 再结合第一张图（顶部俯视）与第二张图（腕部近景）判断快递单标签是否可见\n"
    "3. 根据 包裹位置 + 标签可见性 输出当前任务阶段：\n"
    "   - subtask1：包裹在桌面上 -> 抓取包裹放入托盘\n"
    "   - subtask2：包裹在托盘中，且标签不可见（朝下）-> 翻转包裹使标签朝上\n"
    "   - subtask3：包裹在托盘中，且标签可见（朝上）-> 夹取包裹放入盒子\n"
    "   - subtask4：包裹已在盒子中（托盘和桌面均无包裹）-> 任务完成（done）\n\n"
    "请严格按JSON格式输出，不要添加Markdown标记，不要解释：\n"
    '{"parcel_location": "桌面|托盘|盒子", "label_visible": true|false, "subtask": "subtask1|subtask2|subtask3|subtask4"}'
)
```

**要点**：
1. **两路视觉输入**：顶部全局相机（判断位置）+ 腕部近景相机（辅助判断标签可见性）--多视角互补。
2. **结构化思维链**：prompt 里明确「先判断位置 -> 再判断标签可见性 -> 组合输出阶段」的两步推理规则，微调数据（`train_data_ABC_v3.json`）也按此格式构造。**推理时必须与训练 prompt 逐字一致**，否则模型学到的"两步推理 + JSON 输出"行为不会被正确触发。
3. **输出格式约束**：强制 JSON 三字段输出，降低解析成本。

#### 3.1.3 输入构造（多模态消息）

`classify(top_image, wrist_image)` 构造 OpenAI 格式消息：图像（numpy -> PIL -> JPEG quality=90 -> base64 data URL）两张 + 文本 prompt。图像顺序与 prompt 中「第一张图/第二张图」的描述严格对应。

#### 3.1.4 响应解析--三级容错（鲁棒性设计核心）

VLM 输出不可靠是常态，解析器 `_parse_response()` 做了**三级容错**：

```
原始文本
   │
   ▼
① JSON 提取 (_extract_json)
   │  剥离 ```json 围栏 -> 正则匹配 {...} -> 宽松兜底（首尾大括号截取）
   ├── 成功 -> json.loads -> 取三字段 (confidence=0.9)
   └── 失败 ↓
② 关键词回退 (_keyword_fallback, confidence=0.3)
   │  文本中直接搜 "桌面/托盘/盒子"、"subtask1~4"、"不可见/朝下" 等关键词
   │  （注意: "不可见"是"可见"的子串, 必须先排除前者）
   └── ↓
③ 一致性校验 (_derive_subtask)  ← 必经
      用 (parcel_location, label_visible) 按确定性规则重新推导 subtask：
        桌面 -> subtask1；托盘+不可见 -> subtask2；托盘+可见 -> subtask3；盒子 -> subtask4
      若与模型输出的 subtask 不一致 -> 以推导值为准（打 warning 日志）
```

**设计哲学**：VLM 的 `subtask` 字段可能与另外两个字段自相矛盾（模型并非联合推理三字段），而「位置 + 标签可见性 -> subtask」是**确定性映射**，所以最终决策永远以推导值为准。这一层校验把「VLM 自相矛盾」类错误彻底消除。

另外 `_normalize_bool()` 处理 `"true"/"false"` 字符串陷阱（`bool("false") == True` 的经典坑），`"false"/"0"/"no"/"否"/"不可见"` 都归一化为 `False`。

#### 3.1.5 失败语义

API 调用失败/超时/连接错误 -> 返回 `VLMDecision(success=False)`。失败结果**不参与投票，但也不清空窗口**（见 3.2），编排器整体降级为「保持当前模型继续执行」。

### 3.2 窗口投票共识机制（`window_voter.py`）

#### 3.2.1 机制描述

```
参数: vote_window_size=5, vote_consensus_threshold=3

决策流: deque(maxlen=5) 存 (subtask, model) 元组
窗口满(5帧) -> Counter 统计众数 -> 众数票数 ≥ 3 ?
   ├── 是 -> 达成共识 -> get_consensus() 返回 (subtask, model)，窗口清空
   └── 否 -> 不动作，等待窗口滑动
```

- **失败优雅降级**：失败的 VLM 调用直接跳过投票，窗口保持不动--单次网络抖动不会污染投票。
- **共识后清窗**：避免旧结果（上一个场景的判断）干扰新一轮投票，这是防「决策惯性」的关键。
- **`peek_consensus()` vs `get_consensus()`**：前者只读不消费，后者消费后清空窗口。
- **`force_consensus(subtask)`**：外部强制注入共识（虚拟全票），供特殊场景使用。

#### 3.2.2 为什么是 5/3 而不是单帧？

2B 微调 VLM 单帧判断的错误来源：图像模糊/遮挡（机械臂挡住包裹）、临界场景（包裹半个在托盘外）、光照变化。连拍（见 3.4）期间机械臂静止、场景不变，此时要求 ≥3 帧一致能过滤**独立随机噪声**：若单帧错误率为 p，5 帧中 ≥3 帧"错得一致"的概率约为 10p³ 量级（p=0.2 时约 5.8%），比单帧误判率低一个数量级。代价是**决策延迟**（连拍 5 帧约 0.5~1s），对秒级的分拣任务完全可接受。

#### 3.2.3 窗口满但无共识的处理（编排器侧）

主循环在连拍结束后检查：若窗口已满但没有共识（如 2:2:1 分布），说明场景处于过渡态/临界态，**清空窗口等待下次静止触发再投**，不强行决策。

### 3.3 静止检测器（`stationary_detector.py`，v3 两段式判定）

**用途（保险机制）**：策略模型不会显式报告"我做完了"。当机械臂接近静止一段时间，大概率意味着：① 子任务完成，等待下一个；② 模型失效/卡住；③ 遇到障碍。此时应触发 VLM 重新评估场景。**运动期间 VLM 零干预**--不打断正在执行的策略。

#### 3.3.1 v1 -> v2 -> v3 的演进（踩坑记录）

| 版本 | 判据 | 问题 |
|------|------|------|
| v1 | 单步关节 delta < 0.01745 rad | 25Hz 下等效速度阈值 ≈0.44 rad/s，过于宽松，**低速运动被误判为静止** -> 运动中切换任务 |
| v2 | 纯速度阈值 | 过严：子任务完成后伺服保持的微小抖动（0.001~0.005 rad 来回摆，折算速度可达 0.05 rad/s）**反复打断计时** -> 永不触发，切换流程卡死 |
| v3 | **两段式**（现役） | 兼容两者 |

#### 3.3.2 v3 两段式判定

```
1) 进入判定:  相邻采样速度 < stationary_velocity_threshold (0.05 rad/s ≈ 2.9°/s)
             -> 锁定参考位置, 开始计时
2) 维持判定: 静止期间位置相对参考的总漂移 < stationary_hold_band (0.045 rad ≈ 2.6°)
             -> 持续累计静止时间; 超带则重置
3) 参考重锁: 每隔 stationary_reference_renew_interval (0.5s) 重锁一次参考位置
             -> 把"总漂移"退化为"窗口内漂移", 缓慢单调漂移不会永久卡死计时
4) 触发:     静止时间 ≥ stationary_time_threshold (2.0s) -> update() 返回 True
             (受 stationary_recheck_cooldown=5s 冷却约束, 防止频繁触发)
```

**物理直觉**：伺服抖动是**来回摆动**（总位移≈0）天然被保持带容忍；缓慢单调漂移会累计超出保持带 -> 正确判为运动。带内的 ±1.5° 物理滞后被 2.6° 的带覆盖。

#### 3.3.3 辅助接口

- `is_moving(joints)`：连拍期间的运动监测（独立基准），机械臂恢复运动立即中止连拍。
- `set_fast_recheck(bool)`：快速重评估模式开关（见 3.6.3）。

### 3.4 VLM 连拍（Burst）机制

静止触发后**一次性连拍多帧填满投票窗口**，而不是等多次独立的静止事件（那样太慢）：

```
_start_vlm_burst(abort_on_motion=True, settle_time_s=0.0):
    _burst_active = True, _burst_count = 0
    voter.reset()                       # 清空旧窗口
    _burst_next_time = now + settle     # 可选静置
    _burst_start_time = now             # 整体超时基准

_vlm_burst_tick():                      # 每个主循环步调用
    if 连拍总时长 > vlm_burst_total_timeout (5s): 强制结束   # 防 VLM 持续失败卡死
    if _burst_count >= 5: return        # 已拍满, 等主循环消费共识
    if now < _burst_next_time: return   # 未到间隔 (vlm_burst_interval=0.1s)
    _vlm_check_and_vote()               # 取帧 -> VLM classify -> voter.add_decision
    _burst_count += 1
```

**连拍期间的三个约束**：
1. **策略推理暂停**（主循环直接 `continue`）：手臂已静止，策略输出无意义；同时避免策略与 VLM 竞争相机帧；VLM 需要无遮挡的清晰场景。
2. **运动中止**（`abort_on_motion=True` 时）：每步检测 `stationary_detector.is_moving()`，机械臂恢复运动则立即丢弃半途投票并中止--**在运动中切换任务是被严格禁止的**。
3. **帧新鲜度**：`vlm_frame_max_age=0.5s`，过期帧不送 VLM（应对相机流延迟）。

**连拍结束**（`_finish_vlm_burst`）：
- 中止时：丢弃未消费投票（`voter.reset()`）。
- 静止检测器 `full_reset(keep_cooldown=True)`：保留冷却计时，防止"静止->连拍->立即又触发"的 1~2s 周期性抖动循环（实测出现过每 1~2s 一次连拍、每次暂停机械臂+重新预测 -> 周期性抖动）。
- 若本次连拍**未触发模型切换** -> 启用 fast_recheck（见 3.6.3）。
- `model_manager.reset_active_policy()`：清空策略动作队列，防止连拍结束后回放连拍前基于旧观测的陈旧 chunk（曾导致"抓取前先抬升再恢复"的抖动）。

### 3.5 主循环状态机（`orchestrator.py::_main_loop`）

两个编排器的主循环结构完全一致，按**固定控制频率**（ACT: 30Hz / PI: 20Hz，`busy_wait` 高精度节拍）运行：

```
启动 ──▶ 首次场景评估(连拍投票, 确认初始模型后才动手)   ★ 重要改进
  │
  ▼ ┌────────────────────────────────────────────────────┐
┌─▶ │ 阶段1: 获取观测                                      │
│   │   关节状态 + 帧缓存图像(新鲜度检查, 过期/缺失则跳过本步) │
│   │                                                      │
│   │ 阶段2: 静止检测 + VLM 连拍                            │
│   │   if 连拍中:  运动中止检查 / 连拍tick                  │
│   │   elif 切换保护期内: 跳过静止检测                      │
│   │   else:     静止≥2s -> _start_vlm_burst()             │
│   │                                                      │
│   │ 阶段3: 消费共识 -> _handle_consensus() -> 切换模型/任务  │
│   │   (窗口满但无共识 -> 清窗重来)                          │
│   │                                                      │
│   │ 阶段4: 待机模式处理 _handle_standby()                 │
│   │   (standby 时不推理, 静止检测照常, 等场景复原)          │
│   │                                                      │
│   │ 阶段4.5: 子任务超时检测 (30s 强制重评估)               │
│   │                                                      │
│   │ 阶段5: 策略推理 (连拍中跳过; PI 队列空返回 None 则保持) │
│   │ 阶段6: 轨迹平滑 (EMA + 速率限制 + 保持锁定)            │
│   │ 阶段7: 安全检查 (跳变急停)                            │
│   │ 阶段8: 动作转换(rad->0.001°)+夹爪处理 -> 发送机械臂      │
│   └────────────────────────────────────────────────────┘
```

**首次场景评估**（关键改进）：启动时不直接用默认模型动手，而是先做一轮连拍投票确认场景。旧版只做 1 次 VLM 判断无法达成 5/3 共识，机械臂会先按默认模型执行错误动作（无动作或轻微抬升又放下）。新版在共识确认前暂停策略推理。

**共识处理**（`_handle_consensus`）的三种分支：
1. 目标 = standby 且已在待机 -> 仅更新 subtask 标签；
2. 目标 = 当前模型 -> 无需切换，仅更新标签（此时 `_last_burst_switched=False`，触发 fast_recheck）；
3. 真正切换 -> `model_manager.switch_model()` -> 重置子任务计时/保护期 -> 重置平滑器（用当前关节+夹爪 seed 基线，避免切换后 EMA 把关节拉向旧值）-> `begin_transition()` 过渡平滑 -> 重置静止检测器/保持锁定/chunk 诊断基准。

### 3.6 其它保险机制

#### 3.6.1 子任务超时（`subtask_timeout=30s`）

防止「子任务卡死/循环但关节仍在运动导致静止检测永不触发」。超时后强制连拍（`abort_on_motion=False, settle_time_s=1.0`--机械臂可能正在循环运动中，需要强制重评估，且先静置 1s 让残余运动衰减再取帧）。每个子任务只触发一次（`_subtask_timeout_triggered` 标记，切换后复位）。

#### 3.6.2 切换保护期（`switch_cooldown=3s`）

模型/任务切换成功后的 3s 内**完全禁止** VLM 连拍与再切换--给新模型足够的"预热"执行时间，避免"一静止就切"的震荡。保护期结束时 `full_reset()` 静止检测器（旧数据基准会误触发）。

#### 3.6.3 快速重评估（`fast_recheck`）

一次连拍后若共识没有变化（未切换模型），说明当前模型可能已失效/任务已结束但机械臂还静止着。此时把下一次静止触发阈值从 2s 降到 **1s**（`stationary_fast_recheck_time`），打破"静止 2s + 连拍 1s"的死循环，加快任务切换。当模型切换成功或恢复有效动作（模型输出显著偏离当前关节，`joint_delta >= 0.015 rad`）时恢复正常阈值。

#### 3.6.4 待机模式（`_handle_standby`）

subtask4（包裹在盒子）-> 进入 standby：不推理、不执行动作；静止检测照常工作，周期性触发连拍检查**场景复原**（人工放回包裹到桌面）。检测到非 standby 共识 -> 自动切出待机恢复工作。每 30s 打印一次待机日志。进入待机时清除子任务超时计时（待机不是子任务）。

#### 3.6.5 紧急停止

- `signal.SIGINT/SIGTERM` -> 全局 `_EMERGENCY_STOP` 标志 -> 主循环退出。
- `SafetyMonitor`：相邻步关节跳变 > 0.15 rad (8.6°) 或夹爪跳变 > 0.5，**连续 15 次**警告 -> 急停（连续计数避免了单次抖动误停）。

#### 3.6.6 模型输出显著性诊断（`_near_hold_count`）

模型输出与当前关节几乎一致（<0.86°）持续 100 步（约 3~4s）-> 打 warning 提示"模型在当前场景输出保持姿态"，并保存场景快照（相机图 + 关节 + 模型信息，每次运行限 3 份），用于区分「模型问题/场景分布外」还是「调度链路问题」。恢复有效动作时打印恢复日志并关闭 fast_recheck。

---

## 4. ACT 编排器：多模型切换方案

### 4.1 核心思想

每个子任务训练一个独立的 **ACT（Action Chunking Transformer）策略模型**（CNN 图像编码器 + Transformer，输出动作 chunk），VLM 调度的产物是「切换到哪个模型的权重」：

```
subtask1 -> Task_A (抓桌面包裹放托盘)   ← /model/ACT_model/model/Task_A/pretrained_model
subtask2 -> Task_B (翻转包裹)           ← .../Task_B/pretrained_model
subtask3 -> Task_C (托盘->盒子)          ← .../Task_C/pretrained_model
subtask4 -> standby (无模型)
```

ACT 模型小（数十 MB 量级），三个模型**全部预加载**驻留 GPU（`preload_all()`，可用 `--no-preload-all` 按需加载），切换只是换激活引用，无加载延迟。

### 4.2 模型管理（`act_model_manager.py`）

#### 4.2.1 加载

```python
policy = ACTPolicy.from_pretrained(model_path)
preprocess, postprocess = make_pre_post_processors(policy.config, model_path)
policy.eval(); policy.to(device); policy.reset()
```

#### 4.2.2 切换流程（`switch_model`）--"干净切换"是设计重点

```
1. 旧模型 policy.reset()                      # 清空动作队列/时序集成状态
2. 清空激活引用 (_active_model = "")
3. torch.cuda.empty_cache() + synchronize()   # 清 GPU 缓存
4. gc.collect()                               # Python GC
5. _activate_model(target)                    # 换引用 (失败则回退旧模型; 再失败兜底进 standby)
6. 检查目标模型设备 (从待机恢复时可能需要 .to(device))
7. 目标模型 policy.reset()                    # 全新状态
```

**为什么这么重**：确保每次切换都是"干净"的，无累积隐状态误差、无显存碎片。standby 进入时同理（模型保留 GPU，`standby_keep_on_gpu=True`，避免恢复时 1~2s 无动作空窗）。

#### 4.2.3 推理（`infer_step`）-- 单线程同步模式

每控制步（30Hz）：
```python
obs = {observation.state, observation.images.top, observation.images.wrist}
      # 图像 HWC -> CHW, /255, 扩 batch 维, 转 tensor
processed = preprocess(obs + task prompt)
action = policy.select_action(processed)      # torch.inference_mode()
action = postprocess(action)                  # 归一化逆变换
```

ACT 内部是**队列模式**：`_action_queue` 缓存整段 chunk（`n_action_steps` 步），每次 `select_action` 弹 1 步，队列耗尽才重新预测。`is_new_chunk()` 通过探测队列是否为空来标记 chunk 边界（供诊断/平滑用）。

#### 4.2.4 时序集成（Temporal Ensembling）与夹爪透传（可选机制）

ACT 的 chunk 边界跳变问题：相邻 chunk 在边界处首动作可能大幅跳变 -> 机械臂"小幅度上升下降循环"振荡。时序集成把每步新预测的 chunk 与前序 chunk 按指数衰减权重融合。

**但标准时序集成有一个致命副作用**：夹爪开/合是二值信号（0=闭合, ~1=打开），跨时刻平均会产生中间值（如 0.9 与 0.0 平均 -> 0.45）-> 夹爪"半开半合"，抓取不可靠。

解决方案：**`GripperPassthroughEnsembler` 包装器**--关节维度交由内层 `ACTTemporalEnsembler` 集成，夹爪维度**透传当前步原始值**：

```python
raw_gripper = actions[..., 0, gripper_idx].clone()
ensembled = self._inner.update(actions)        # 关节做时序集成
ensembled[..., gripper_idx] = raw_gripper       # 夹爪替换为原始值, 保持开/合信号清晰
```

（注：当前默认 `act_temporal_ensemble_coeff=0.0` 即禁用集成，因为最终采用对齐 HIL 的纯 EMA + 选择性限幅方案解决了抖动，见 7.1；该机制作为可选项保留。）

### 4.3 动作后处理链（编排器内）

```
策略输出 (7维: 6关节+夹爪)
  -> TrajectorySmoother.smooth()      # 逐关节 EMA + J2/J3 速率限制 + 夹爪死区
  -> _apply_hold_lock()               # 保持锁定 (抑制微振荡)
  -> SafetyMonitor.check()            # 跳变检查
  -> _send_action()                   # rad->0.001° + 关节限位 clamp + GripperFilter -> Piper SDK
```

`GripperFilter.process()`：归一化夹爪值 -> clip 到 [0,1] -> 强制闭合判定（mapped = raw×70mm×1000 < 35000 即 raw<0.5 视为噪声微颤 -> 强制闭合 0，实测是 Task_B/C 抓取时夹爪能闭合的关键）-> 可选非线性放大（默认禁用）-> 映射为 0.001mm 指令。

### 4.4 启动流程（`run_orchestrator.py`）

argparse 解析 CLI（可覆盖任何 config 字段）-> 构造 `OrchestratorConfig` -> `ACTVLMOrchestrator(config)` -> `initialize()`（相机 -> 机械臂连接/使能 -> 预加载模型 -> 初始化平滑器/过滤器）-> `run()`（首次场景评估 + 主循环）-> 结束时 `_shutdown()`（断开相机/清理模型/保存决策日志 JSON/打印统计）。

---

## 5. PI0.5 编排器：单模型任务文本切换方案

### 5.1 核心思想

PI0.5 是**多任务条件的 VLA 模型**（PaliGemma 3B 视觉语言骨干 + 流匹配动作专家）：同一份权重通过不同的 **task 文本**条件化输出不同行为。因此：

```
只加载 1 个模型: /model/Pimodel/60K/pretrained_model (batch5 合并数据集训练, 支持 3 任务)

VLM 共识切换 = 停推理线程 -> policy.reset() + 新建空 ActionQueue
             -> 更新任务标签与 task 文本 -> (移到预设位姿后) 重启推理线程
切换开销: 毫秒级 (模型常驻 GPU, 不卸载/加载权重)
```

**Task 文本必须与训练数据集的 task 逐字一致**（PI 模型对 task token 敏感，来源 `datasets/batch5{A,B,C}_joint_merged/meta/tasks.parquet`）：

```python
TASK_PROMPTS = {
    "Task_A": "Grab the package and place it on the barcode scanning table.",
    "Task_B": "Flip the package if the barcode is not facing up.",
    "Task_C": "Grab the scanned package and place it into the box.",
}
```

任务文本读写经 `threading.Lock` 保护（`_set_task_prompt` / `_get_task_prompt`），保证后台推理线程与主循环的一致性。

### 5.2 模型加载（`pi_model_manager.py::load_pi05_policy`）

checkpoint 的 `config.json` 是训练机导出的，需要三步适配才能在本机加载：

```
1. 剔除 0407 特有的 "type" 字段 (Evo-RL 的 PI05Config 无此字段)
2. input_features / output_features 的普通 dict -> PolicyFeature 对象
3. tokenizer_name 强制覆盖为本机离线路径 (/model/paligemma-3b-pt-224)
   (checkpoint 里保存的是训练机路径, 本机不存在, 不覆盖则 tokenizer 加载失败)
然后: PI05Policy.from_pretrained(ckpt_path, config=config)  # 内部已处理权重键重映射
最后: 注入 RTC 配置 (policy.config.rtc_config = ...; policy.init_rtc_processor())
      + make_pre_post_processors(..., preprocessor_overrides={tokenizer_processor: ...})
```

加载完成后做一次 **CUDA 预热**（3 次空推理，约 1~2s 的算子编译/显存分配开销），避免开局动作跳变；模型不切换，预热只做一次（`_warmup_done`）。

### 5.3 推理架构：后台线程 + ActionQueue + RTC（与 ACT 版最大的差异）

```
┌─────────── 主线程 (20Hz 控制循环) ───────────┐      ┌──── 后台线程 "PI-RTC-Inference" ────┐
│ infer_step(): 从 ActionQueue.get() 取 1 动作  │      │ while not shutdown:                │
│   队列空 -> 返回 None -> 主循环保持姿态等待    │      │   队列剩余 > 动态阈值? -> sleep 继续 │
│                                              │◀────│   捕获同步点 (idx, left_over)      │
│ 平滑 -> 安全检查 -> 发送机械臂                 │ 队列 │   取最新观测 -> 预处理               │
│                                              │ merge│   predict_action_chunk (RTC 引导)  │
└──────────────────────────────────────────────┘      │   后处理 -> queue.merge()           │
                                                      └────────────────────────────────────┘
```

**为什么用后台线程**：PI0.5 单次 chunk 推理（3B 模型 + 多步去噪）耗时可达数百 ms，远超 50ms 的控制周期。同步推理会导致控制循环断档（机械臂僵住）。后台线程把「预测」与「执行」解耦：执行端稳定 20Hz 消费队列，预测端在队列将耗尽前异步补货。

#### 5.3.1 触发时机的动态阈值（`SmartLatencyTracker`）

```python
threshold = max( ceil(预估延迟/每步时间) + execution_horizon//2,   # 动态部分
                 pi_action_queue_threshold=15 )                     # 基础阈值 (RTC 开启时)

预估延迟 = EMA(实际推理延迟) × 1.1 安全余量
  - EMA(alpha=0.15): 对偶发 Jitter 不敏感, 一次 300ms 异常不会让后续都按 300ms 预留
  - 延迟上升时临时提高 alpha 到 0.5 (快速响应变慢)
  - inference_delay_steps = max(EMA 估计, 上次实际 steps_passed + 1)
    ★ 兜底逻辑: 拼接点必须落在 RTC 硬对齐区(权重=1), 若估计<实际则对齐点偏早 -> 边界跳变
```

#### 5.3.2 推理核心（`_inference_loop` 步骤）

```
1. 捕获同步点: idx_at_start = queue.get_action_index()
               prev_chunk_left_over = queue.get_left_over()      # 上一 chunk 未执行完的原始动作
2. 获取观测:   _prepare_observation()  (关节 + 帧缓存图像, HWC->CHW /255)
               obs["task"] = [当前任务文本]                        # ★ VLM 调度的切换点
3. 模型推理:   policy.predict_action_chunk(processed_obs,
                   prev_chunk_left_over=...,                      # RTC 对齐引导
                   inference_delay=预估步数,
                   execution_horizon=16,
                   noise=零噪声张量(若确定性采样))
   注意用 torch.no_grad() 而非 inference_mode: RTC 的 guidance 内部需要 enable_grad 做 autograd.grad
4. 后处理:     postprocess -> (chunk_size, action_dim)
5. 延迟统计:   latency_tracker.update(实际耗时)
6. 计算实际消耗: steps_passed = idx_at_end - idx_at_start  (推理期间主循环吃掉了多少步)
7. 合并:       queue.merge(original_actions, postprocessed_chunk, real_delay=steps_passed, ...)
   -> RTC 模式: 整体替换队列, 丢弃前 real_delay 步 (它们对应推理期间已被执行的时间)
   -> 非RTC:   追加拼接
```

#### 5.3.3 RTC（Real-Time Chunking）原理简述（`src/lerobot/policies/rtc/`）

RTC 来自 Physical Intelligence 的 Kinetix 实现，解决「chunk 边界跳变」：新 chunk 预测完成时，机械臂已经执行了上一 chunk 的一部分，直接替换会产生不连续。RTC 在**流匹配去噪过程中**引入引导项，强制新 chunk 的前缀与上一 chunk 的剩余部分（`prev_chunk_left_over`）对齐：

```
带引导的去噪: v_t' = v_t - guidance_weight × d/dx_t [ ||(x1_t - left_over) × prefix_weights||² ]
  - x1_t = x_t - t·v_t 是当前噪声解所指向的"终点动作"
  - prefix_weights (EXP 调度): 前 inference_delay 步权重=1 (硬对齐, 必须与正在执行的
    旧动作完全一致), 之后指数衰减到 0 (新 chunk 自由发挥)
  - guidance_weight = c(τ)·inv_r², 上限 max_guidance_weight=8 (真机调优甜点)
```

真机调优结论（2026-08-25）：H=16/G=8 时 chunk 边界跳变仅 1.6x GT；H=8/G=4 时高达 8.2x GT。

**参数**（`config.py`）：`rtc_enabled=True`、`rtc_execution_horizon=16`（对齐窗口）、`rtc_max_guidance_weight=8.0`（引导强度）、`rtc_prefix_attention_schedule="EXP"`（指数衰减过渡）。

`ActionQueue`（`policies/rtc/action_queue.py`）维护两份序列：`queue`（后处理动作，供执行）与 `original_queue`（原始动作，供计算 left_over）；`get()` 返回 clone 防外部修改；`merge()` 在 RTC 模式下整体替换并跳过 `real_delay` 步。

### 5.4 采样模式：确定性 vs 流匹配（`pi_deterministic_noise`）

```python
# 确定性采样 (默认推荐): 把流匹配的初始噪声固定为零
noise = torch.zeros(bsize, chunk_size, max_action_dim)   # make_deterministic_noise()
```

| 模式 | 原理 | 抖动 | 多模态性 |
|------|------|------|---------|
| `deterministic`（默认） | 零噪声起点，去噪轨迹确定 | 实测 2.07x -> **0.86x GT**，零额外延迟 | 失去（模式平均），gripper 最大约 1.1 不到全开 1.4（对抓取影响小） |
| `flow` | 每次采样随机噪声 | 存在采样随机性抖动，需 `flow_ema_alpha=0.5` 轻量 EMA 抑制（仅此模式启用） | 保留 |

### 5.5 任务切换的完整编排（编排器侧 `_handle_consensus`）

与 ACT 版的关键差异：**先切换任务文本 + 停推理线程，移动到预设位姿后，再重启推理线程**：

```
switch_model(target, restart_thread=False):
    1. 停止后台推理线程        # 避免旧任务文本的推理写入新队列
    2. policy.reset() + 新建空 ActionQueue    # RTC 对齐归零
    3. 更新任务标签与 task 文本
    (restart_thread=False, 先不重启线程)

_handle_consensus (编排器):
    switch_model(target, restart_thread=False)
    -> _move_to_task_preset(target)             # ★ 缓慢移动到该任务的平均初始位姿
    -> model_manager.start_inference_thread()   # 基于新位姿 + 新任务文本重新预测
    -> 重置平滑器(当前关节 seed) / 静止检测器 / 保持锁定 / 保护期...
```

### 5.6 预设位姿机制（`_move_to_task_preset` + `move_to_preset_pose`）

PI0.5 对初始状态分布敏感：切换任务时机械臂可能停在上一个任务的结束位姿（训练分布外），导致模型"无动作"。解决：切换后**缓慢移动到该任务训练数据首帧的平均位姿**（读取当前关节 `GetArmJointMsgs()` -> 线性插值 100 步 × 0.02s ≈ 2s -> 到位后夹爪打开），让初始状态回到训练分布内：

```
Task_A (放到扫码台): (-0.012, 0.004, -0.004, 0.045, 0.443, 0.022)  # home/原点
Task_B (翻面):       (0.137, 1.327, -1.341, 0.172, 1.137, -0.292)  # 工作区
Task_C (放入纸箱):   (0.046, 1.113, -1.400, 0.238, 1.213, -0.242)  # 工作区
兜底:               (0.307, 1.304, -1.277, -0.048, 0.953, 0.116)
standby -> 使用 standby_preset_task(默认 Task_A) 的位姿待机
```

位姿来源：`inference_pi05_adarms_fix.py` 的 `resolve_preset_joints`（各任务训练数据首帧均值，2026-08-26 实测）。

### 5.7 PI 版特有的其它机制

1. **推理线程 Watchdog**：`infer_step()` 发现后台线程死亡（致命异常）时自动尝试重启（`_thread_warned_dead` 防日志刷屏），队列空时返回 None 让主循环保持姿态等待。
2. **夹爪闭合增强**（`gripper_gain`，仅 PI 版）：模型输出的夹爪在闭合阶段可能停在 0.05~0.25（物理上有间隙夹不紧）。**分段线性压缩**：比例 ≤0.25 时乘 gain=0.40（夹得更紧），[0.25, 0.50] 线性过渡回正常，>0.50 不压缩。映射连续无阶跃，不破坏模型闭环。
3. **相邻步跳变报警**（`jump_alert_deg=5°`）：RTC+平滑下 chunk 内相邻步应 <2°；某一步突然超阈值大概率是 chunk 拼接点对齐不准，打 warning 并附队列长度，便于事后从诊断 CSV 定位。
4. **诊断日志**（`diag_logger.py`）：两张 CSV--`pi05_diag_infer_*.csv`（推理延迟/EMA/预估与实际步数/队列/joint_span/gripper_span/归一化观测）与 `pi05_diag_exec_*.csv`（步/时刻/各关节角/夹爪/步间跳变）。用于定位三类典型问题：模型无动作（joint_span 小 + 观测偏离训练均值）、chunk 衔接跳变（step_jump 突增）、流匹配抖动（step_jump 频繁超标）。
5. **流匹配 EMA**（`flow_ema_alpha=0.5`，仅 flow 模式）：对发送动作做轻量 EMA 抑制高频抖动；确定性采样下自动不启用。

---

## 6. 两种方案深度对比

### 6.1 架构与切换机制对比

| 维度 | ACT 编排器 (v2) | PI0.5 编排器 |
|------|----------------|--------------|
| 策略骨干 | ACTPolicy（CNN 编码器 + Transformer，小模型） | PI05Policy（PaliGemma 3B VLA + 流匹配动作专家） |
| 模型数量 | **3 个独立模型**（Task_A/B/C 各一份权重） | **1 个多任务模型**（batch5 合并数据训练） |
| 任务切换动作 | 换激活模型引用 + 清 GPU 缓存 + GC（模型已预加载，秒级内完成） | 停推理线程 -> 换 task 文本 -> 重置状态 -> 重启线程（**毫秒级**） |
| 任务条件化 | 隐式（每个模型的训练数据即任务定义） | 显式 task 文本（必须与训练数据集 task 逐字一致） |
| 切换后初始化 | 无位姿管理（模型从当前观测直接预测） | **缓慢移动到任务平均初始位姿**（训练分布内的起点） |
| 推理模式 | 主循环内同步 `select_action`（~5ms，不阻塞 30Hz） | 后台线程 + ActionQueue 异步补货（数百 ms 推理不影响 20Hz 执行） |
| chunk 边界处理 | 时序集成（可选，默认关）+ EMA/限幅平滑 | **RTC 实时分块**（去噪过程引导对齐，H=16/G=8/EXP） |
| 采样确定性 | 模型本身确定性 | 双模式：零噪声确定性（默认）/ 流匹配随机（+EMA 抑制） |
| 控制频率 | 30 Hz | 20 Hz |
| 显存占用 | 3 份 ACT 权重（小） | 1 份 3B 权重（大）+ vLLM 单独进程 |
| 新任务扩展成本 | 需新训练 + 新权重 + 显存线性增长 | 只需在合并数据集上微调 + 登记一条 task 文本 |
| 任务间技能干扰 | 无（权重物理隔离） | 存在（共享权重，依赖多任务训练的泛化） |
| 待机实现 | 卸载激活引用（模型保留 GPU） | 停推理线程（模型保留 GPU，移到 standby 位姿） |

### 6.2 调度层完全一致的部分

两套编排器以下机制**逐行同构**（只差 import 路径）：VLM prompt 与三级容错解析、5/3 窗口投票、v3 两段式静止检测、VLM 连拍（5 帧/0.1s 间隔/5s 总超时/运动中止）、子任务 30s 超时、切换 3s 保护期、fast_recheck 快速重评估、待机模式、SafetyMonitor 急停、保持锁定、轨迹平滑器（同一套参数：J2 α=0.12/J3 α=0.18 逐关节 EMA + J2/J3 0.015 rad/帧限速 + 夹爪死区 0.15）、模型输出显著性诊断。

**这验证了框架设计的可移植性**：调度层与策略层解耦，换策略骨干不需要动调度算法。

### 6.3 各自的适配要点

| 问题 | ACT 版的答案 | PI 版的答案 |
|------|-------------|------------|
| 策略不知道任务结束 | 静止检测 + VLM 重评估 | 同左（共用） |
| chunk 边界抖动 | 对齐 HIL 采集的纯 EMA + 选择性限幅（放弃复杂过渡机制） | RTC 引导对齐 + 零噪声确定性采样 |
| 初始状态分布外 | -（ACT 对此不敏感，从任意位姿都能收敛） | 预设位姿移动（训练首帧均值） |
| 夹爪信号弱 | GripperFilter 强制闭合（阈值 35000） | 同左 + 分段增益压缩（gripper_gain） |
| 机械臂响应滞后 | MotionCtrl_2(100%, 0xAD MIT 高跟随) | 同左（都是对齐 HIL 采集参数的结论） |

---

## 7. 底层执行链路的工程细节（两者共用）

### 7.1 轨迹平滑器（`TrajectorySmoother`，对齐 HIL 采集策略 S0）

```
流水线: 逐关节 EMA 低通 -> 选择性速率限制 (仅 J2/J3) -> 夹爪死区
参数:   ema_alphas = [0.30, 0.12, 0.18, 0.30, 0.30, 0.30, 0.50]   # J1..J6 + gripper
        rate_limits = [None, 0.015, 0.015, None, None, None, None] # 仅 J2/J3 限 0.86°/帧
        gripper_dead_zone = 0.15                                    # 预测值低于此强制置零
        switch_transition_duration_s = 1.0                         # 切换后过渡期收紧平滑
```

- J2/J3 是模型预测质量最差的两个关节，超强调平滑（α=0.12/0.18 + 限速）；夹爪 α=0.50 响应更快保证抓取及时。
- **演进史（v5->v9）**：早期版本有窗口平均/全局增量限幅/chunk 边界过渡（限幅追赶/锚定）等复杂机制，结果发现这些机制本身就是抖动来源（平滑限幅过紧 -> 机械臂在 chunk 边界追赶滞后 -> 观测落后于轨迹 -> 模型重新预测时 chunk 起点前跳 -> 正反馈放大，实测观测同步时边界跳变仅 1.9°、失同步时高达 49°）。**v9 完全对齐 HIL 采集（recording_loop.py）的平滑思路**：J1/J4/J5/J6 不限幅，EMA 直通，机械臂紧跟 chunk，观测保持同步，反而无抖动。

### 7.2 保持锁定（`_apply_hold_lock`，抑制微振荡）

背景：模型"保持位姿"时每次 chunk 重预测的目标会漂移（实测 ~0.5~2°），机械臂跟随产生小幅振荡，且振荡使静止检测永不触发 -> 切换流程卡死。

```
双条件判别 (缺一不可):
  - raw_drift:  原始目标相对保持窗口参考位移 < 0.02 rad (1.15°)  -> 模型目标真的稳定 (真·保持)
  - hold_delta: 原始目标与当前关节差异 < 0.06 rad (3.4°)        -> 机械臂已接近目标 (容忍伺服滞后)
双条件连续满足 45 步 (~1.6s) -> 锁定关节到当前平滑位置 (夹爪不锁)
锁定期: 模型目标重新漂移 (≥2×带) 或脱离机械臂 (≥2×带) -> 解除锁定跟随真实运动
```

锁定让机械臂真正静止 -> 静止检测得以触发 -> "静止 -> VLM 重评估 -> 切换"流程恢复。

### 7.3 相机管理（`camera_manager.py`，v2.0 统一帧缓存）

后台线程是**唯一的相机读取者**（30Hz 轮询双相机），最新帧 + 时间戳写入带锁缓存；ACT/PI/VLM 统一从 `get_latest_frames()` 取帧。解决多方竞争同一 `VideoCapture` 导致的帧错位/抖动。帧新鲜度检查（>0.5s 视为过期，跳过本步）应对相机流卡顿。

### 7.4 机器人接口与动作发送（`_send_action`）

```
关节: rad -> 0.001° (×57324.84) -> clamp 到 JOINT_LIMITS 关节限位表 -> JointCtrl(*cmds)
      (关节4 额外限位保护 -70000)
夹爪: 归一化 -> (PI版: 分段增益压缩) -> GripperFilter -> GripperCtrl(cmd, 1000, 0x01, 0)
运动模式: MotionCtrl_2(0x01, 0x01, 100, 0xAD)  # 100% 速度 + MIT 高跟随
机械臂: Piper 6 自由度 (CAN 总线 can1), lerobot.robots.single_piper
```

关键经验（两者一致）：**运动驱动参数必须与 HIL 采集时一致**（采集从臂为 speed_ratio=100 + high_follow=0xAD）。早期用 (30%, 0x00) 导致机械臂响应慢/位置滞后 -> RTC 的 `prev_chunk_left_over` 与实际位置错位 -> chunk 切换抖动。对齐后问题消失。

机械臂通信异常按"瞬时异常"处理：记录 warning 但不中断循环（Piper SDK 的 CAN 偶发丢包可自恢复）。

---

## 8. 关键技术问题与解决经验（实战记录）

以下问题均在真机调试中实际遇到并解决，是这套系统最有分享价值的部分：

| # | 问题 | 根因 | 解决 | 涉及机制 |
|---|------|------|------|---------|
| 1 | 启动时机械臂做错误动作 | 单次 VLM 判断无法达成 5/3 共识，默认模型先执行 | 启动即连拍投票，共识前暂停策略推理 | 首次场景评估 |
| 2 | 机械臂还在运动却触发切换 | v1 静止判据过松（等效 0.44 rad/s） | v3 两段式判定 + 连拍期间运动中止 | StationaryDetector v3 |
| 3 | 静止检测永不触发，切换卡死 | 伺服保持抖动反复打断计时；保持时目标漂移 0.5~2° 造成振荡 | 保持带 + 参考重锁 + 保持锁定 | StationaryDetector / HoldLock |
| 4 | 每 1~2s 周期性抖动一次 | 连拍结束后立即再触发连拍 | `full_reset(keep_cooldown=True)` 保留 5s 冷却 | Burst 收尾 |
| 5 | "先抬升再恢复"的怪异动作 | 连拍暂停后回放旧观测预测的陈旧 chunk | 连拍结束 `reset_active_policy()` 清空动作队列 | Burst 收尾 |
| 6 | 切换后夹爪被拉向闭合 | 平滑器基线固定用 0.0 初始化 | 用真实关节+夹爪观测 seed 平滑器基线 | 共识处理 |
| 7 | chunk 边界抖动（ACT） | 过度平滑使观测-动作失同步，正反馈放大（1.9° vs 49°） | v9 对齐 HIL：EMA 直通 + 仅 J2/J3 限速；驱动参数 (100, 0xAD) | TrajectorySmoother |
| 8 | chunk 边界跳变（PI） | 新旧 chunk 拼接点不连续；推理延迟估计偏低 | RTC 引导对齐 H=16/G=8；延迟估计 max(EMA, 上次实际+1) | RTC / SmartLatencyTracker |
| 9 | PI 流匹配高频抖动 | 每次采样噪声随机，chunk 间不连续 | 零噪声确定性采样（2.07x->0.86x GT），flow 模式下加 EMA | pi_deterministic_noise |
| 10 | PI 切换任务后无动作 | 初始位姿偏离训练分布 | 切换后移到任务平均初始位姿（训练首帧均值） | 预设位姿机制 |
| 11 | 夹爪抓不住/半开半合 | 时序集成平均夹爪信号；模型输出停在 0.05~0.25 | 夹爪透传 + 强制闭合阈值 + (PI)分段增益 | GripperPassthrough / GripperFilter |
| 12 | VLM 输出自相矛盾 | subtask 字段与位置/可见性字段不一致 | 确定性映射推导覆盖 subtask | 三级容错解析 |
| 13 | 死循环"静止->连拍->不变" | 共识未变化时仍按正常阈值等 2s | fast_recheck 降到 1s 加速重评估 | 快速重评估 |
| 14 | 子任务卡死但关节在动 | 循环运动使静止检测永不触发 | 30s 子任务超时强制重评估（不运动中止 + 静置 1s） | subtask_timeout |
| 15 | 任务切换震荡（刚切又切） | 新模型还没执行就被静止检测打断 | 3s 切换保护期 | switch_cooldown |

---

## 9. 相关文件索引

### 9.1 两套编排器（本文主角）

```
src/lerobot/
├── act_vlm_orchestrator_v2/            # ACT + VLM 编排器
│   ├── orchestrator.py                 # 主循环状态机 (ACTVLMOrchestrator)
│   ├── act_model_manager.py            # 多 ACT 模型加载/切换/推理 + 时序集成包装器
│   ├── vlm_client.py                   # VLM 客户端 (prompt + 三级容错解析)
│   ├── window_voter.py                 # 窗口投票
│   ├── stationary_detector.py          # 静止检测 v3
│   ├── camera_manager.py               # 相机统一帧缓存
│   ├── config.py                       # 全部配置
│   ├── run_orchestrator.py             # CLI 入口
│   ├── start_orchestrator.sh           # 一键启动
│   ├── test_vlm_only.py                # VLM 独立测试 (不控制机械臂)
│   └── README.md
├── pi05_vlm_orchestrator/              # PI0.5 + VLM 编排器
│   ├── orchestrator.py                 # 主循环状态机 (PIVLMOrchestrator)
│   ├── pi_model_manager.py             # 单模型管理 + 后台推理线程 + RTC (PIModelManager)
│   ├── vlm_client.py / window_voter.py / stationary_detector.py / camera_manager.py  # 同构
│   ├── diag_logger.py                  # 推理/执行诊断 CSV
│   ├── config.py                       # 配置 (含 TASK_PROMPTS / PRESET_POSES / RTC 参数)
│   ├── run_orchestrator.py / start_orchestrator.sh / test_vlm_only.py / README.md
│   └── test_single_act/                # 单任务 ACT 推理回归测试
└── act_vlm_orchestrator_v2_bp/         # (备份分支, 早期版本)
```

### 9.2 依赖的项目文件

| 文件 | 用途 |
|------|------|
| `src/lerobot/policies/act/modeling_act.py` | `ACTPolicy` / `ACTTemporalEnsembler`（ACT 策略实现） |
| `src/lerobot/policies/pi05/modeling_pi05.py` | `PI05Policy`（`predict_action_chunk` 含 RTC 引导入口） |
| `src/lerobot/policies/pi05/configuration_pi05.py` | `PI05Config` |
| `src/lerobot/policies/rtc/modeling_rtc.py` | `RTCProcessor`（带引导去噪 + prefix 权重调度） |
| `src/lerobot/policies/rtc/action_queue.py` | `ActionQueue`（双序列队列 + merge） |
| `src/lerobot/policies/rtc/configuration_rtc.py` | `RTCConfig` / `RTCAttentionSchedule` |
| `src/lerobot/policies/factory.py` | `make_pre_post_processors`（前后处理器构造） |
| `src/lerobot/robots/single_piper/` | Piper 机械臂接口（`SinglePiperConfig`、CAN 通信） |
| `src/lerobot/configs/types.py` | `FeatureType` / `PolicyFeature` / `RTCAttentionSchedule` |
| `src/lerobot/inference_pi05_adarms_fix.py` | PI05 真机推理参考实现（编排器推理链路的对齐来源：RTC 调优参数、预设位姿、DiagLogger、运动参数） |
| `src/lerobot/inference_new_joint_act.py` 等 | ACT 真机推理参考实现（关节限位表、GripperFilter、SafetyMonitor 的移植来源） |
| `datasets/batch5{A,B,C}_joint_merged/meta/tasks.parquet` | 训练任务文本来源（TASK_PROMPTS 必须逐字一致） |
| `act_vlm_logs/` | 运行日志（orchestrator_*.log + vlm_decisions_*.json 决策历史） |

### 9.3 外部依赖

| 组件 | 说明 |
|------|------|
| vLLM 服务 | 承载微调后的 Qwen3-VL-2B（`--gpu-memory-utilization 0.5 --max-model-len 8192 --served-model-name qwen3-vl-2b`），OpenAI 兼容 API |
| Piper SDK | `MotionCtrl_2` / `JointCtrl` / `GripperCtrl` / `GetArmJointMsgs`（CAN 总线） |
| OpenCV | 相机采集（`cv2.VideoCapture`，top ID=10 / wrist ID=4，640×480@30fps） |

---

## 附：分享要点提炼（一页总结）

1. **分层解耦**：VLM 管"做什么"（场景 -> 子任务），策略模型管"怎么做"（观测 -> 动作），调度层与策略层通过 `switch_model(task)` 一个接口解耦，ACT/PI0.5 两套实现共享同一调度算法。
2. **共识而非单帧**：5/3 窗口投票 + 静止触发的连拍，用约 1s 的决策延迟换取一个数量级的误判率下降；解析端再加"确定性映射覆盖"，消灭 VLM 自相矛盾。
3. **运动期间零干预**：所有 VLM 重评估都由"静止 ≥2s"或"子任务超时 30s"触发，切换永远不在运动中进行。
4. **事件驱动 + 多重保险**：静止检测（任务完成/卡死）、子任务超时（循环卡死）、fast_recheck（死循环加速）、切换保护期（震荡抑制）、待机轮询（场景复原）层层兜底。
5. **两种任务切换范式的取舍**：多模型（技能隔离好、扩展贵、切换慢）vs 单模型多任务文本（切换毫秒级、扩展易、有技能干扰、需预设位姿归位）。
6. **chunk 边界是具身控制的核心难题**：ACT 用"对齐 HIL 的简单平滑"胜过"复杂过渡机制"（观测-动作同步性 > 平滑强度）；PI0.5 用 RTC 在去噪过程中做引导对齐 + 确定性采样。
7. **真机工程=补偿物理世界的不完美**：伺服抖动（保持带+保持锁定）、夹爪间隙（强制闭合+分段增益）、机械臂响应滞后（驱动参数对齐 HIL）。








