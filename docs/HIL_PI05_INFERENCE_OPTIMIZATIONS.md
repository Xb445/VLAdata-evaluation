# 人类在环 (HIL) 数据采集中 PI0.5 推理优化策略汇总

> 更新日期: 2026-08-28
> 适用范围: `lerobot-human-inloop-record` 流程（`run_hil_record.sh` 启动），PI0.5 策略模式（S0）
> 实测状态: 三个子任务真机验证，执行相对稳定 ✅

---

## 一、HIL 采集流程与 PI0.5 推理路径

```
run_hil_record.sh
  └─ lerobot-human-inloop-record (入口: lerobot_human_inloop_record.py)
       └─ record() (lerobot_record.py)
            ├─ [每集前] 预设位姿移动  ← 优化①
            ├─ record_loop() (recording_loop.py)  核心循环
            │    ├─ PI0.5 推理: policy.select_action()  (单步, 内部 50 步 chunk 队列, 无 RTC)
            │    ├─ S0 策略动作 → 逐关节 EMA + 限速  ← 优化②
            │    ├─ PI0.5 夹爪二值滞回            ← 优化③
            │    ├─ S1 人类介入 → 主臂遥操 + 轻量 EMA  ← 优化④
            │    └─ 干预状态机 S0/S1/S2 切换
            └─ [集后] 失败复位位姿移动  ← 优化⑤
```

**PI0.5 推理要点**:
- 走 `policy.select_action()`（`n_action_steps=50`，内部 `_action_queue` 每帧弹 1 步，无 RTC 引导）
- checkpoint 的 `policy_preprocessor.json` 保存训练机 tokenizer 路径 → 需本地化覆盖（优化⑥）
- 模型: 60K checkpoint（`model/Pimodel/60K/pretrained_model`），任务文本经 tokenizer 条件化动作

---

## 二、优化策略清单

### ① 每集录制前预设位姿（训练初始分布对齐）
| 项 | 值/说明 |
|---|---|
| 文件 | `lerobot_record.py` → `record()` 每集 `record_loop` 前 |
| 配置 | `--preset_pose`(默认 true) / `--preset_pose_joints` / `--preset_pose_gripper`=1.0 / `--preset_pose_steps`=100 / `--preset_pose_step_dt`=0.02 |
| 位姿 | `_resolve_preset_joints()`: 任务文本含 `barcode scanning table`→home(A) / `Flip`→(B) `(0.341,1.188,-1.411,-0.003,1.166,-0.002)` / `box`→(C) `(0.174,1.355,-1.452,0.098,1.188,-0.052)`；否则用 `--preset_pose_joints` 兜底 `(0.307,1.304,-1.277,-0.048,0.953,0.116)` |
| 方式 | 读当前关节 → 线性插值 100 步×20ms≈2s → 经 `policy_sync_executor` 广播从臂+主臂镜像 → 到位打开夹爪 |
| 作用 | 初始状态落在训练分布内，避免 S0 收到分布外状态导致行为不稳；每集起点一致 |
| 门控 | `policy is not None`（纯遥操采集不触发） |

### ② S0 策略动作逐关节 EMA + 限速
| 项 | 值/说明 |
|---|---|
| 文件 | `recording_loop.py` → `_POLICY_EMA_PER_JOINT` / `_POLICY_RATE_LIMIT_PER_JOINT` |
| EMA α | J1=0.30, **J2=0.12**, **J3=0.18**, J4/J5/J6=0.30, gripper=0.50 |
| 限速 | J2/J3 = 0.015 rad/帧（≈0.86°/帧，防突跳/反转） |
| 作用 | 低通滤除扩散模型的逐帧高频噪声；J2/J3 模型学得差 → 加强平滑 + 限速 |
| 门控 | 仅 S0 策略动作（`act_processed_policy is not None`）；S1 遥操不受影响 |

### ③ PI0.5 夹爪二值滞回 + 驻留（★ 本次集成核心）
| 项 | 值/说明 |
|---|---|
| 文件 | `recording_loop.py` → `PI05_GRIPPER_*` 常量 + 状态机 |
| 阈值 | 开 `>0.55` / 闭 `<0.30` / 中间保持上一状态 |
| 驻留 | 状态切换后至少驻留 `40` 步（≈1.3s @30fps）防扇动 |
| 输出 | 闭合 → 0.0（完全闭合，跳过平滑/EMA 的拖慢）；打开 → 1.0 |
| 复位 | S0→S1 与 S1→S0 时复位为 open（配合预设位姿已打开夹爪） |
| 作用 | 解决 PI0.5 夹爪意图在 chunk 内全开↔全闭往返（极差>0.8）导致的"夹不到/闭合慢" |
| 门控 | `is_pi05_policy`（仅 PI05）+ `not is_intervention`（S1 人类介入不受影响）；ACT 策略完全直通 |

### ④ S1 人类介入遥操轻量 EMA
| 项 | 值/说明 |
|---|---|
| 文件 | `recording_loop.py` → `TELEOP_EMA_ALPHA` |
| 参数 | `0.7`（延迟≈1帧 @30Hz，人无感知）；夹爪不参与 EMA（即时响应） |
| 作用 | 主臂位置反馈含 0.001° 量化噪声 + 机械微振，透传会让从臂轻微抖动/顿挫；轻量低通滤高频微抖 |
| 门控 | 仅 S1 介入分支（`intervention_enabled` 需 policy+teleop 同时存在） |

### ⑤ 失败/结束复位位姿
| 项 | 值/说明 |
|---|---|
| 文件 | `lerobot_human_inloop_record.py` → `_HumanInloopFailureResetController` |
| 机制 | 首次运行按 ENTER 捕获当前位姿存 JSON；每集结束（成功/失败）缓慢复位双臂到该位姿（`_slow_reset_all_arms_to_pose`，3s 插值，主臂同步） |
| 作用 | 每集从一致位姿开始，数据集更干净 |

### ⑥ Tokenizer 本地路径修复（HIL 启动的前提）
| 项 | 值/说明 |
|---|---|
| 文件 | `lerobot_record.py` → `make_pre_post_processors` override |
| 内容 | `tokenizer_processor.tokenizer_name` 覆盖为 `cfg.policy.tokenizer_name`（本机 `/home/ub/Videos/Evo-RL/model/paligemma-3b-pt-224`），门控 `cfg.policy is not None` |
| 作用 | checkpoint 的 `policy_preprocessor.json` 保存训练机路径 `/data/zhanghang/...`，不覆盖会 `HFValidationError` 崩溃 |

### ⑦ 策略-主臂同步镜像（policy_sync_to_teleop）
| 项 | 值/说明 |
|---|---|
| 文件 | `lerobot_record.py` → `PolicySyncDualArmExecutor`；`run_hil_record.sh` |
| 机制 | S0 模式同一动作并行广播给从臂 `robot.send_action` + 主臂 `teleop.send_feedback`（主臂位置跟随模式） |
| 作用 | 主臂镜像从臂运动，S0↔S1 切换无跳变；干预时切换主臂到重力补偿可拖动 |

### ⑧ 干预状态机 S0/S1/S2（人机协同框架）
| 项 | 值/说明 |
|---|---|
| 文件 | `recording_loop.py` / `lerobot_human_inloop_record.py` |
| 键位 | `I` 切换 / `s` 成功 / `f` 失败 / `→` 跳过 / `ESC` 停止 |
| 切换 | S0(策略) →按I→ S1(示教, 主臂重力补偿可拖动) →按I→ S2(释放) → S0 |
| 复位 | 切换时 `policy.reset()` + 清空 EMA/滞回状态 + 主臂模式切换 |
| 数据 | 记录 `complementary_info.policy_action`(原始策略) / `is_intervention` / `collector_policy_id` |

---

## 三、关键参数速查表

| 参数 | 位置 | 默认值 | 作用 |
|---|---|---|---|
| `preset_pose` | lerobot_record.py | true | 每集前移预设位姿 |
| `preset_pose_joints` | lerobot_record.py | 训练均值兜底 | 预设位姿 6 关节 rad |
| `PI05_GRIPPER_BINARY_ENABLED` | recording_loop.py | true | 夹爪二值滞回总开关 |
| `PI05_GRIPPER_OPEN_ABOVE` | recording_loop.py | 0.55 | 夹爪开判定阈值 |
| `PI05_GRIPPER_CLOSE_BELOW` | recording_loop.py | 0.30 | 夹爪闭判定阈值 |
| `PI05_GRIPPER_MIN_DWELL` | recording_loop.py | 40 | 状态切换最小驻留步数 |
| `POLICY_EMA_PER_JOINT[J2/J3]` | recording_loop.py | 0.12/0.18 | 薄弱关节加强平滑 |
| `POLICY_RATE_LIMIT_PER_JOINT` | recording_loop.py | J2/J3=0.015 | 限速防突跳 |
| `TELEOP_EMA_ALPHA` | recording_loop.py | 0.7 | S1 遥操防抖 |
| `GRIPPER_DEAD_ZONE` | recording_loop.py | 0.0 | 夹爪死区（ACT 常用 0.57，PI05 用滞回替代） |
| `CAM_WRIST_INDEX/CAM_TOP_INDEX` | run_hil_record.sh | 10 / 4 | 相机映射（本机实测） |
| `TASK_DESC` | run_hil_record.sh | 随任务 | 任务文本（与训练 A/B/C 一致时自动匹配预设位姿） |

---

## 四、调优建议（若需调整）

1. **夹爪仍轻微扇动** → 收紧滞回带：`PI05_GRIPPER_OPEN_ABOVE` 0.55→0.65、`CLOSE_BELOW` 0.30→0.25；
2. **抓/放切换有延迟感** → 减小驻留：`PI05_GRIPPER_MIN_DWELL` 40→25；
3. **S0 仍有轻微抖动** → J2/J3 限速 `0.015`→`0.010`，或 α 再降；
4. **S1 手感偏"黏"** → `TELEOP_EMA_ALPHA` 0.7→0.85（或 1.0 关闭）；
5. **新任务采集** → 用 `--dataset.single_task="新任务描述" --preset_pose_joints="j0,j1,j2,j3,j4,j5"` 指定初始位姿（任务文本不匹配 A/B/C 时用兜底位姿）。

## 五、与 flowmatch 推理脚本的差异说明（HIL 专属适配）

| 优化 | flowmatch 推理脚本 | HIL 数据采集 |
|---|---|---|
| RTC 多线程引导 | ✅ 启用（H=16, G=8） | ❌ 不适用（`select_action` 无 RTC，单线程循环） |
| 夹爪二值滞回 | ✅ | ✅（已移植） |
| 预设位姿 | ✅ 启动前一次 | ✅ 每集前（任务文本自动匹配） |
| 轨迹平滑 | 限幅裁剪式 | 逐关节 EMA+限速（HIL 风格） |
| 安全监控/跳变报警 | ✅ | ❌ 未移植（保持 HIL 原逻辑） |

## 六、HIL 逐关节 EMA + 限速 → flowmatch 同步（2026-08-28）

真机对比发现 HIL 采集的 PI05 推理效果优于 flowmatch 直接推理，核心差异在关节平滑：
HIL 对模型学得差的 **J2/J3** 用更低 EMA α（0.12/0.18）+ 显式限速 0.015 rad/帧强力抑制突跳/反转，
flowmatch 原平滑对所有关节一视同仁。

**已同步到 `inference_pi05_flowmatch.py`**（新增配置，默认开启）：
- `--hil_smooth_enabled=true`：把 9e.0 的单α流匹配 EMA 替换为 HIL 逐关节 EMA + J2/J3 限速
- `--hil_ema_alphas="0.30,0.12,0.18,0.30,0.30,0.30"`（J1..J6，对齐 HIL）
- `--hil_rate_limits="0.0,0.015,0.015,0.0,0.0,0.0"`（J1..J6 rad/帧，0=不限）
- 保留 9c 平滑器的硬限幅作为底层安全；夹爪仍走二值滞回
- 仿真验证：J2 突跳 0→0.2 rad 被限速到 0.015 rad/帧（冲击分散 ~13 帧）

> ⚠ 注意：flowmatch 的 J2/J3 限速(0.015) 与 HIL 完全一致，但 flowmatch 还有 RTC 对齐 + 硬限幅，
> 若叠加后感觉动作偏"粘"，可调大 `hil_ema_alphas` 或调小 `hil_rate_limits`。

