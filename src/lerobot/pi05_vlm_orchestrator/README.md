# PI0.5 + VLM 多任务联调编排器（单模型多任务）

## 概述

基于 VLM (Qwen3-VL-2B) 大语言模型调度 **同一个 PI0.5 模型** 在多个任务之间切换。

PI0.5 是支持多任务条件的 VLA 模型：同一份权重通过不同的 **task 文本**（任务描述）条件化输出不同行为。
因此编排器只加载 **ONE 个** PI 模型，VLM 判定场景变化时仅切换 task 文本 + 重置推理状态，
不涉及模型卸载/加载，切换开销极小（毫秒级）。

### 架构

```
┌─────────────────────────────────────────────────────────────┐
│                   PI-VLM Orchestrator                       │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Camera   │   │   VLM    │   │  Window  │   │   PI0.5  │ │
│  │  Manager  │──▶│  Client  │──▶│  Voter   │──▶│  Model   │ │
│  │           │   │ (vLLM)   │   │ (5帧≥3)  │   │ (单模型) │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                      │       │
│  ┌──────────┐   ┌──────────┐                         │       │
│  │Stationary│   │  Safety  │                         ▼       │
│  │ Detector │──▶│  Monitor │◀───────────  Piper 机械臂      │
│  │ (2~3s静止)│   │          │    (后台推理线程 + ActionQueue) │
│  └──────────┘   └──────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

### 场景分类 → 任务文本映射

| VLM 分类 | 场景描述 | 任务标签 | 切换到的 task 文本（与训练数据集逐字一致） |
|---------|---------|---------|----------------------------------------|
| subtask1 | 包裹在桌面 | Task_A | `Grab the package and place it on the barcode scanning table.` |
| subtask2 | 包裹在托盘，条码不可见 | Task_B | `Flip the package if the barcode is not facing up.` |
| subtask3 | 包裹在托盘，条码可见 | Task_C | `Grab the scanned package and place it into the box.` |
| subtask4 | 包裹在盒子 | standby | — (待机等待场景复原) |

### 核心机制

1. **单模型**：只加载一个 PI 模型（默认 `/home/ub/Videos/Evo-RL/model/Pimodel/60K/pretrained_model`，
   该模型在 batch5 合并数据集上训练，支持 3 个任务）
2. **窗口投票**：5 次 VLM 判断中 ≥3 次一致才采纳结果，防止模型误判
3. **静止检测（保险机制）**：机械臂接近静止 2~3s 时自动触发 VLM 重评估
4. **任务切换 + 预设位姿**：VLM 共识变化 → 停推理线程 → 切换 task 文本 → **缓慢移动到该任务的
   平均初始位姿（训练 state 均值，见下表）** → 重启推理线程。模型常驻 GPU，切换毫秒级完成
5. **待机位姿**：包裹在盒子（subtask4/standby）时在 Task_A 的初始位姿待机，等待场景复原
6. **PI 后台推理 + RTC**：后台线程监控 ActionQueue，队列将耗尽时基于最新观测预测 action chunk，
   通过 RTC (Real-Time Chunking) 对齐上一 chunk 残余动作，抑制边界跳变
   （架构与 `inference_pi05_adarms_fix.py` 完全一致）
7. **两种采样推理**：`deterministic`（零噪声确定性采样，默认推荐）或 `flow`（流匹配去噪随机采样），
   通过 `--inference-mode` 选择

### 各任务平均初始位姿（6 关节 rad）

来源: 2026-08-27 重新统计 batch5A/B/C 各任务首帧 `observation.state` 的均值
（此前 2026-08-26 实测位姿对 Task B/C 匹配较差，j1/j3/j5 等偏离训练分布）。
切换任务后机械臂缓慢移动到对应位姿，让初始状态落在训练分布内
（可用 `--preset-joints-a/b/c` 覆盖）。

| 任务 | 平均初始位姿 (rad) |
|------|-------------------|
| Task_A (放到扫码台) | `(-0.012, 0.004, -0.004, 0.045, 0.443, 0.022)` — home/原点 |
| Task_B (翻面) | `(0.341, 1.188, -1.411, -0.003, 1.166, -0.002)` — 训练初始均值 |
| Task_C (放入纸箱) | `(0.174, 1.355, -1.452, 0.098, 1.188, -0.052)` — 训练初始均值 |
| 兜底 | `(0.307, 1.304, -1.277, -0.048, 0.953, 0.116)` |

## 目录结构

```
pi05_vlm_orchestrator/
├── __init__.py              # 包初始化
├── config.py                # 配置数据类 (模型路径/任务文本/RTC/投票/平滑)
├── vlm_client.py            # VLM API 客户端 (vLLM)
├── window_voter.py          # 窗口投票机制
├── pi_model_manager.py      # PI0.5 单模型管理与后台 RTC 推理引擎 (task 文本切换)
├── stationary_detector.py   # 机械臂静止检测
├── camera_manager.py        # 相机管理
├── orchestrator.py          # 主编排器
├── run_orchestrator.py      # 启动入口
├── test_vlm_only.py         # VLM 独立测试
├── start_orchestrator.sh    # 一键启动脚本
└── README.md                # 本文件
```

## 前置条件

### 1. 启动 vLLM 服务

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
  --model /home/ub/Videos/0407/model/qwen_model/Qwen3-VL-2B-Instruct-abc-v3 \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 8192 --max-num-seqs 4 \
  --dtype auto --trust-remote-code \
  --served-model-name qwen3-vl-2b
```

### 2. 确保 PI0.5 模型就绪（单个模型，支持 3 任务）

```
/home/ub/Videos/Evo-RL/model/Pimodel/60K/pretrained_model/
```

- 默认路径可用 `--model` 覆盖
- 模型加载时会做兼容适配：剔除 `type` 字段、特征转 `PolicyFeature`、
  tokenizer 覆盖为本机离线路径 `/home/ub/Videos/Evo-RL/model/paligemma-3b-pt-224`

### 3. 确保硬件连接

- Piper 机械臂（CAN 总线）
- 顶部相机（ID=10）
- 腕部相机（ID=4）

## 使用方法

### 一键启动

```bash
cd /home/ub/Videos/Evo-RL/src
./lerobot/pi05_vlm_orchestrator/start_orchestrator.sh
```

### 手动启动

```bash
cd /home/ub/Videos/Evo-RL

# 默认配置 (模型路径已指向本机 60K 模型)
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator

# 指定 PI 模型路径 (单模型, task 文本切换任务)
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator \
  --model /path/to/pretrained_model

# 指定默认任务
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --default-task Task_B

# 调试模式
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --debug

# CPU 推理 (调试用)
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --device cpu

# 干跑模式 (安全验证状态机: 连接+读取机械臂, 但不发送动作指令)
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --dry-run

# RTC 参数
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator \
  --rtc-horizon 16 --rtc-guidance 8.0 --rtc-schedule EXP

# 关闭 RTC (队列模式)
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --no-rtc

# 推理采样模式: 确定性采样 (默认) / 流匹配去噪 (随机)
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --inference-mode deterministic
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --inference-mode flow

# 关闭任务切换后的预设位姿移动
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --no-preset-pose

# 自定义各任务平均初始位姿 (6 关节 rad)
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator \
  --preset-joints-a 0.11 1.13 -1.10 -0.03 0.84 0.10 \
  --preset-joints-b 0.43 1.47 -1.42 -0.09 1.06 0.12

# 自定义投票参数
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --vote-window 7 --vote-threshold 5

# 禁用轨迹平滑
python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --no-smooth
```

### VLM 独立测试（不控制机械臂）

```bash
# 单次实时相机测试
python -m lerobot.pi05_vlm_orchestrator.test_vlm_only --use-cameras

# 持续监控模式
python -m lerobot.pi05_vlm_orchestrator.test_vlm_only --use-cameras --continuous

# 使用静态图片测试
python -m lerobot.pi05_vlm_orchestrator.test_vlm_only \
  --image-top /path/to/top.jpg \
  --image-wrist /path/to/wrist.jpg
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--vlm-url` | `http://localhost:8000/v1` | vLLM API 地址 |
| `--vlm-model` | `qwen3-vl-2b` | vLLM 模型名 |
| `--model` | `/home/ub/Videos/Evo-RL/model/Pimodel/60K/pretrained_model` | PI0.5 模型路径（单模型） |
| `--default-task` | `Task_A` | 默认任务标签 (Task_A/B/C) |
| `--rtc-enabled/--no-rtc` | 启用 | RTC 开关 |
| `--rtc-horizon` | 16 | RTC 对齐窗口 (真机调优: 16 优于 8) |
| `--rtc-guidance` | 8.0 | RTC 引导强度 (真机调优: 8 为甜点) |
| `--rtc-schedule` | `EXP` | RTC 前缀注意力调度 (EXP/LINEAR/CONST) |
| `--deterministic-noise/--no-deterministic-noise` | 启用 | 零噪声确定性采样 (等价 `--inference-mode deterministic/flow`) |
| `--inference-mode` | `deterministic` | 推理采样模式: `deterministic`=确定性采样(零噪声) / `flow`=流匹配去噪(随机) |
| `--queue-threshold` | 15 | 触发新推理的队列剩余阈值 |
| `--preset-pose/--no-preset-pose` | 启用 | 任务切换后移动到任务平均初始位姿 |
| `--preset-steps` | 100 | 预设位姿移动插值步数 (100 步 ≈ 2s) |
| `--preset-step-dt` | 0.02 | 预设位姿移动每步间隔 (秒) |
| `--preset-gripper` | 1.0 | 预设位姿夹爪开度 (0~1) |
| `--preset-joints-a/b/c` | 训练均值 | 各任务平均初始位姿 (6 关节 rad) |
| `--standby-preset-task` | `Task_A` | 待机 (包裹在盒子) 时使用的初始位姿对应任务 |
| `--preset-fallback` | 训练均值 | 任务匹配失败时的兜底位姿 (6 关节 rad) |
| `--motion-speed` | 100 | 运动速度百分比 (对齐 HIL 采集) |
| `--motion-high-follow/--no-motion-high-follow` | 启用 | MIT/高跟随模式 (mit_mode=0xAD) |
| `--flow-ema-alpha` | 0.2 | 流匹配去噪模式的轻量 EMA 系数 (0 禁用, 仅关节, 确定性采样下自动不启用) |
| `--gripper-binary/--no-gripper-binary` | 启用 | 夹爪二值滞回控制 (解决"夹不到"/"闭合慢", 0=回退连续模式) |
| `--gripper-open-above` | 0.55 | 夹爪意图>此值→开 (二值滞回) |
| `--gripper-close-below` | 0.30 | 夹爪意图<此值→闭 (二值滞回) |
| `--gripper-min-dwell` | 40 | 夹爪状态切换最小驻留步数 (防扇动) |
| `--gripper-gain` | 0.40 | 夹爪闭合增强增益 (0 禁用, 仅二值模式关闭时生效) |
| `--gripper-gain-below` | 0.25 | 夹爪闭合增强启用阈值 |
| `--gripper-gain-above` | 0.50 | 夹爪闭合增强恢复阈值 |
| `--jump-alert-deg` | 5.0 | 关节相邻步跳变报警阈值(度) |
| `--gripper-jump-alert` | 0.15 | 夹爪相邻步跳变报警阈值 (0~1 比例, 独立于 [JUMP]) |
| `--joint-limit-rad` | 0.10 | 安全监控关节限幅 (rad/步 ≈5.7°/, 同步 flowmatch 调优) |
| `--safety-gripper-limit` | 0.35 | 安全监控夹爪限幅 (0~1) |
| `--diag/--no-diag` | 启用 | 诊断日志 (logs/pi05_diag_*.csv) |
| `--vote-window` | 5 | 投票窗口大小 |
| `--vote-threshold` | 3 | 共识阈值 |
| `--stationary-time` | 2.0 | 静止持续时间阈值 (秒), 触发 VLM 重评估 |
| `--stationary-joint` | 0.01745 | 关节变化阈值 (rad) |
| `--fast-recheck-time` | 1.0 | 快速重评估静止阈值 (秒) |
| `--burst-count` | 5 | 静止后 VLM 连拍帧数 |
| `--burst-interval` | 1.5 | VLM 连拍帧间间隔 (秒) |
| `--subtask-timeout` | 30.0 | 子任务超时 (秒), 超时强制重评估, 0=禁用 |
| `--switch-cooldown` | 3.0 | 任务切换后保护时间 (秒) |
| `--fps` | 20 | 控制频率 (Hz, 对齐 pi05 推理脚本) |
| `--device` | cuda | 推理设备 |
| `--hold-band` | 0.06 | 保持锁定判定带 (rad≈3.4°) |
| `--hold-steps` | 45 | 保持锁定连续步数阈值 |
| `--dry-run` | false | 干跑模式: 连接+读取机械臂但不发送动作 |
| `--debug` | false | 详细调试输出 |

## 与 ACT 编排器的区别

| 项目 | ACT 编排器 (act_vlm_orchestrator_v2) | PI 编排器 (本目录) |
|------|-------------------------------------|-------------------|
| 策略 | ACTPolicy (CNN 编码器) | PI05Policy (Paligemma 3B VLA) |
| 模型数量 | 多个独立模型 (Task_A/B/C) | **单个模型**, 切换 task 文本 |
| 任务切换 | 卸载/加载不同权重 | 只切换 task 文本 + 重置推理状态 (毫秒级) |
| 推理 | 单线程同步 `select_action` | 后台线程 + ActionQueue + RTC |
| 模型加载 | `ACTPolicy.from_pretrained` | config 适配 + `PI05Policy.from_pretrained` + tokenizer 覆盖 |
| RTC | 不支持 | 支持 (H=16/G=8/EXP) |
| 平滑 | 纯 EMA + 选择性限幅 | 同左 (对齐 HIL/pi05 推理) |

## 日志

运行日志和 VLM 决策记录保存在 `--log-dir` 指定的目录（默认 `./act_vlm_logs/`）。

日志文件：
- `orchestrator_YYYYMMDD_HHMMSS.log` — 完整运行日志
- `vlm_decisions_TIMESTAMP.json` — VLM 决策历史记录
