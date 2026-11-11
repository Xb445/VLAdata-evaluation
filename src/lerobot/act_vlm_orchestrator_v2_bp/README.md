# ACT + VLM 多任务联调编排器

## 概述

基于 VLM (Qwen3-VL-2B) 大语言模型实现多个 ACT 子任务模型 (Task_A/B/C) 之间的智能联调。

### 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ACT-VLM Orchestrator                      │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Camera   │   │   VLM    │   │  Window  │   │   ACT    │ │
│  │  Manager  │──▶│  Client  │──▶│  Voter   │──▶│  Model   │ │
│  │           │   │ (vLLM)   │   │ (5帧≥3)  │   │  Manager │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                      │       │
│  ┌──────────┐   ┌──────────┐                         │       │
│  │Stationary│   │  Safety  │                         ▼       │
│  │ Detector │──▶│  Monitor │◀───────────  Piper 机械臂      │
│  │ (3s静止) │   │          │                                │
│  └──────────┘   └──────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

### 场景分类映射

| VLM 分类 | 场景描述 | ACT 模型 | 动作 |
|---------|---------|---------|------|
| subtask1 | 包裹在桌面 | Task_A | 抓取包裹→放入托盘 |
| subtask2 | 包裹在托盘，标签不可见 | Task_B | 翻转包裹使标签可见 |
| subtask3 | 包裹在托盘，标签可见 | Task_C | 抓取包裹→放入盒子 |
| subtask4 | 包裹在盒子 | standby | 待机等待场景复原 |

### 核心机制

1. **窗口投票**：5 次 VLM 判断中 ≥3 次一致才采纳结果，防止模型误判
2. **静止检测（保险机制）**：机械臂接近静止 3s 时自动触发 VLM 重评估
3. **模型切换清缓存**：切换 ACT 模型时清除 GPU 缓存 + 重置模型状态，消除累积误差

## 目录结构

```
act_vlm_orchestrator/
├── __init__.py              # 包初始化
├── config.py                # 配置数据类
├── vlm_client.py            # VLM API 客户端 (vLLM)
├── window_voter.py          # 窗口投票机制
├── act_model_manager.py     # ACT 模型管理与切换
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
  --model /home/ub/Videos/0407/model/qwen3_vl_2b_parcel_subtask_merged \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 8192 --max-num-seqs 4 \
  --dtype auto --trust-remote-code \
  --served-model-name qwen3-vl-2b
```

### 2. 确保 ACT 模型就绪

```
/home/ub/Videos/0407/model/ACT_model/model/
├── Task_A/pretrained_model/   # 桌面→托盘
├── Task_B/pretrained_model/   # 托盘翻转
└── Task_C/pretrained_model/   # 托盘→盒子
```

### 3. 确保硬件连接

- Piper 机械臂（CAN 总线）
- 顶部相机（ID=10）
- 腕部相机（ID=4）

## 使用方法

### 一键启动

```bash
cd /home/ub/Videos/Evo-RL/src
./lerobot/act_vlm_orchestrator_v2/start_orchestrator.sh
```

### 手动启动

```bash
cd /home/ub/Videos/Evo-RL

# 默认配置 (模型路径已指向本机 /home/ub/Videos/0407/model/ACT_model/model)
python -m lerobot.act_vlm_orchestrator_v2.run_orchestrator

# 调试模式
python -m lerobot.act_vlm_orchestrator_v2.run_orchestrator --debug

# CPU 推理 (调试用)
python -m lerobot.act_vlm_orchestrator_v2.run_orchestrator --device cpu

# 节省 GPU 显存 (按需加载模型)
python -m lerobot.act_vlm_orchestrator_v2.run_orchestrator --no-preload-all

# 干跑模式 (安全验证状态机: 连接+读取机械臂, 但不发送动作指令)
python -m lerobot.act_vlm_orchestrator_v2.run_orchestrator --dry-run

# 自定义投票参数
python -m lerobot.act_vlm_orchestrator_v2.run_orchestrator --vote-window 7 --vote-threshold 5

# 禁用轨迹平滑
python -m lerobot.act_vlm_orchestrator_v2.run_orchestrator --no-smooth
```

### VLM 独立测试（不控制机械臂）

```bash
# 单次实时相机测试
python -m lerobot.act_vlm_orchestrator.test_vlm_only --use-cameras

# 持续监控模式
python -m lerobot.act_vlm_orchestrator.test_vlm_only --use-cameras --continuous

# 使用静态图片测试
python -m lerobot.act_vlm_orchestrator.test_vlm_only \
  --image-top /path/to/top.jpg \
  --image-wrist /path/to/wrist.jpg
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--vlm-url` | `http://localhost:8000/v1` | vLLM API 地址 |
| `--vlm-model` | `qwen3-vl-2b` | vLLM 模型名 |
| `--vote-window` | 5 | 投票窗口大小 |
| `--vote-threshold` | 3 | 共识阈值 |
| `--stationary-time` | 2.0 | 静止持续时间阈值 (秒), 触发 VLM 重评估 |
| `--stationary-joint` | 0.01745 | 关节变化阈值 (rad) |
| `--fast-recheck-time` | 1.0 | 快速重评估静止阈值 (秒): 连拍后未切换时加速重评估 |
| `--no-fast-recheck` | false | 禁用快速重评估 |
| `--burst-count` | 5 | 静止后 VLM 连拍帧数 |
| `--burst-interval` | 1.5 | VLM 连拍帧间间隔 (秒) |
| `--subtask-timeout` | 30.0 | 子任务超时 (秒), 超时强制重评估, 0=禁用 |
| `--switch-cooldown` | 3.0 | 模型切换后保护时间 (秒) |
| `--vlm-interval` | 50 | VLM 主动检查间隔 (步) |
| `--fps` | 20 | ACT 推理频率 |
| `--device` | cuda | 推理设备 |
| `--temporal-ensemble-coeff` | 0 | ACT 时序集成系数 (默认 0=禁用=队列模式; 0.01 会缩小动作幅度, 建议保持 0) |
| `--chunk-transition-duration` | 0.25 | chunk 边界过渡平滑时长 (秒), 抑制新 chunk 首动作跳变, 设 0 禁用 (默认 0.25s=8 步@30Hz, 短停顿+无抖动) |
| `--hold-band` | 0.03 | 保持锁定判定带 (rad≈1.7°): 模型输出连续多步在当前关节带内视为"保持", 冻结关节抑制微振荡, 设 0 禁用 |
| `--hold-steps` | 60 | 保持锁定连续步数阈值 (≈2s @30Hz), 达到后激活锁定 |
| `--dry-run` | false | 干跑模式: 连接+读取机械臂但不发送动作 |
| `--debug` | false | 详细调试输出 |

## 日志

运行日志和 VLM 决策记录保存在 `--log-dir` 指定的目录（默认 `./act_vlm_logs/`）。

日志文件：
- `orchestrator_YYYYMMDD_HHMMSS.log` — 完整运行日志
- `vlm_decisions_TIMESTAMP.json` — VLM 决策历史记录
