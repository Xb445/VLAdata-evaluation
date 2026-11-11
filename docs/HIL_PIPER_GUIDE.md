# Piper 双臂人类在环 (Human-in-the-Loop) 全流程说明文档

> **项目**: Evo-RL (基于 HuggingFace LeRobot)  
> **硬件**: 双 Piper 机械臂 + 双 RealSense D435 深度相机  
> **模型**: Pi0.5 (π0.5) 视觉-语言-动作策略  
> **更新日期**: 2026-07-17

---

## 一、系统架构概览

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HIL 全流程架构                                 │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ RealSense│    │  Pi0.5 模型   │    │  键盘监听器   │               │
│  │ 摄像头×2 │    │ (GPU 推理)   │    │ (pynput)     │               │
│  └────┬─────┘    └──────┬───────┘    └──────┬───────┘               │
│       │                 │                   │                        │
│       ▼                 ▼                   ▼                        │
│  ┌─────────────────────────────────────────────────┐                │
│  │              recording_loop.py                   │                │
│  │          (核心录制循环, 30/10 fps)                │                │
│  │                                                  │                │
│  │   ┌──────────────────────────────────────┐      │                │
│  │   │     Intervention State Machine        │      │                │
│  │   │                                       │      │                │
│  │   │  S0 (策略模式) ──按I──▶ S1 (示教模式) │      │                │
│  │   │       ▲                      │        │      │                │
│  │   │       └──────按I─────────────┘        │      │                │
│  │   │         (经 S2 释放态中转)             │      │                │
│  │   └──────────────────────────────────────┘      │                │
│  └────────┬───────────────────────┬────────────────┘                │
│           │                       │                                  │
│           ▼                       ▼                                  │
│  ┌────────────────┐    ┌──────────────────────┐                     │
│  │ PiperFollower  │    │   PiperLeader         │                     │
│  │ (从臂, can1)   │    │   (主臂, can0)        │                     │
│  │                 │    │                        │                     │
│  │ 位置指令模式    │    │ S0: 位置跟随模式       │                     │
│  │ 接收动作指令    │    │ S1: 示教模式(0x02)     │                     │
│  │ 执行关节运动    │    │      重力补偿+可拖动   │                     │
│  └────────────────┘    └──────────────────────┘                     │
│                                                                      │
│  数据流:                                                             │
│  S0: Camera+State → Pi0.5 → action → follower.send_action()         │
│                          └→ leader.send_feedback() (镜像)            │
│  S1: leader.get_action() → delta映射 → follower.send_action()       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 二、核心模块说明

### 2.1 脚本入口

| 文件 | 功能 |
|------|------|
| `lerobot_human_inloop_record.py` | HIL 录制入口。自动配置 `policy_sync_to_teleop=True`（S0 主臂镜像）、`intervention_state_machine_enabled=True`（干预状态机）、`enable_episode_outcome_labeling=True`（成功/失败标注）。加载失败重置位姿。 |
| `lerobot_record.py` | 录制主控制器。包含 `RecordConfig`（所有配置项），负责创建数据集、加载策略、连接双臂/摄像头、启动键盘监听、循环录制。 |

### 2.2 核心录制循环

| 文件 | 功能 |
|------|------|
| `recording_loop.py` | **核心循环**。每帧执行：观测读取 → 策略推理 → 遥操作读取 → 干预状态机决策 → 动作发送 → 数据写入。包含 S1 delta 映射和 EMA 平滑滤波。 |
| `recording_hil.py` | HIL 辅助工具。定义 `INTERVENTION_STATE_*` 常量、`ACPInferenceConfig`、`PolicySyncDualArmExecutor`（S0 双臂同步执行器）、策略运行时状态捕获/恢复。 |

### 2.3 标注与数据

| 文件 | 功能 |
|------|------|
| `recording_annotations.py` | 数据标注工具。episode 成功/失败标签标准化、`collector_policy_id` 解析（标注每帧数据来源是策略还是人类）。 |
| `lerobot_patch_hil_dataset_schema.py` | HIL 数据集 Schema 补丁，用于向后兼容。 |
| `transition.py` | RL Transition 数据结构，包含 `complementary_info` 字段用于传递 HIL 元数据。 |

### 2.4 遥操作设备

| 文件 | 功能 |
|------|------|
| `teleoperator.py` | 遥操作设备抽象基类。定义 `get_action()`（读取主臂关节）、`send_feedback()`（向主臂发送位置指令）、`set_manual_control()`（切换手动/跟随模式）接口。 |
| `piper_leader.py` | Piper 主臂实现。包含：重力补偿线程管理、夹爪被动模式、示教模式 (0x02)、位置跟随模式、关节/夹爪读数（弧度单位）。 |
| `gravity_compensation.py` | 重力补偿线程。使用 `MotionCtrl_2(0x01, 0x02, ...)` 进入 Piper 原生示教模式，固件自动处理重力补偿。 |

### 2.5 机器人

| 文件 | 功能 |
|------|------|
| `piper_follower.py` | Piper 从臂实现。接收关节位置指令并执行，读取观测（关节位置 + 摄像头图像）。 |

### 2.6 工具

| 文件 | 功能 |
|------|------|
| `control_utils.py` | 键盘监听器（`init_keyboard_listener`），支持 I/S/F/→/ESC 等快捷键，中文提示。 |
| `piper_sdk.py` | Piper SDK 工具。关节名称常量、单位转换（milli↔弧度）、CAN 模式守护（`guard_piper_ctrl_mode_on_connect`）。 |

---

## 三、工作流程详解

### 3.1 状态机

```
                          ┌─────────────────┐
                          │   S0 (策略模式)  │
                          │  state = 0.0    │
                          │                 │
                          │  Pi0.5 驱动从臂 │
                          │  主臂位置镜像   │
                          └────────┬────────┘
                                   │ 按下 I 键
                                   ▼
                          ┌─────────────────┐
                          │   S1 (示教模式)  │
                          │  state = 1.0    │
                          │                 │
                          │  主臂拖动遥控   │
                          │  delta 映射从臂 │
                          │  EMA 平滑滤波   │
                          └────────┬────────┘
                                   │ 按下 I 键
                                   ▼
                          ┌─────────────────┐
                          │   S2 (释放态)    │
                          │  state = 2.0    │
                          │                 │
                          │  策略 reset     │
                          │  清空缓存       │
                          └────────┬────────┘
                                   │ 下一帧自动
                                   ▼
                          ┌─────────────────┐
                          │   S0 (策略模式)  │
                          └─────────────────┘
```

### 3.2 数据流

```
S0 (策略模式):
  摄像头 → 图像观测 ─┐
  从臂关节 → 状态观测 ┤→ Pi0.5 模型推理 → 动作 → 从臂执行
                      │                         └→ send_feedback → 主臂镜像
                      └→ 写入 Dataset (is_intervention=0, collector=policy)

S1 (示教模式):
  主臂关节 → get_action() → 读数(弧度) → EMA平滑 → delta映射 → 从臂执行
                                                     └→ 写入 Dataset (is_intervention=1, collector=human)
```

### 3.3 Delta 映射机制

进入 S1 时捕获参考位姿：
- `teleop_ref` = 主臂当前关节位置（弧度）
- `robot_ref` = 从臂当前关节位置（来自 `obs` 观测）

S1 每帧：
- `delta = teleop_current - teleop_ref`（主臂的相对运动）
- `robot_target = robot_ref + delta`（从臂跟随相同相对运动）

EMA 平滑滤波（关节 α=0.5, 夹爪 α=0.9）抑制主臂读数噪声。

> **夹爪特殊处理**：夹爪不使用 Delta 映射，而是采用**绝对位置映射**
> (`follower_gripper = leader_gripper`)。原因：
> 1. 夹爪运动是离散的（捏合/松开），Delta 映射在参考位姿不一致时会产生偏移
> 2. 夹爪行程有界 [0, 70mm]，Delta 映射在边界处容易饱和
> 3. 夹爪使用独立 EMA (α=0.9) 而非关节的 α=0.5，减少响应延迟

### 3.4 Episode 生命周期

```
开始录制
  ├─ 加载 Pi0.5 模型 (GPU)
  ├─ 连接双臂 (CAN) + 摄像头 (OpenCV)
  ├─ 加载/捕获 失败重置位姿
  │
  ├─ Episode N 录制 (60s / 10fps)
  │   ├─ S0: 策略推理驱动从臂
  │   ├─ [按 I] S1: 人类介入示教
  │   ├─ [按 I] 返回 S0
  │   ├─ [按 S/F] 标记成功/失败 → 结束
  │   └─ 自动结束 (episode_time_s 超时, 默认失败)
  │
  ├─ 环境重置 (reset_time_s)
  │   └─ 纯遥操作，可拖动主臂调整从臂位置
  │
  ├─ Episode N+1 ...
  │
  └─ 停止录制 (ESC / num_episodes 完成)
      ├─ 视频编码 (SVT-AV1)
      ├─ 断开双臂/摄像头
      └─ 保存完整 Dataset
```

---

## 四、使用方法

### 4.1 前置条件

```bash
# 激活环境
conda activate evo-rl

# 确认 CAN 接口状态
lerobot-setup-can --mode=test --interfaces=can0,can1

# 确认摄像头可用
python3 -c "
import cv2
for idx, name in [(4, 'wrist'), (10, 'top')]:
    cap = cv2.VideoCapture(idx)
    ret, frame = cap.read()
    print(f'{name} (idx {idx}): {\"OK\" if ret else \"FAIL\"}')
    cap.release()
"
```

### 4.2 基础命令

```bash
lerobot-human-inloop-record \
    --robot.type=piper_follower \
    --robot.port=can1 \
    --robot.id=slave_arm \
    --robot.require_calibration=False \
    --robot.cameras='{wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}' \
    --teleop.type=piper_leader \
    --teleop.port=can0 \
    --teleop.id=master_arm \
    --teleop.require_calibration=False \
    --policy.path=/path/to/pretrained_model \
    --dataset.repo_id=<user>/<dataset_name> \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=60 \
    --dataset.fps=30 \
    --dataset.single_task="Task description" \
    --dataset.push_to_hub=False \
    --display_data=true
```

### 4.3 参数详解

#### 机器人 (--robot.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | str | — | 从臂类型：`piper_follower` / `piperx_follower` |
| `port` | str | — | CAN 接口名称，如 `can1` |
| `id` | str | — | 臂标识，用于标定文件命名 |
| `require_calibration` | bool | `True` | 是否需要标定。无标定文件时设为 `False` 使用直通模式 |
| `cameras` | dict | `{}` | 摄像头配置，YAML 格式。支持 `opencv` 和 `realsense` 类型 |
| `enable_on_connect` | bool | `True` | 连接后自动使能臂 |
| `sync_gripper` | bool | `True` | 是否同步夹爪 |

#### 遥操作 (--teleop.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | str | — | 主臂类型：`piper_leader` / `piperx_leader` |
| `port` | str | — | CAN 接口名称，如 `can0` |
| `id` | str | — | 臂标识 |
| `require_calibration` | bool | `True` | 是否需要标定 |
| `manual_control` | bool | `True` | 启动时是否进入示教模式 |
| `gravity_comp_mit_kp` | float | `0.0` | 重力补偿位置反馈增益（建议保持 0） |
| `gravity_comp_mit_kd` | float | `0.0` | 重力补偿速度阻尼（非零会增加拖动阻力） |
| `gravity_comp_torque_limit` | float | `8.0` | 重力补偿力矩上限 |
| `gravity_comp_tx_ratio` | tuple | `(0.2×6)` | 各关节重力补偿比例（J1-J6），可针对下垂关节加大 |
| `sync_gripper` | bool | `True` | 是否同步夹爪 |

#### 策略 (--policy.*)

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | str | 预训练模型路径（本地目录或 HF repo） |
| `device` | str | 推理设备，默认 `cuda` |

#### 数据集 (--dataset.*)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `repo_id` | str | — | 数据集标识，格式 `<user>/<name>` |
| `num_episodes` | int | `50` | 录制 episode 数量 |
| `episode_time_s` | int/float | `60` | 每个 episode 录制时长（秒） |
| `reset_time_s` | int/float | `60` | episode 间环境重置时长（秒） |
| `fps` | int | `30` | 录制帧率。降低可减缓推理速度 |
| `single_task` | str | — | 任务描述（必填） |
| `video` | bool | `True` | 是否编码视频 |
| `push_to_hub` | bool | `True` | 是否推送到 HuggingFace Hub |
| `vcodec` | str | `libsvtav1` | 视频编码器 |

#### 其他

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `display_data` | bool | `False` | 是否启动 Rerun 可视化窗口 |
| `intervention_toggle_key` | str | `i` | 干预切换键 |
| `episode_success_key` | str | `s` | 标记成功键 |
| `episode_failure_key` | str | `f` | 标记失败键 |
| `play_sounds` | bool | `True` | 是否播放语音提示 |

### 4.4 键盘快捷键

| 按键 | 功能 |
|------|------|
| `I` | 切换 S0(策略模式) ↔ S1(示教模式) |
| `S` | 标记当前 episode 成功并结束 |
| `F` | 标记当前 episode 失败并结束 |
| `→` (右箭头) | 跳过当前 episode |
| `←` (左箭头) | 重录上一个 episode |
| `ESC` | 停止录制并退出 |

### 4.5 无策略纯遥操作测试

```bash
lerobot-human-inloop-record \
    --robot.type=piper_follower \
    --robot.port=can1 \
    --robot.id=slave_arm \
    --robot.require_calibration=False \
    --teleop.type=piper_leader \
    --teleop.port=can0 \
    --teleop.id=master_arm \
    --teleop.require_calibration=False \
    --dataset.repo_id=test/piper_teleop_test \
    --dataset.num_episodes=2 \
    --dataset.episode_time_s=30 \
    --dataset.single_task="Teleop test" \
    --dataset.push_to_hub=False
```

---

## 五、数据集格式

录制完成的数据集为标准 LeRobot v3 格式，可直接用于训练：

```
~/.cache/huggingface/lerobot/<user>/<dataset_name>/
├── meta/
│   ├── info.json       # 数据集元信息
│   ├── stats.json      # 归一化统计
│   ├── episodes.json   # episode 列表
│   └── tasks.json      # 任务描述
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # 观测/动作/标注数据
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    ├── observation.images.wrist/
    │   ├── episode_000000.mp4
    │   └── ...
    └── observation.images.top/
        ├── episode_000000.mp4
        └── ...
```

### 关键数据字段

| 字段 | 形状 | 说明 |
|------|------|------|
| `observation.state` | (7,) | 从臂 7 维关节状态（6 关节 + 1 夹爪） |
| `observation.images.wrist` | (480, 640, 3) | 腕部摄像头 RGB 图像 |
| `observation.images.top` | (480, 640, 3) | 顶部摄像头 RGB 图像 |
| `action` | (7,) | 实际执行的 7 维动作 |
| `complementary_info.policy_action` | (7,) | 策略原始输出（即使被干预也保存） |
| `complementary_info.is_intervention` | (1,) | 是否为干预帧（0=策略, 1=人类） |
| `complementary_info.state` | (1,) | 状态机状态（0=S0, 1=S1, 2=S2） |
| `complementary_info.collector_policy_id` | (1,) | 数据来源标识（策略路径或 "human"） |
| `task` | str | 任务描述 |
| `episode_success` | str | episode 级别标签：success / failure |

### 查看数据集

```bash
# 数据集报告
lerobot-dataset-report --dataset test/eval_piper_hil_pi05

# 可视化
lerobot-dataset-viz --dataset test/eval_piper_hil_pi05 --episode 0
```

---

## 六、调试与故障排除

### 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| 双臂连接后不使能 | CAN 模式残留 | `guard_piper_ctrl_mode_on_connect` 自动切换，或断电重启 |
| 主臂 S0 不跟随从臂 | 位置指令模式未激活 | 确认 `policy_sync_to_teleop=True`，S0 初始化已自动处理 |
| S1 从臂抖动 | 主臂读数噪声 | EMA 平滑已内置（α=0.5），调整 `_ema_alpha` 变量 |
| 主臂拖动不灵活 | kd 值过大 | 保持 `gravity_comp_mit_kd=0.0` |
| 远端关节下垂 | tx_ratio 偏低 | 增加 J4/J5 的 tx_ratio：`--teleop.gravity_comp_tx_ratio='[0.2,0.2,0.2,0.5,0.5,0.3]'` |
| 摄像头连接失败 | 索引或分辨率错误 | 用 `v4l2-ctl --list-devices` 确认设备，宽高为 640×480 |
| 数据集名称报错 | 策略模式下需 `eval_` 前缀 | 改为 `test/eval_<name>` |
| **S1 夹爪不同步/延迟大** | EMA 平滑 + Delta 映射不适合夹爪；夹爪 CAN 反馈等待阻塞 | 已修复：夹爪独立 EMA(α=0.9)+绝对映射+非阻塞读取+力矩提升至 4N/m |
| **示教器夹爪无反馈** | 未调用 `GripperTeachingPendantParamConfig` 初始化示教器参数 | 已修复：连接时自动配置示教器 (range_per=100, max_range=70mm, friction=1)。SDK 要求首次使用前必须配置 |
| **纯遥操夹爪完全不跟随** | `_set_gripper_passive(effort=0)` 发送一次后固件超时停止编码器 CAN 报告 | 已修复：① 被动模式 effort=50(0.05N·m) ② 每帧 GripperCtrl keep-alive |
| **夹爪闭合力不足/响应慢** | `gripper_effort_default` 默认值偏低（1 N/m） | 已提升默认值为 4000 (4 N/m)，或手动指定 `--robot.gripper_effort_default=5000` |

### 日志关键字

| 日志 | 含义 |
|------|------|
| `🎮 S0 模式` | 进入策略驱动模式 |
| `🟢 S1 示教模式已激活` | 进入人类示教模式 |
| `📐 S1 delta reference captured` | Delta 参考位姿已捕获 |
| `🔵 控制权已释放` | 从 S1 返回 S0 |
| `🔄 环境重置中` | Episode 间重置阶段 |
| `Map: 100% xxx/xxx` | 视频编码完成 |

---

## 七、已实施的优化总结

| 优化项 | 说明 |
|--------|------|
| 主臂读数弧度单位统一 | `get_action()` 返回弧度（÷57.3），与从臂 `send_action()` 期望一致 |
| 夹爪被动模式 | `_set_gripper_passive()`：编码器保持活跃 + 零力矩，可自由捏合且读数实时 |
| Piper 原生示教模式 | `MotionCtrl_2(0x01, 0x02)` 替代 Pinocchio MIT，固件自动重力补偿 |
| S1 Delta 映射 | 主臂相对运动 → 从臂跟随，消除绝对位置跳变 |
| EMA 平滑滤波 | 关节 α=0.5 抑制振动；**夹爪 α=0.9（近直通）** |
| **夹爪绝对位置映射** | S1 中夹爪使用绝对映射（从臂=主臂），替代 Delta 映射，消除偏移累积 |
| **夹爪非阻塞读取 + Keep-Alive + 示教器配置** | `_read_raw_action()` 先读夹爪再读关节；不等待夹爪 CAN 帧；每帧 GripperCtrl keep-alive；连接时自动 `GripperTeachingPendantParamConfig` 初始化示教器 |
| **夹爪力矩提升** | 默认 `gripper_effort_default` 从 1000(1N/m) → 4000(4N/m)；被动模式使用 effort=50(0.05N·m) 保持编码器活跃 |
| 自动模式切换 | `guard_piper_ctrl_mode_on_connect` 检测并切换示教→位置模式 |
| 中文操作提示 | 启动面板 + 按键提示 + 状态切换提示 |
| 调试输出清理 | 移除 `srj:` 调试打印 |
