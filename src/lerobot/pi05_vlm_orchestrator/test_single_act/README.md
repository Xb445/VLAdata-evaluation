# 单个 ACT 模型推理效果测试

对 `act_vlm_orchestrator_v2` 大模型调度项目中的单个 ACT 模型做独立推理测试。
基于 `src/lerobot/inference_new_joint_act.py` 简化而来，仅对齐编排器的
**平滑系数** 和 **机械臂 MIT 驱动模式**，不含状态锁等复杂逻辑。

## 文件

```
test_single_act/
├── test_single_act_inference.py   # 单 ACT 模型推理测试脚本 (唯一核心文件)
├── run_test.sh                    # 一键启动脚本
├── README.md
└── logs/                          # 运行日志 (自动生成)
```

## 与编排器对齐的内容

### 1. 平滑系数 (复用编排器 TrajectorySmoother)

| 参数 | 编排器值 |
|------|---------|
| EMA alpha (逐关节) | `[0.30, 0.12, 0.18, 0.30, 0.30, 0.30, 0.50]` |
| 速率限制 (rad/帧, 仅 J2/J3) | `[None, 0.015, 0.015, None, None, None, None]` |
| 夹爪死区 | `0.15` |

直接复用编排器 `orchestrator.py` 中的 `TrajectorySmoother` 类，保证行为完全一致。

### 2. 机械臂 MIT 高跟随模式

```python
# 使能后 + 每步发送前 (与编排器 _enable_robot / _send_action 一致)
robot.piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)   # 100% 速度 + MIT 高跟随
```

> 旧版 `inference_new_joint_act.py` 用 `MotionCtrl_2(0x01, 0x01, 30, 0x00)`，
> 机械臂跟不上 ACT chunk 命令会造成观测-动作失同步与抖动，现已对齐。

## 平滑开关 (后处理优化开关)

| 参数 | 行为 |
|------|------|
| `--smooth.mode=smooth` (默认) | 开启编排器风格平滑 (EMA + 限速 + 夹爪死区) |
| `--smooth.mode=none` | 关闭平滑，直接输出模型原始动作 (对比用) |

## 预移动到训练初始位姿 (v4 新增, 避免推理开始时机身跳变)

**背景**: 训练时机械臂从训练初始位姿开始执行任务。若推理时当前位姿与训练
初始位姿差异大, 模型首个 chunk 会命令机械臂**迅速跳到**训练初始位姿
(大关节跳变)。

**方案**: 推理循环开始前, 从当前位姿在关节空间线性插值平滑移动到训练初始位姿
(默认 300 步 × 20ms ≈ 6 秒), 消除跳变。

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--preset_pose` | `True` | 是否预移动 (干跑模式自动关闭) |
| `--preset_target` | 自动 | 目标位姿 (7 值: j1..j6 弧度, gripper 0-1); 为空时从模型 normalizer 自动读取训练初始位姿 (`observation.state.mean`) |
| `--preset_steps` | `300` | 插值步数 (300 步 × 0.02s ≈ 6s) |
| `--preset_delay` | `0.02` | 每步间隔 (秒) |
| `--preset_min_diff_deg` | `3.0` | 与目标最大关节差异小于该值(度)时自动跳过 |

自动跳过规则: 若当前位姿与训练初始位姿最大关节差异 < `--preset_min_diff_deg`
(默认 3°) 则跳过预移动, 直接开始推理。关闭方式: `--preset_pose False`。

## 退出行为 (v3 修复)

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--exit_mode` | `hold` | 退出时机械臂处理: `hold`=保持使能并固定在原位 / `disable`=失能(抱闸固定) |

退出时自动执行的清理 (无论 Ctrl+C、正常结束还是异常):
1. **恢复 Ctrl+C**: 覆盖编排器模块导入时注册的 SIGINT/SIGTERM 处理器 (只置标志不退出),
   恢复 KeyboardInterrupt 中断行为, Ctrl+C 可正常退出;
2. **机械臂处理**: 默认读取当前关节并下发保持, 使机械臂保持使能固定在原位
   (或 `--exit_mode=disable` 失能, 依靠伺服抱闸固定);
3. **释放相机**: 释放 OpenCV VideoCapture, 避免 `/dev/video*` 被占用导致下次启动报
   `Failed to open OpenCVCamera`;
4. **断开 CAN**: `DisconnectPort()` 关闭 CAN 端口与接收线程。

> ⚠️ 若之前运行异常残留了进程占用相机, 可用 `ps aux | grep test_single_act`
> 找到残留 PID 后 `kill -9 <PID>` 释放相机。

## 使用方法

### 一键启动

```bash
cd /home/ub/Videos/Evo-RL/src/lerobot/act_vlm_orchestrator_v2/test_single_act
chmod +x run_test.sh

./run_test.sh                  # 默认 Task_A, 平滑开启, 真机控制
./run_test.sh Task_B           # 测试 Task_B
./run_test.sh Task_A --dry_run True # 观察模式 (只推理不动作)
./run_test.sh Task_A --smooth.mode=none   # 关闭平滑
```

### 手动启动 (参数更灵活)

```bash
cd /home/ub/Videos/Evo-RL

# 观察模式 (安全验证模型输出)
python -m lerobot.act_vlm_orchestrator_v2.test_single_act.test_single_act_inference \
  --robot.type=single_piper --robot.id=single_piper_robot --robot.ctrl_mode="joint" \
  --robot.port=can1 \
  --robot.cameras="{wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
  --ckpt_path=/home/ub/Videos/0407/model/ACT_model/model/Task_A/pretrained_model \
  --task="Grab the parcel from the table and place it on the tray" \
  --dry_run True

# 真机控制 (对齐编排器平滑 + MIT 模式 + 预移动)
#  (同上, 去掉 --dry_run True; 默认会自动平滑移动到训练初始位姿)

# 关闭平滑对比原始输出
#  (同上, 加 --smooth.mode=none)

# 关闭预移动 (不移动机械臂, 直接推理)
#  (同上, 加 --preset_pose False)
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--robot.*` | - | 机器人配置 (同 inference_new_joint_act.py) |
| `--ckpt_path` | - | ACT 模型 checkpoint 目录 |
| `--task` | `Perform the manipulation task` | 任务描述 |
| `--smooth.mode` | `smooth` | 平滑开关: smooth / none |
| `--fps` | `25.0` | 控制频率 Hz |
| `--device` | `cuda` | 推理设备 |
| `--inference_time_s` | `10000` | 最大运行秒数 |
| `--dry_run` | `False` | 观察模式: 只推理不发送动作 |
| `--log_dir` | `./act_vlm_logs` | 日志目录 |
| `--save_frame` | `0` | 每 N 步保存相机帧 (0=不保存) |
| `--exit_mode` | `hold` | 退出时机械臂处理: hold / disable |
| `--preset_pose` | `True` | 推理前平滑移动到训练初始位姿 |
| `--preset_target` | 自动 | 预移动目标位姿 (7 值) |
| `--preset_steps` | `300` | 预移动插值步数 |
| `--preset_delay` | `0.02` | 预移动每步间隔 (秒) |
| `--preset_min_diff_deg` | `3.0` | 预移动跳过阈值 (度) |

## 日志

每次运行自动保存到 `--log_dir`：
- `single_act_YYYYmmdd_HHMMSS.log` — 终端双写完整日志
- `actions_YYYYmmdd_HHMMSS.jsonl` — 逐帧动作日志 (原始/最终动作、下发命令、推理耗时)
- `frames_*/` — 相机帧快照 (`--save_frame N` 时)

## 与旧版 (inference_new_joint_act.py) 的差异

- 平滑器: 旧版为 窗口平均+统一EMA+全关节限幅+死区；新版复用编排器 逐关节EMA+仅J2/J3限速+夹爪死区
- MIT 模式: `MotionCtrl_2(0x01,0x01,30,0x00)` → `MotionCtrl_2(0x01,0x01,100,0xAD)`
- 新增: 终端+文件双写日志、逐帧 JSONL 动作日志、`--dry_run`、`--save_frame`、`--exit_mode`
- 新增 (v3): 恢复 Ctrl+C 中断、退出时释放相机/断开 CAN、机械臂保持原位或失能
- 移除: 编排器的保持锁定/状态锁/VLM 等复杂逻辑
