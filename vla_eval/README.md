# VLA 训练数据评估操作说明

本目录用于审计 LeRobotDataset v3.0 数据、构造对照训练、评估模型，并输出技术质量 Q、任务覆盖 C、闭环表现 P 和数据效用 U。命令从仓库根目录运行。训练和仿真使用仓库的 LeRobot 0.4.4 实现。

## 现有仓库的接口

| 现有入口 | 作用 | 评估目录的接入方式 |
|---|---|---|
| [lerobot_dataset.py](../src/lerobot/datasets/lerobot_dataset.py) 与 [dataset_tools.py](../src/lerobot/datasets/dataset_tools.py) | v3 数据读取、视频切片、按 episode 导出、汇总选定轨迹的统计 | 用 PyArrow 和 FFmpeg 审计原始交付，复用 `split_dataset` 导出训练子集 |
| [lerobot_train.py](../src/lerobot/scripts/lerobot_train.py) 与 [SmolVLA 训练脚本](../scripts/train_smolvla_quick.sh) | 加载基座模型、优化和保存处理器及检查点 | 从同一检查点启动各版本训练，固定种子、有效 batch 和最终步数 |
| [lerobot_eval.py](../src/lerobot/scripts/lerobot_eval.py) | 在已注册仿真环境中运行策略并写出 `eval_info.json` | 根据冻结的任务与种子清单接收逐案例结果 |
| [RTC 数据诊断示例](../examples/rtc/eval_dataset.py) | 使用策略时间偏移和数据集动作块评估 RTC | 复用 `resolve_delta_timestamps` 和检查点处理器，输出去除 padding 后的动作误差 |
| [数据集报告](../src/lerobot/scripts/lerobot_dataset_report.py) | 汇总 HIL 成功标注、接管比例、轨迹长度和数据集字段 | 补充时间与动作审计、来源隔离、训练对照及数据评分 |
| [SmolVLA 外部仿真脚本](../scripts/eval_smolvla_sim.sh) 与 [Piper 编排器](../src/lerobot/act_vlm_orchestrator_v2/orchestrator.py) | 外部仿真与真机运行，记录任务过程 | 按固定案例运行，通过 `import-rollouts` 导入已核验结果 |

`eval_smolvla_sim.sh` 依赖外部仿真项目与其 Python 环境。Piper 编排器的 `trial_stats.jsonl` 中，`success` 依据执行完 Task_A/B/C 且未超时判定。做数据评估时，应回看视频或物理状态，确认包裹的位置、释放与稳定等业务成功条件，再将试次绑定到案例。

## 快速验证

已有仓库环境需要能导入 `lerobot`、`torch`、`pyarrow`，系统需要 `ffmpeg` 和 `ffprobe`。静态审计只用 PyArrow 和 FFmpeg。先运行以下命令查看入口和独立统计测试。

```bash
python -m vla_eval --help
python -m unittest discover -s vla_eval/tests -v
```

在单独的 CPU 虚拟环境可复现原生集成测试。选择新建的测试目录，保留当前 GPU 训练环境的依赖。

```bash
python -m venv /tmp/vla-eval-cpu
source /tmp/vla-eval-cpu/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
CC=gcc python -m pip install -e '.[pusht]' 'scipy>=1.10,<1.15' 'transformers==4.57.1'
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 python -m vla_eval.tests.smoke --out "$PWD/vla_eval/outputs/cpu-smoke"
```

集成测试使用仓库 API 生成带视频的 v3 数据，放入一个待隔离轨迹和一个有动作跳变的轨迹。它运行三组数据、两个训练种子，随后加载检查点执行离线诊断与 PushT 闭环测试。测试使用小型 ACT 和人工构造数据，结果用于验证接口与统计流程。

## 1 准备数据与协议

以下例子假设数据放在 `/data/piper_parcel_v3`，配置与输出放在仓库之外。复制示例文件后，逐项填写实际相机名、状态和动作维度、量纲、限位、任务和场景。示例限位及时间阈值仅用于展示字段格式，评分使用本项目经验证的实际阈值。

```bash
mkdir -p /data/vla_eval_parcel
cp vla_eval/examples/protocol_piper.example.json /data/vla_eval_parcel/protocol.json
cp vla_eval/examples/recipe_pi05.example.json /data/vla_eval_parcel/recipe.json
cp vla_eval/examples/cases_real.example.json /data/vla_eval_parcel/cases.json
python -m lerobot.scripts.lerobot_dataset_report --dataset /data/piper_parcel_v3
```

- `protocol.json` 固定机器人、相机、动作和状态名称、单位、审计阈值、任务权重与覆盖配额。Piper follower 当前代码将关节角换算为弧度，并将夹爪位置除以配置开口范围。请以实际记录入口和 `meta/info.json` 为准。参考 [Piper follower](../src/lerobot/robots/piper_follower/piper_follower.py)。
- [episodes.example.jsonl](examples/episodes.example.jsonl) 每行描述一条原始 episode。完整清单覆盖全部 `episode_index`。按采集会话、原始视频、增强来源或其它依赖关系定义 `source_group`。同组分配到同一 `train`、`dev` 或 `test`。填好业务任务、覆盖格子，并在人工确认动作定义与指令匹配后填写两个核验字段。
- [timing.example.jsonl](examples/timing.example.jsonl) 每行按 `episode_index` 与 `frame_index` 提供共用时钟上的采集、接收、决策、下发与生效时间，单位为 ns。时间来自采集日志或经校准的时钟映射。
- `recipe.json` 指定本地完整模型目录、训练步数、有效 batch、种子和共用参数。基座目录包含 `config.json`、`model.safetensors` 及仓库要求的两个处理器配置。预先准备 tokenizer、视觉基座等本地缓存。Pi0.5 若启用分位数归一化，确认 episode 统计中具备所需分位数。可在已核对模型配置后为同组实验统一选择 `MEAN_STD`。
- `cases.json` 冻结一个仿真或真机案例集。为每个案例记录任务、重置场景、独立来源组、聚类 ID 和种子，并声明控制配置与成功判据。方案开发使用 `dev`，最终效果审计使用独立 `test`。当前每份计划绑定一个闭环案例集。

历史数据只有 LeRobot 的相对 `timestamp` 时，可将 `require_raw_timing` 设为 `false` 并省略 `--raw-timing`。报告保留时间同步缺项，该部分技术质量和依赖它的总分显示未评估。建议将人工核验记录与原始文件一起保留。

## 2 审计与冻结训练对照

```bash
eval_dir=/data/vla_eval_parcel
dataset=/data/piper_parcel_v3

python -m vla_eval audit \
  --dataset "$dataset" --protocol "$eval_dir/protocol.json" \
  --episodes "$eval_dir/episodes.jsonl" --raw-timing "$eval_dir/timing.jsonl" \
  --out "$eval_dir/audit.json"

python -m vla_eval plan \
  --audit "$eval_dir/audit.json" --protocol "$eval_dir/protocol.json" \
  --recipe "$eval_dir/recipe.json" --cases "$eval_dir/cases.json" \
  --out "$eval_dir/plan.json"
```

审计检查 Parquet 分片、episode 边界、帧时间网格、动作量纲与限位、抽样视频解码、黑帧、疑似冻结、原始时间关系和来源。移动轨迹的精确动作及状态重复会提示复核，跨分组重复会阻止计划继续。对近重复、语义质量和视觉遮挡仍需人工复核。可以提高 `samples_per_camera` 扩大视频检查范围。

计划生成三组训练数据。`baseline` 保留通过硬性审计的训练轨迹，`clean` 按配置的异常码筛选，`matched_random` 按清洗版的覆盖格子和轨迹数从基线抽样。当覆盖配额下存在其他组合时，随机对照会重新抽样以避免与清洗版完全相同。若无法构造不同的组合，报告标明该项比较缺少独立对照。各组总帧数单独列出。相同轨迹数仍可能对应不同的有效训练窗口数。若要消融一个来源组，在配方的 `exclude_source_groups` 中添加如 `{"without_session_8":["session-8"]}` 的定义，并用同预算结果比较。

## 3 训练并诊断动作块

```bash
python -m vla_eval train \
  --dataset "$dataset" --plan "$eval_dir/plan.json" --out "$eval_dir/experiment"

run="$eval_dir/experiment/runs/clean/11/run.json"
python -m vla_eval offline \
  --dataset "$dataset" --plan "$eval_dir/plan.json" \
  --protocol "$eval_dir/protocol.json" --run "$run" \
  --device cuda --stride 4 --draws 3 --out "${run%/*}/offline.json"
```

每个版本单独物化训练子集并根据选中 episode 汇总统计。训练时使用本地已核对的基座，保存最终步检查点、实际命令、训练日志和 `run.json`。训练前会重新验证输入数据与基座哈希。可使用 `--variant clean --seed 11` 只执行一个任务，已完成且通过完整性检查的运行可复用。中断造成的未完成运行目录可保留排障，重跑时选择新输出目录。

对各模型使用同一参考 split、stride 和 draws。离线诊断用检查点自带处理器对 ACT、SmolVLA 或 π0.5 采样动作块，按记录单位还原预测并过滤尾部 padding。结果包含每维 MAE、首个有效步误差、按固定尺度计算的误差及 episode 平均。每个动作块同时记录预测和原始标签的内部跳变及越界次数，便于排查 Pi0.5 动作块内部的突变。相邻窗口会重叠，逐块次数用于定位异常。原始标签与预测都应使用同一动作定义。使用依赖历史状态的自定义动作变换时，需要接入对应的推理处理器并先核验还原结果。

## 4 闭环测试

### 使用仓库原生仿真

在案例清单里设置 `environment=simulation`，添加 `simulation_args`，并为每条案例设置原生环境的 `suite` 与 `task_index`。每个原生任务采用相同、连续的案例种子列表。CPU 测试脚本中的 PushT 配置可作为填写示例。

```bash
python -m vla_eval sim \
  --plan "$eval_dir/plan.json" --run "$run" --out "${run%/*}/simulation"
```

仿真命令使用与模型动作空间、输入相机和状态定义匹配的环境。`sim` 记录底层命令、日志、原始 `eval_info.json` 和整理后的 `rollouts.json`。

### 使用已有真机或外部仿真

按冻结清单运行现有入口，每条记录对应一个独立的重置案例。填写 [rollouts.example.jsonl](examples/rollouts.example.jsonl) 的字段，复制 `run.json` 中的检查点哈希，附原始试次或视频的证据路径。报告可接受成功、失败、超时、接管、未启动和有原因的设施故障。设施故障后复测须保留原尝试并递增 `attempt`。

```bash
python -m vla_eval import-rollouts \
  --plan "$eval_dir/plan.json" --run "$run" \
  --records "${run%/*}/verified_trials.jsonl" --out "${run%/*}/rollouts.json"
```

## 5 汇总与解释结果

给全部计划种子运行离线和闭环命令。使用原生仿真时按下列路径汇总。导入已有真机结果时，将 rollout 文件通配符改为各次运行下的 `rollouts.json`。

```bash
python -m vla_eval report \
  --audit "$eval_dir/audit.json" --protocol "$eval_dir/protocol.json" \
  --plan "$eval_dir/plan.json" \
  --offline "$eval_dir"/experiment/runs/*/*/offline.json \
  --rollouts "$eval_dir"/experiment/runs/*/*/simulation/rollouts.json \
  --out "$eval_dir/report"
```

阶段性只生成已具备证据的分项时，可省略尚未产生的 `--offline` 或 `--rollouts`。查看 `report.md` 与 `score_report.json`。`episode_metrics.jsonl` 保存逐轨迹评分，`review_queue.jsonl` 保存异常码、计数与首个位置。

| 分项 | 计算方式 |
|---|---|
| Q | 完整性 10%、时序 25%、视觉 15%、动作 30%、语言 10%、来源 10%。严重问题隔离，缺失维度保留为 null |
| C | 按业务覆盖格子计数独立来源组，到达配额后饱和，再按固定权重求和 |
| P | 每个种子按任务权重计算闭环成功率，再跨训练种子平均。报告逐任务分母和 Wilson 区间 |
| U | 候选 P 减基线 P，以百分点报告。清洗对等量随机组的差值另列。配对区间按训练种子和任务内案例簇重采样 |
| 总分 | Q、C 和标准评测证据齐全时计算 `0.25*Q + 0.15*C + 0.60*P`。权重用于首版工程比较，需用业务任务校准 |

待验证、缺测或仍需复测的案例会留在报告中。E0 表示数据审计，E1 增加离线诊断，E2 增加完整闭环案例，E3 增加锁定测试集上的多种子配对比较。E3 描述证据来源，效果方向和稳定性依据差值及区间判断。仿真与真机的结果应在各自协议下解释。少量案例上的区间和训练排名仅用于试运行。

## 实现范围

- 原生 v3 审计、训练对照、离线动作指标、闭环导入和统计报告位于本目录。
- 当前数据效用针对数据版本或来源组。单轨迹质量和异常定位来自规则及复核记录。梯度影响函数和单轨迹重训练归因可在现有对照证据稳定后接入。
- 真实 Pi0.5 或 SmolVLA 的训练、外部仿真及 Piper 测试需使用对应的模型、数据与执行环境。当前可复现集成测试覆盖仓库的 ACT 及 PushT 接口。
