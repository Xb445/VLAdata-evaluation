#!/usr/bin/env bash
# ==============================================================================
# Pi0.5 模型快速启动训练脚本
# ==============================================================================
# 使用方法:
#   方式1（推荐）: 直接修改下方参数后运行
#       bash scripts/train_pi05.sh
#
#   方式2: 通过环境变量覆盖特定参数
#       BATCH_SIZE=1 STEPS=5000 bash scripts/train_pi05.sh
#
#   方式3: 快速测试（3步验证脚本可用性）
#       bash scripts/train_pi05.sh --test
#
#   方式4: 全参数微调（非 LoRA，对比实验用）
#       bash scripts/train_pi05.sh --full-ft
#
# ★ 2026-08-14 调参说明（基于 0407 pi05_adarms_fix 全量 checkpoint 分析）：
#   问题：60000 步训练中，loss 在 ~20k 平台(0.041)，20k 后预测质量(ADE)反而退化
#        （gripper 双峰退化、ADE 漂移），后段在地板 lr 下空转破坏生成质量
#   措施：STEPS 60000→25000、退火 30000→22000、预热 1000→500，
#         25k 步内完成退火，避开过训练区（历史最优在 15k~20k）
#   注意：训练无内置验证集/早停，25k 会产出 5k/10k/15k/20k/25k 五个 checkpoint，
#         训练结束后请逐个评估 ADE 并挑选最优者部署
#
# 环境要求:
#   - conda 环境: lerobot (Evo-RL 项目默认环境)
#   - GPU: 推荐 80GB 显存（如 H100/A100）
#   - 数据集: /data/zhanghang/Videos/datasets/
#   - 预训练模型: /data/zhanghang/Videos/0407/model/pi05_base
# ==============================================================================

# 快速测试模式
if [ "${1:-}" = "--test" ]; then
    echo ">>> 快速测试模式：运行 3 步训练验证脚本可用性 <<<"
    export BATCH_SIZE=1
    export STEPS=3
    export LOG_FREQ=1
    export SAVE_FREQ=100000
    export NUM_WORKERS=0
    export NUM_GPUS=1
    export USE_PEFT=false
    export OUTPUT_DIR="outputs/train/test_pi05_quick"
    export JOB_NAME="test_pi05_quick"
    # 清理上次测试的残留输出目录，避免 FileExistsError
    rm -rf "outputs/train/test_pi05_quick" 2>/dev/null || true
fi

# 全参数微调模式（非 LoRA，用于与 LoRA 对比实验）
if [ "${1:-}" = "--full-ft" ]; then
    echo ">>> 全参数微调模式（非 LoRA）：2 卡训练，对比 LoRA 效果 <<<"
    export NUM_GPUS=2
    export USE_PEFT=false
    export BATCH_SIZE=64                  # 全参数训练显存需求大，单卡 batch 不宜过高
    export JOB_NAME="pi05_fullft_bs64_lr5e-5_25K"
    export OUTPUT_DIR="outputs/train/pi05_bs64_lr5e-5_25K"
    export OPTIMIZER_LR=5e-5           # 全参数训练用较低学习率，防止梯度爆炸
    export SCHEDULER_DECAY_LR=5e-6
    export GRADIENT_CHECKPOINTING=true   # 必须开启，否则 80GB 显存也不够
    export FREEZE_VISION_ENCODER=true
    export TRAIN_EXPERT_ONLY=true
    export LOG_FREQ=100
    export SAVE_FREQ=5000
    export STEPS=25000   # 0407 分析：loss 20k 平台、15-20k 最优、20k 后退化 → 25k
    export RESUME=false
    # 清理上次残留输出目录
    rm -rf "outputs/train/pi05_bs64_lr5e-5_25K" 2>/dev/null || true
fi

# 全参数微调续训模式（从 checkpoint 恢复被中断的训练）
if [ "${1:-}" = "--full-ft-resume" ]; then
    echo ">>> 全参数微调续训模式：从 pi05_fullft checkpoint 恢复 <<<"
    export NUM_GPUS=2
    export USE_PEFT=false
    export BATCH_SIZE=64
    export JOB_NAME="pi05_fullft_bs64_lr5e-5_25K"
    export OUTPUT_DIR="outputs/train/pi05_bs64_lr5e-5_25K"
    export OPTIMIZER_LR=5e-5
    export SCHEDULER_DECAY_LR=5e-6
    export GRADIENT_CHECKPOINTING=true
    export FREEZE_VISION_ENCODER=true
    export TRAIN_EXPERT_ONLY=true
    export LOG_FREQ=50
    export SAVE_FREQ=5000
    export STEPS=25000
    export RESUME=true                  # ← 关键：从 checkpoint 续训
fi

# ACP 模式（使用价值模型标记后的数据，注入 Advantage: positive/negative 标签）
if [ "${1:-}" = "--acp" ]; then
    echo ">>> ACP 模式：使用价值模型标记后的 ACP 数据 <<<"
    export ACP_ENABLE=true
    export ACP_INDICATOR_FIELD="complementary_info.acp_indicator"
    export ACP_INDICATOR_DROPOUT_PROB=0.1
    export JOB_NAME="pi05_acp_bs96_lr5e-5"
    export OUTPUT_DIR="outputs/train/pi05_acp"
    rm -rf "outputs/train/pi05_acp" 2>/dev/null || true
fi

# ==============================================================================
# 用户配置区 - 所有参数支持环境变量覆盖（${ENV_VAR:-默认值}）
# ==============================================================================

# ============================================================
# 5.0 GPU 与分布式训练配置
# ============================================================

# 使用的 GPU 数量。1 为单卡训练，>1 则通过 accelerate 启动多卡分布式训练
# 原理：多卡时脚本自动切换为 accelerate launch --multi_gpu --num_processes=N
# 调优：设为 0 或不设置则自动检测（使用所有可用 GPU）
#       设为 1 使用单卡（lerobot-train 直接启动）
#       设为 2/3/4... 使用对应数量的 GPU 进行数据并行训练
# 注意：多卡训练时 batch_size 为每张卡的 batch_size，有效 batch = batch_size × GPU 数
# 警告：多卡训练需确保 accelerate 已配置（运行 accelerate config 进行初始化）
NUM_GPUS="${NUM_GPUS:-2}"

# 策略运行的设备（cuda / cpu）
# 影响：通常设为 cuda；调试或在无 GPU 环境时设为 cpu（极慢）
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

# HuggingFace Hub 上的模型仓库 ID（用于推送训练好的模型）
# 影响：push_to_hub=true 时必须设置；离线/本地训练时设为任意占位值即可
POLICY_REPO_ID="${POLICY_REPO_ID:-local/do_not_upload}"

# ============================================================
# 5.0-A ACP（自动课程规划）配置
# ============================================================
# 说明：ACP 需要先用价值模型对数据集进行推理标记（lerobot_value_infer）
#       标记后数据集中会多出 complementary_info.acp_indicator 列（0/1）
#       ACP 训练时，会根据 indicator 自动修改 task prompt 文本：
#         indicator=1 → "task\nAdvantage: positive"（高质量帧，模仿学习）
#         indicator=0 → "task\nAdvantage: negative"（低质量帧，修正学习）

# 是否启用 ACP 标签注入（需要数据集已被价值模型标记）
ACP_ENABLE="${ACP_ENABLE:-false}"

# ACP 指示器字段名（数据集中标记 0/1 的列名）
ACP_INDICATOR_FIELD="${ACP_INDICATOR_FIELD:-complementary_info.acp_indicator}"

# ACP 标签 dropout 概率（随机丢弃 ACP 标签，增加鲁棒性）
ACP_INDICATOR_DROPOUT_PROB="${ACP_INDICATOR_DROPOUT_PROB:-0.1}"

# ============================================================
# 5.1 路径配置
# ============================================================
# 说明：以下路径需要根据实际环境修改。使用绝对路径避免相对路径歧义。

# 项目根目录（Evo-RL 项目源码所在位置）
# 影响：脚本将 cd 到此目录执行训练，所有相对路径以此为基准
PROJECT_DIR="${PROJECT_DIR:-/data/zhanghang/Videos/Evo-RL}"

# 数据集在本地磁盘的实际存储路径（必须包含 meta/ 和 data/ 子目录）
# 影响：设为错误路径会导致 "FileNotFoundError: meta/info.json"
DATASET_ROOT="${DATASET_ROOT:-/data/zhanghang/Videos/datasets/batch5_joint_merged}"

# Pi0.5 预训练基座模型路径（包含 config.json + model.safetensors）
# 影响：不指定则随机初始化（效果差）。通常使用 pi05_base 作为起点进行微调
POLICY_PRETRAINED_PATH="${POLICY_PRETRAINED_PATH:-/data/zhanghang/Videos/0407/model/pi05_base}"

# PaliGemma tokenizer 路径（用于文本 token 化）
# 影响：离线环境必须指向本地路径，在线环境可使用 "google/paligemma-3b-pt-224"
TOKENIZER_NAME="${TOKENIZER_NAME:-/data/zhanghang/Videos/0407/model/paligemma-3b-pt-224}"

# Conda 虚拟环境名称（Evo-RL 项目环境）
# 影响：需确保环境中已安装 lerobot 及其依赖（torch, transformers 等）
CONDA_ENV="${CONDA_ENV:-evo-rl}"

# ============================================================
# 5.2 基础训练参数（TrainPipelineConfig）
# 文件位置：configs/train.py
# 作用：定义训练流程的全局行为——输出管理、训练循环、日志、断点续训
# ============================================================

# --- 输出与任务管理 ---

# 输出根目录。留空则自动生成时间戳目录：outputs/train/<日期>/<时间>_<job_name>
# 调优：设为固定路径便于管理；留空则每次运行自动隔离，防止覆盖
# 注：固定输出目录便于训练后逐个评估中间 checkpoint（见 STEPS 说明）
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/pi05_tuned_bs256_lr2.5e-5_30K}"

# 任务名称。用于区分不同实验，自动出现在输出目录名中
# 调优：建议包含关键超参数信息，如 "pi05_bs4_lr2.5e-5"
# 注：tuned = 依据 0407 全量分析调参（25k 步 / 22k 退火 / 500 预热）
JOB_NAME="${JOB_NAME:-pi05_tuned_bs256_lr2.5e-5_30K}"

# 是否从 checkpoint 断点续训
# 影响：true 时从 output_dir 下最新 checkpoint 恢复模型+优化器状态
# 注意：resume=true 时必须确保 output_dir 指向已有训练目录
RESUME="${RESUME:-false}"

# 随机种子（覆盖 Python/NumPy/PyTorch/CUDA）
# 影响：固定种子可保证结果可复现；改为不同值可探索不同初始化
SEED="${SEED:-1000}"

# --- 训练循环控制 ---

# DataLoader 工作进程数。决定数据加载的并行度
# 调优：设为 CPU 核心数的 2~4 倍；若 GPU 利用率低（数据加载瓶颈），适当增大
# 警告：过大导致内存溢出；设为 0 则主进程同步加载（调试用）
NUM_WORKERS="${NUM_WORKERS:-4}"

# 单步训练的样本数（batch size）。受 GPU 显存强约束
# 调优：H100(80GB) 全参数训练建议 2~4；启用梯度检查点可提至 8~16
# 警告：超出显存会 OOM；过小则训练不稳定、收敛慢
BATCH_SIZE="${BATCH_SIZE:-128}"

# 总训练步数。决定训练时长和模型收敛程度
# ★ 2026-08-14 全量 checkpoint 分析：
#   - 训练 loss 在 ~20k 即平台（0.052→0.041 后不再降）
#   - 预测质量(ADE)在 15k~20k 最优，20k 后反而退化（gripper 双峰退化、ADE 漂移）
#   - 60000 步远超甜点区 → 改为 25000（含少量裕量到达平台）
# 影响：steps 过小欠拟合；过大过拟合且浪费时间
STEPS="${STEPS:-30000}"

# 评估频率（步）。由于本数据集无仿真环境，设为 0 跳过评估
# 调优：有仿真环境时建议 20000~50000；评估耗时较长可增大间隔
EVAL_FREQ="${EVAL_FREQ:-0}"

# 日志打印频率（步）。控制终端和 WandB 的 loss 输出频率
# 调优：值越小日志越密（方便调试但刷屏）；生产环境建议 50~200
LOG_FREQ="${LOG_FREQ:-100}"

# Checkpoint 保存频率（步）。每 save_freq 步保存一次模型
# 调优：频繁保存占用磁盘；间隔过大丢失中间成果。steps 结束时总会强制保存一次
# 建议：5000~20000，根据总步数调整
# ★ 重要：Evo-RL 训练无内置验证集/早停（真实机器人数据）。25k 步产出
#   5k/10k/15k/20k/25k 五个 checkpoint，训练结束后请逐个评估并挑选 ADE
#   最优者部署（历史经验：中间步数通常优于最后一步）。
SAVE_FREQ="${SAVE_FREQ:-5000}"

# 是否保存 checkpoint 文件
# 影响：false 可节省磁盘空间（仅调试时使用），但无法恢复训练或推理
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"

# 是否使用策略自带的优化器/调度器预设（Pi0.5 预置 AdamW + CosineAnnealing）
# 影响：true 时自动读取 PI05Config 中的 optimizer/scheduler 配置
#        false 时需在训练命令中手动传入 --optimizer 和 --scheduler 参数
USE_POLICY_TRAINING_PRESET="${USE_POLICY_TRAINING_PRESET:-true}"

# 训练同步容忍度（秒），用于分布式训练的进程间同步
# 调优：单卡训练保持默认即可；多卡训练网络波动时可适度增大
TOLERANCE_S="${TOLERANCE_S:-0.0001}"

# ============================================================
# 5.3 数据集配置（DatasetConfig）
# 文件位置：configs/default.py
# 作用：指定数据源、筛选及预处理策略
# ============================================================

# 数据集标识符（用于命名和日志，非文件路径；实际路径见 DATASET_ROOT）
# 影响：配合 DATASET_ROOT 使用；本地训练时填写任意有意义的名称即可
DATASET_REPO_ID="${DATASET_REPO_ID:-batch5_joint_merged}"

# 指定使用的 Episode 子集，如 "0,1,2,10,11"
# 影响：留空使用全部 1937 个 episode；指定后仅用部分数据（快速实验/排除损坏轨迹）
DATASET_EPISODES="${DATASET_EPISODES:-}"

# 是否使用 ImageNet 统计值标准化图像（mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]）
# 影响：true 匹配 SigLIP 预训练输入分布（推荐）；false 使用数据集自身统计
USE_IMAGENET_STATS="${USE_IMAGENET_STATS:-true}"

# 是否启用流式传输（边下载边训练，不占用本地磁盘）
# 影响：true 适合超大数据集+良好网络；false 先下载完整数据再训练（推荐）
STREAMING="${STREAMING:-false}"

# ============================================================
# 5.4 WandB 实验追踪配置（WandBConfig）
# 作用：将训练指标同步到 Weights & Biases 云端平台，便于可视化和对比实验
# ============================================================

# 是否启用 WandB 日志记录
# 影响：true 时需网络连接和 wandb 账号；false 仅本地日志
WANDB_ENABLE="${WANDB_ENABLE:-false}"

# WandB 项目名称（所有实验归类到此项目下）
WANDB_PROJECT="${WANDB_PROJECT:-lerobot_pi05}"

# WandB 组织/团队名（留空使用个人账号）
WANDB_ENTITY="${WANDB_ENTITY:-}"

# 实验备注（显示在 WandB 运行列表中，便于区分不同实验）
WANDB_NOTES="${WANDB_NOTES:-pi05 training on batch5_joint_merged}"

# ============================================================
# 5.5 PEFT/LoRA 微调配置（PeftConfig）
# 文件位置：configs/default.py
# 作用：参数高效微调——只训练极少量注入参数，大幅节省显存和训练时间
# 原理：在原模型旁插入低秩矩阵（A: d×r, B: r×d），仅训练 A 和 B
# ============================================================

# 是否启用 PEFT（LoRA）微调
# 影响：true → 只训练约 1~5% 参数，显存需求降低 40~60%，推荐开启
#        false → 全参数训练（效果可能更好但需大显存 + 更多数据防过拟合）
USE_PEFT="${USE_PEFT:-false}"

# PEFT 方法类型。当前主要支持 LORA（Low-Rank Adaptation）
# 可选值：LORA | IA3 | PREFIX_TUNING（需代码确认支持情况）
PEFT_METHOD="${PEFT_METHOD:-LORA}"

# LoRA 秩（Rank）。决定注入的低秩矩阵维度
# 调优：r=8（轻量）~ r=64（强适配）。r 越大可训参数越多、适配能力越强
#       但对小数据集过拟合风险也增大。Pi0.5 建议 r=16
LORA_R="${LORA_R:-16}"

# LoRA 目标模块列表（正则表达式，匹配参数名）。留空使用 Pi0.5 默认值
# 默认：gemma_expert 的 q_proj、v_proj + 投影层（state_proj 等）
# 调优：可扩展至 k_proj、o_proj、mlp 等层增强适配能力（但增加参数量）
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"

# 全参数训练模块列表。这些模块不使用 LoRA，而是完整更新所有权重
# 默认：输出投影层（action_out_proj）、归一化层（LayerNorm）等
# 原因：这些层对数值精度敏感、参数量极小（<1%），全参数训练效果更好
LORA_FULL_TRAINING_MODULES="${LORA_FULL_TRAINING_MODULES:-}"

# ============================================================
# 5.6 Pi0.5 策略专用配置（PI05Config）
# 文件位置：policies/pi05/configuration_pi05.py
# 作用：控制模型架构、训练超参与推理行为
# ============================================================

# --- 基础标识 ---

# 策略类型（必须为 pi05）
POLICY_TYPE="${POLICY_TYPE:-pi05}"

# 是否在训练结束后自动推送到 HuggingFace Hub
# 影响：true 需配置 policy.repo_id 和网络连接；离线环境设为 false
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

# --- 模型架构参数 ---

# PaliGemma VLM 变体。决定视觉语言模型的规模
# 可选值：gemma_300m（轻量、快速）| gemma_2b（更强、更慢、显存更大）
# 影响：2B 版本理解能力更强但训练/推理慢约 2~3 倍
PALIGEMMA_VARIANT="${PALIGEMMA_VARIANT:-gemma_2b}"

# 动作专家（Action Expert）变体。处理动作序列的模型
# 可选值：gemma_300m | gemma_2b（通常 300m 已足够，2B 仅在极端复杂任务中使用）
ACTION_EXPERT_VARIANT="${ACTION_EXPERT_VARIANT:-gemma_300m}"

# 模型计算精度
# 可选值：bfloat16（推荐，显存减半）| float32（稳定，显存大）
# 注：已适配 bfloat16——SigLIP 视觉塔保持 float32，其余部分使用 bfloat16
DTYPE="${DTYPE:-bfloat16}"

# 预测的动作序列长度（chunk_size）。模型一次预测未来 50 步的动作
# 影响：越大则预测范围越长，但计算量增加。50 是 OpenPI 的标准值
CHUNK_SIZE="${CHUNK_SIZE:-50}"

# 每次实际执行的动作步数（≤ chunk_size）
# 影响：执行 n_action_steps 步后重新预测，实现滚动窗口。设为 chunk_size 则完全替换
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"

# 状态向量最大维度（实际数据不足时自动填零）
# 影响：必须 ≥ 数据集中实际状态维度（本数据集为 7）
#       此值必须与预训练模型一致（pi05_base 为 32），否则投影层形状不匹配
MAX_STATE_DIM="${MAX_STATE_DIM:-32}"

# 动作向量最大维度（实际数据不足时自动填零）
# 影响：同上，必须与预训练模型一致（pi05_base 为 32）
MAX_ACTION_DIM="${MAX_ACTION_DIM:-32}"

# 文本 token 的最大长度（用于任务描述）
# 影响：过小截断文本；过大浪费显存。200 为 OpenPI 默认值
TOKENIZER_MAX_LENGTH="${TOKENIZER_MAX_LENGTH:-200}"

# 额外空白摄像头数量。当数据集中摄像头数少于预训练模型时使用
# 影响：值为 N 时自动添加 N 个全零图像通道（以 -1 填充，适配 SigLIP）
#       本数据集有 wrist + top 两个摄像头，预训练模型有 3 个，故设为 1
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"

# 各模态归一化模式（JSON 字符串）。控制数据预处理时的归一化策略
# 可选值：IDENTITY（不做归一化）| MEAN_STD（Z-score）| QUANTILES（分位数归一化）
# 示例：'{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
# 影响：VISUAL 通常 IDENTITY（SigLIP 内置归一化）；STATE/ACTION 可选 MEAN_STD 或 QUANTILES
# 注意：留空则使用 Pi0.5 默认值（VISUAL=IDENTITY, STATE/ACTION=QUANTILES）
#NORMALIZATION_MAPPING="${NORMALIZATION_MAPPING:-}"
# 新代码：显式指定 ACTION/STATE 使用 MEAN_STD，VISUAL 保持 IDENTITY
# ★ 2026-08-25 修复：draccus 无法从裸字符串解析 dict，必须传 JSON 格式（键值加双引号）。
#   支持简洁格式 {ACTION:MEAN_STD,...}（脚本会在构建命令时自动转换为 JSON）。
if [ -z "${NORMALIZATION_MAPPING:-}" ]; then
    NORMALIZATION_MAPPING='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
fi

# ============================================================
# 5.7 优化器参数（AdamW）
# Pi0.5 使用 AdamW 优化器，参考 OpenPI 官方配置
# ============================================================

# 峰值学习率（CosineAnnealing 的起始 LR）
# 调优：微调建议 2.5e-5 ~ 1e-4；从头训练可升至 3e-4
#       过大导致训练不稳定/梯度爆炸；过小收敛极慢
OPTIMIZER_LR="${OPTIMIZER_LR:-2.5e-5}"

# AdamW 数值稳定性常数（防止除零）
# 影响：通常无需调整（1e-8 为标准值）；混合精度训练时可增至 1e-7
OPTIMIZER_EPS="${OPTIMIZER_EPS:-1e-8}"

# L2 正则化强度（Weight Decay）。防止过拟合
# 调优：过拟合时增大（如 0.05~0.1）；欠拟合时减小（如 0.001~0.005）
#       注意：AdamW 的 weight_decay 与学习率解耦，比传统 L2 正则更稳定
# 注：0407 分析显示 20k 后有过拟合式退化，若 25k 训练后仍出现过拟合，
#     可尝试增大至 0.05（本次默认保持 0.01 以隔离步数变量）
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-0.01}"

# 全局梯度裁剪范数（Gradient Clipping）。防止梯度爆炸
# 调优：遇到 loss=NaN 或梯度爆炸时减小（如 0.5）；梯度消失时可增大（如 5.0）
#       设为 0 则完全禁用梯度裁剪（不推荐）
OPTIMIZER_GRAD_CLIP_NORM="${OPTIMIZER_GRAD_CLIP_NORM:-1.0}"

# ============================================================
# 5.8 学习率调度器参数（CosineDecayWithWarmup）
# 策略：线性预热 → 余弦衰减至最终学习率
# 自动缩放：当 steps < scheduler_decay_steps 时自动按比例缩放
# ============================================================

# 预热步数（学习率从 0 线性上升到 optimizer_lr）
# 调优：大数据集保持 1000~2000；小数据集（<10000 步）可降至 500
#       预热有助于训练初期稳定，避免前几步梯度方向偏差过大
# 注：25k 步规模按比例缩小预热至 500（约 2% 步数）
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"

# 余弦退火总步数。学习率从 optimizer_lr 衰减到 scheduler_decay_lr
# ★ 2026-08-14 分析：原 30000 退火 + 60000 步 → 后 3 万步在 5e-6 地板 lr 空转，
#   反而破坏生成质量（gripper 双峰 16%→33%、ADE 漂移）。改为 22000，在 25k 步内
#   完成退火，仅留 ~3k 步低 lr 微调。
# 自动缩放：实际训练步数小于此值时，自动按比例缩放所有调度器参数
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-25000}"

# 最终学习率（余弦退火结束时的 LR）
# 调优：通常设为峰值 LR 的 1/5 ~ 1/10。过小则后期几乎不更新
#       设为与 optimizer_lr 相同时退化为常数 LR
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"

# ============================================================
# 5.9 流匹配（Flow Matching）参数
# 作用：控制扩散过程的噪声调度与去噪策略
# 原理：Pi0.5 使用 Flow Matching 替代 DDPM，直接从噪声预测速度场
# ============================================================

# 推理时的去噪步数（ODE Solver 步数）
# 影响：越大生成质量越高但推理越慢；训练时不影响（仅推理用）
#       10 是 OpenPI 默认值，在速度和质量间取得平衡
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"

# Beta 分布 α 参数（控制时间步采样分布）
# 影响：α > β 时偏向采样小 t（噪声较大的阶段），加强对困难样本的学习
#       通常保持默认 1.5
TIME_SAMPLING_BETA_ALPHA="${TIME_SAMPLING_BETA_ALPHA:-1.5}"

# Beta 分布 β 参数（控制时间步采样分布）
# 影响：通常保持默认 1.0
TIME_SAMPLING_BETA_BETA="${TIME_SAMPLING_BETA_BETA:-1.0}"

# 时间缩放因子（避免 t 恰好为 0 或 1 导致数值问题）
# 影响：通常无需调整。0.999 表示时间范围 [0.001, 0.999]
TIME_SAMPLING_SCALE="${TIME_SAMPLING_SCALE:-0.999}"

# 时间偏移（配合 scale 使用，进一步远离边界）
# 影响：通常无需调整
TIME_SAMPLING_OFFSET="${TIME_SAMPLING_OFFSET:-0.001}"

# 正弦位置编码的最小/最大周期
# 影响：控制时间步嵌入的频率范围。通常保持 OpenPI 默认值
MIN_PERIOD="${MIN_PERIOD:-0.004}"
MAX_PERIOD="${MAX_PERIOD:-4.0}"

# ============================================================
# 5.10 显存与性能优化参数
# 这些参数在显存、速度、精度之间进行权衡
# ============================================================

# 梯度检查点（Gradient Checkpointing）。用计算时间换显存空间
# 影响：true → 节省 40~50% 显存，但训练速度降低约 30%
#       大 Batch 或显存不足时必须开启；显存充裕时关闭可加速
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

# 是否使用 torch.compile 加速模型
# 影响：true → 首次编译需数分钟，后续前向/反向传播加速 10~30%
#       但 compile 可能与某些算子不兼容，建议模型稳定后再开启
COMPILE_MODEL="${COMPILE_MODEL:-false}"

# torch.compile 的编译模式
# 可选值：max-autotune（最优化，编译慢但运行快）| reduce-overhead（平衡）
#         | default（快速编译，加速较少）
COMPILE_MODE="${COMPILE_MODE:-max-autotune}"

# 冻结视觉编码器（SigLIP Vision Tower）
# 影响：true → 冻结约 300M 参数，显著减少显存和训练时间
#       当视觉场景变化不大时（固定机位），冻结不会明显影响效果
#       当场景变化大（不同光照/背景）时，取消冻结可提升泛化
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"

# 仅训练动作专家（Action Expert）+ 投影层
# 影响：true → 冻结整个 VLM（PaliGemma），只训练动作专家和输入/输出投影层
#       极省显存（可训参数从 3.5B 降至 ~577M），适合微调场景
#       若任务与预训练数据差异较大，设为 false 解冻更多参数
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"

# ============================================================
# 5.11 RA-BC（奖励加权行为克隆）参数
# 作用：从次优演示数据中优先学习高回报行为
# 原理：根据样本的奖励值对 loss 施加权重，鼓励模仿高质量轨迹
# 注意：当前数据集无奖励标注，默认关闭
# ============================================================

# 是否启用 RA-BC
# 影响：启用后自动加载 rabc_progress_path 中的预计算奖励权重
#       需要数据集包含奖励信息或预计算的 progress 文件
USE_RABC="${USE_RABC:-false}"

# 硬阈值分位数（取值范围 0~1）。决定选取前 kappa*100% 的高质量样本
# 例如 0.01 表示只对奖励最高的前 1% 样本施加高权重
RABC_KAPPA="${RABC_KAPPA:-0.01}"

# 数值稳定性常数。计算权重时防止分母为零
RABC_EPSILON="${RABC_EPSILON:-1e-6}"

# 双头模型模式。sparse：仅最后一步预测奖励；dense：每帧都预测
# 影响：sparse 计算量更低，通常足够
RABC_HEAD_MODE="${RABC_HEAD_MODE:-sparse}"

# ============================================================
# 5.12 数据集特征重命名映射
# 作用：当数据集特征名与模型期望不一致时进行映射
# 格式："数据集key:模型key,数据集key:模型key"
# 示例："observation.images.wrist:observation.images.cam1"
# ============================================================

RENAME_MAP="${RENAME_MAP:-}"

# ==============================================================================
# 脚本逻辑（无需修改）
# ==============================================================================

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
SEP="============================================================"
SEP2="------------------------------------------------------------"

echo ""
echo "$SEP"
echo "  Pi0.5 模型训练脚本"
echo "  启动时间: $TIMESTAMP"
echo "$SEP"

# 打印参数摘要表
echo ""
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │              训 练 参 数 摘 要                  │"
echo "  ├─────────────────────────────────────────────────┤"
printf "  │ %-25s %-24s │\n" "  项目目录" "$(basename "$PROJECT_DIR")"
printf "  │ %-25s %-24s │\n" "  数据集" "$(basename "$DATASET_ROOT")"
printf "  │ %-25s %-24s │\n" "  预训练模型" "$(basename "$POLICY_PRETRAINED_PATH")"
printf "  │ %-25s %-24s │\n" "  输出目录" "${OUTPUT_DIR:-自动生成}"
echo "  ├─────────────────────────────────────────────────┤"
printf "  │ %-25s %-24s │\n" "  训练步数 (steps)" "$STEPS"
printf "  │ %-25s %-24s │\n" "  批次大小 (per GPU)" "$BATCH_SIZE"
if [ "${NUM_GPUS:-1}" -gt 1 ]; then
    EFFECTIVE_BS=$((BATCH_SIZE * NUM_GPUS))
    printf "  │ %-25s %-24s │\n" "  GPU 数量" "$NUM_GPUS (有效BS=$EFFECTIVE_BS)"
else
    printf "  │ %-25s %-24s │\n" "  GPU 数量" "${NUM_GPUS:-1}"
fi
printf "  │ %-25s %-24s │\n" "  数据精度 (dtype)" "$DTYPE"
printf "  │ %-25s %-24s │\n" "  数据加载进程 (workers)" "$NUM_WORKERS"
echo "  ├─────────────────────────────────────────────────┤"
printf "  │ %-25s %-24s │\n" "  使用 PEFT/LoRA" "$USE_PEFT"
[ "$USE_PEFT" = true ] && printf "  │ %-25s %-24s │\n" "    LoRA 秩 (r)" "$LORA_R"
printf "  │ %-25s %-24s │\n" "  梯度检查点" "$GRADIENT_CHECKPOINTING"
printf "  │ %-25s %-24s │\n" "  冻结视觉编码器" "$FREEZE_VISION_ENCODER"
printf "  │ %-25s %-24s │\n" "  仅训练动作专家" "$TRAIN_EXPERT_ONLY"
echo "  ├─────────────────────────────────────────────────┤"
printf "  │ %-25s %-24s │\n" "  VLM 变体" "$PALIGEMMA_VARIANT"
printf "  │ %-25s %-24s │\n" "  动作专家变体" "$ACTION_EXPERT_VARIANT"
printf "  │ %-25s %-24s │\n" "  预测动作长度 (chunk)" "$CHUNK_SIZE"
printf "  │ %-25s %-24s │\n" "  状态/动作维度" "${MAX_STATE_DIM}/${MAX_ACTION_DIM}"
echo "  ├─────────────────────────────────────────────────┤"
printf "  │ %-25s %-24s │\n" "  峰值学习率 (lr)" "$OPTIMIZER_LR"
printf "  │ %-25s %-24s │\n" "  最终学习率" "$SCHEDULER_DECAY_LR"
printf "  │ %-25s %-24s │\n" "  预热步数" "$SCHEDULER_WARMUP_STEPS"
printf "  │ %-25s %-24s │\n" "  余弦退火步数" "$SCHEDULER_DECAY_STEPS"
printf "  │ %-25s %-24s │\n" "  权重衰减" "$OPTIMIZER_WEIGHT_DECAY"
printf "  │ %-25s %-24s │\n" "  梯度裁剪范数" "$OPTIMIZER_GRAD_CLIP_NORM"
echo "  ├─────────────────────────────────────────────────┤"
printf "  │ %-25s %-24s │\n" "  日志频率 (步)" "$LOG_FREQ"
printf "  │ %-25s %-24s │\n" "  保存频率 (步)" "$SAVE_FREQ"
printf "  │ %-25s %-24s │\n" "  WandB 日志" "$WANDB_ENABLE"
printf "  │ %-25s %-24s │\n" "  RA-BC" "$USE_RABC"
printf "  │ %-25s %-24s │\n" "  随机种子 (seed)" "$SEED"
echo "  └─────────────────────────────────────────────────┘"
echo ""

# ---- 阶段 1：环境准备 ----
echo "[1/4] 环境准备"
echo "$SEP2"

cd "$PROJECT_DIR" || { echo "  ✗ 错误: 无法进入项目目录 $PROJECT_DIR"; exit 1; }
echo "  工作目录: $(pwd)"

# 激活 conda 环境
source /data/zhanghang/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "$CONDA_ENV" || { echo "  ✗ 错误: 无法激活 conda 环境 $CONDA_ENV"; exit 1; }
echo "  Conda 环境: $CONDA_ENV"
echo "  Python:     $(which python)"
echo "  PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA 可用:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU 数量:   $(python -c 'import torch; print(torch.cuda.device_count())')" 2>/dev/null || echo "  GPU:        检测失败"
echo "  GPU 型号:   $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"

# 设置离线模式（避免 HuggingFace 网络超时）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "  网络模式:   离线 (HF_HUB_OFFLINE=1)"

# 自动清理已存在的输出目录（避免多卡训练时多进程同时报 FileExistsError）
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ] && [ "$RESUME" != "true" ]; then
    echo "  ⚠ 输出目录已存在，自动清理: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi
echo ""

# ---- 阶段 2：构建命令 ----
echo "[2/4] 构建训练命令"
echo "$SEP2"

# 根据 GPU 数量决定使用 accelerate（多卡）还是 lerobot-train（单卡）
if [ "${NUM_GPUS:-1}" -gt 1 ]; then
    # 多卡分布式训练：使用 HuggingFace Accelerate
    # --num_machines=1 --mixed_precision=no --dynamo_backend=no 消除 DDP 默认值警告
    TRAIN_CMD="accelerate launch --multi_gpu --num_processes=$NUM_GPUS --num_machines=1 --mixed_precision=no --dynamo_backend=no -m src.lerobot.scripts.lerobot_train"
    echo "  训练模式:   多卡分布式（accelerate launch, ${NUM_GPUS} GPUs）"
else
    # 单卡训练：使用 python -m 方式启动（兼容 Evo-RL 源安装）
    TRAIN_CMD="python -m src.lerobot.scripts.lerobot_train"
    echo "  训练模式:   单卡（python -m src.lerobot.scripts.lerobot_train）"
fi

CMD="$TRAIN_CMD"

# --- normalization_mapping 自动转换（draccus 需要 JSON dict，不接受裸字符串）---
# 支持两种输入格式，统一转为 JSON：
#   1. JSON:      {"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}
#   2. 简洁格式:  {ACTION:MEAN_STD,STATE:MEAN_STD,VISUAL:IDENTITY}
# 若输入以单引号/双引号包裹，先剥除外壳再转换。
if [ -n "$NORMALIZATION_MAPPING" ]; then
    NORMALIZATION_MAPPING=$(python - "$NORMALIZATION_MAPPING" <<'PYEOF'
import json, sys
s = sys.argv[1].strip()
# 剥除外层引号外壳
if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
    s = s[1:-1].strip()
if not s:
    sys.exit(0)
try:
    d = json.loads(s)
except Exception:
    # 简洁格式 {K:V,K:V} -> JSON
    inner = s.strip('{}').strip()
    items = [p.strip() for p in inner.split(',')] if inner else []
    d = {}
    for it in items:
        k, v = [p.strip() for p in it.split(':', 1)]
        d[k] = v
print(json.dumps(d))
PYEOF
)
    echo "  normalization_mapping: $NORMALIZATION_MAPPING"
fi

# --- 基础训练参数 ---
[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output_dir=$OUTPUT_DIR"
[ -n "$JOB_NAME" ] && CMD="$CMD --job_name=$JOB_NAME"
[ "$RESUME" = true ] && CMD="$CMD --resume=true"

# 续训时自动查找 checkpoint 中的 train_config.json
if [ "$RESUME" = true ] && [ -n "$OUTPUT_DIR" ]; then
    CKPT_DIR="$OUTPUT_DIR/checkpoints"
    if [ -d "$CKPT_DIR" ]; then
        # 找数字最大的 checkpoint 目录
        LATEST_CKPT=$(ls -d "$CKPT_DIR"/*/ 2>/dev/null | grep -v "/last/" | sort -V | tail -1 | sed 's|/$||')
        if [ -n "$LATEST_CKPT" ]; then
            CONFIG_PATH="$LATEST_CKPT/pretrained_model/train_config.json"
            if [ -f "$CONFIG_PATH" ]; then
                CMD="$CMD --config_path=$CONFIG_PATH"
                echo "  续训配置:    $CONFIG_PATH"
            fi
        fi
    fi
fi

[ -n "$SEED" ] && CMD="$CMD --seed=$SEED"
[ -n "$NUM_WORKERS" ] && CMD="$CMD --num_workers=$NUM_WORKERS"
[ -n "$BATCH_SIZE" ] && CMD="$CMD --batch_size=$BATCH_SIZE"
[ -n "$STEPS" ] && CMD="$CMD --steps=$STEPS"
[ -n "$EVAL_FREQ" ] && CMD="$CMD --eval_freq=$EVAL_FREQ"
[ -n "$LOG_FREQ" ] && CMD="$CMD --log_freq=$LOG_FREQ"
[ -n "$SAVE_FREQ" ] && CMD="$CMD --save_freq=$SAVE_FREQ"
[ "$SAVE_CHECKPOINT" = false ] && CMD="$CMD --save_checkpoint=false"
[ "$USE_POLICY_TRAINING_PRESET" = false ] && CMD="$CMD --use_policy_training_preset=false"
[ -n "$TOLERANCE_S" ] && CMD="$CMD --tolerance_s=$TOLERANCE_S"

# --- 数据集参数 ---
CMD="$CMD --dataset.repo_id=$DATASET_REPO_ID"
CMD="$CMD --dataset.root=$DATASET_ROOT"
[ -n "$DATASET_EPISODES" ] && CMD="$CMD --dataset.episodes=$DATASET_EPISODES"
[ "$USE_IMAGENET_STATS" = false ] && CMD="$CMD --dataset.use_imagenet_stats=false"
[ "$STREAMING" = true ] && CMD="$CMD --dataset.streaming=true"

# --- WandB 参数 ---
if [ "$WANDB_ENABLE" = true ]; then
    CMD="$CMD --wandb.enable=true"
    [ -n "$WANDB_PROJECT" ] && CMD="$CMD --wandb.project=$WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ] && CMD="$CMD --wandb.entity=$WANDB_ENTITY"
    [ -n "$WANDB_NOTES" ] && CMD="$CMD --wandb.notes=$WANDB_NOTES"
fi

# --- PEFT 参数 ---
if [ "$USE_PEFT" = true ]; then
    CMD="$CMD --peft.method_type=$PEFT_METHOD"
    CMD="$CMD --peft.r=$LORA_R"
    [ -n "$LORA_TARGET_MODULES" ] && CMD="$CMD --peft.target_modules=$LORA_TARGET_MODULES"
    [ -n "$LORA_FULL_TRAINING_MODULES" ] && CMD="$CMD --peft.full_training_modules=$LORA_FULL_TRAINING_MODULES"
fi

# --- Pi0.5 策略参数 ---
CMD="$CMD --policy.type=$POLICY_TYPE"
CMD="$CMD --policy.pretrained_path=$POLICY_PRETRAINED_PATH"
CMD="$CMD --policy.paligemma_variant=$PALIGEMMA_VARIANT"
CMD="$CMD --policy.action_expert_variant=$ACTION_EXPERT_VARIANT"
CMD="$CMD --policy.dtype=$DTYPE"
CMD="$CMD --policy.chunk_size=$CHUNK_SIZE"
CMD="$CMD --policy.n_action_steps=$N_ACTION_STEPS"
CMD="$CMD --policy.max_state_dim=$MAX_STATE_DIM"
CMD="$CMD --policy.max_action_dim=$MAX_ACTION_DIM"
CMD="$CMD --policy.tokenizer_max_length=$TOKENIZER_MAX_LENGTH"
CMD="$CMD --policy.empty_cameras=$EMPTY_CAMERAS"
# tokenizer_name: PI05Config 现已支持该字段（本地路径离线加载）。
# 默认指向本地 /data/zhanghang/Videos/0407/model/paligemma-3b-pt-224，可通过 TOKENIZER_NAME 覆盖。
CMD="$CMD --policy.tokenizer_name=$TOKENIZER_NAME"
CMD="$CMD --policy.device=$POLICY_DEVICE"
CMD="$CMD --policy.repo_id=$POLICY_REPO_ID"
[ "$PUSH_TO_HUB" = false ] && CMD="$CMD --policy.push_to_hub=false"
[ -n "$NORMALIZATION_MAPPING" ] && CMD="$CMD --policy.normalization_mapping='$NORMALIZATION_MAPPING'"

# --- 优化器参数 ---
CMD="$CMD --policy.optimizer_lr=$OPTIMIZER_LR"
CMD="$CMD --policy.optimizer_eps=$OPTIMIZER_EPS"
CMD="$CMD --policy.optimizer_weight_decay=$OPTIMIZER_WEIGHT_DECAY"
CMD="$CMD --policy.optimizer_grad_clip_norm=$OPTIMIZER_GRAD_CLIP_NORM"

# --- 调度器参数 ---
CMD="$CMD --policy.scheduler_warmup_steps=$SCHEDULER_WARMUP_STEPS"
CMD="$CMD --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS"
CMD="$CMD --policy.scheduler_decay_lr=$SCHEDULER_DECAY_LR"

# --- 流匹配参数 ---
CMD="$CMD --policy.num_inference_steps=$NUM_INFERENCE_STEPS"
CMD="$CMD --policy.time_sampling_beta_alpha=$TIME_SAMPLING_BETA_ALPHA"
CMD="$CMD --policy.time_sampling_beta_beta=$TIME_SAMPLING_BETA_BETA"
CMD="$CMD --policy.time_sampling_scale=$TIME_SAMPLING_SCALE"
CMD="$CMD --policy.time_sampling_offset=$TIME_SAMPLING_OFFSET"
CMD="$CMD --policy.min_period=$MIN_PERIOD"
CMD="$CMD --policy.max_period=$MAX_PERIOD"

# --- 性能优化参数 ---
[ "$GRADIENT_CHECKPOINTING" = true ] && CMD="$CMD --policy.gradient_checkpointing=true"
CMD="$CMD --policy.compile_model=$COMPILE_MODEL"
[ "$COMPILE_MODEL" = true ] && [ -n "$COMPILE_MODE" ] && CMD="$CMD --policy.compile_mode=$COMPILE_MODE"
[ "$FREEZE_VISION_ENCODER" = true ] && CMD="$CMD --policy.freeze_vision_encoder=true"
[ "$TRAIN_EXPERT_ONLY" = true ] && CMD="$CMD --policy.train_expert_only=true"

# --- RA-BC 参数 ---
if [ "$USE_RABC" = true ]; then
    CMD="$CMD --use_rabc=true"
    [ -n "$RABC_KAPPA" ] && CMD="$CMD --rabc_kappa=$RABC_KAPPA"
    [ -n "$RABC_EPSILON" ] && CMD="$CMD --rabc_epsilon=$RABC_EPSILON"
    [ -n "$RABC_HEAD_MODE" ] && CMD="$CMD --rabc_head_mode=$RABC_HEAD_MODE"
fi

# --- ACP 参数（价值模型驱动的自动课程规划）---
if [ "$ACP_ENABLE" = true ]; then
    CMD="$CMD --acp.enable=true"
    CMD="$CMD --acp.indicator_field=$ACP_INDICATOR_FIELD"
    CMD="$CMD --acp.indicator_dropout_prob=$ACP_INDICATOR_DROPOUT_PROB"
    echo "  ACP:          启用 (indicator=$ACP_INDICATOR_FIELD, dropout=$ACP_INDICATOR_DROPOUT_PROB)"
fi

# --- 数据集特征重命名 ---
if [ -n "$RENAME_MAP" ]; then
    IFS=',' read -ra MAPPINGS <<< "$RENAME_MAP"
    for MAPPING in "${MAPPINGS[@]}"; do
        KEY="${MAPPING%%:*}"
        VALUE="${MAPPING#*:}"
        CMD="$CMD --rename_map.$KEY=$VALUE"
    done
fi

echo "  策略类型:    $POLICY_TYPE"
echo "  任务名称:    $JOB_NAME"
echo "  训练模式:    $(if [ "${NUM_GPUS:-1}" -gt 1 ]; then echo "多卡 (${NUM_GPUS} GPUs, accelerate launch)"; else echo "单卡 (lerobot-train)"; fi)"
echo "  CLI 参数数:  $(echo "$CMD" | wc -w) 个"
echo ""

# ---- 阶段 3：开始训练 ----
echo "[3/4] 启动训练"
echo "$SEP2"
echo ""
echo "  ⚠ 模型加载约需 60~90 秒（3.5B 参数），期间无输出属正常现象"
echo "  ⚠ 若 5 分钟后仍无输出，请检查 GPU 显存是否充足（需 ≥ 16 GB）"
if [ "${NUM_GPUS:-1}" -gt 1 ]; then
    echo "  ⚠ 多卡训练：有效 batch_size = $BATCH_SIZE × $NUM_GPUS = $((BATCH_SIZE * NUM_GPUS))"
fi
echo ""
echo "  ──────────── 训练命令（可复制单独执行）────────────"
echo "  $CMD"
echo "  ────────────────────────────────────────────────────"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 执行训练（将所有输出直接透传到终端，不做过滤）
eval "$CMD" 2>&1
EXIT_CODE=$?

# 计算耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

# 格式化耗时
if [ $HOURS -gt 0 ]; then
    TIME_STR="${HOURS}时${MINUTES}分${SECONDS}秒"
elif [ $MINUTES -gt 0 ]; then
    TIME_STR="${MINUTES}分${SECONDS}秒"
else
    TIME_STR="${SECONDS}秒"
fi

# ---- 阶段 4：训练结束 ----
echo ""
echo "[4/4] 训练结束"
echo "$SEP"
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓  训练成功完成！"
else
    echo "  ✗  训练异常退出，退出码: $EXIT_CODE"
fi
echo "     总耗时: $TIME_STR"
echo "     启动时间: $TIMESTAMP"
echo "     结束时间: $(date "+%Y-%m-%d %H:%M:%S")"

# 检查输出目录
if [ -n "$OUTPUT_DIR" ]; then
    CHECK_DIR="$OUTPUT_DIR"
else
    # 找到实际生成的输出目录
    CHECK_DIR=$(ls -dt outputs/train/*/ 2>/dev/null | head -1)
fi

if [ -d "$CHECK_DIR" ]; then
    echo "     输出目录: $CHECK_DIR"
    if [ -d "$CHECK_DIR/checkpoints" ]; then
        LATEST=$(ls "$CHECK_DIR/checkpoints" 2>/dev/null | grep -v last | sort -n | tail -1)
        echo "     最新 checkpoint: checkpoints/$LATEST"
    fi
fi
echo ""
echo "$SEP"
echo ""

exit $EXIT_CODE
