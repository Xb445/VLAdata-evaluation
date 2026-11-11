# Evo-RL 项目环境部署指南

本目录用于在**另一台电脑**上快速复现当前开发机的 `evo-rl-videos` conda 环境（ACT + VLM 编排器、Piper 机械臂遥操作、数据采集/训练所需的完整 Python 环境）。

## 目录内容

| 文件 | 说明 |
| --- | --- |
| `setup_evo_rl_videos.sh` | 一键部署脚本（自动创建 conda 环境 + 安装全部依赖） |
| `requirements-evo-rl-videos.txt` | 依赖版本锁定清单（由源开发机 `pip freeze` 生成） |

## 源环境信息（已在源开发机确认）

| 项目 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu (Linux x86_64) |
| conda 环境名 | `evo-rl-videos` |
| Python | 3.10.12 (conda-forge) |
| PyTorch | 2.7.1 + torchvision 0.22.1 + torchcodec 0.5（**CUDA 12.6 构建**） |
| lerobot | 0.4.4（本地 editable，来自本项目） |
| transformers | 4.53.3（`fix/lerobot_openpi` 分支，本地目录） |
| piper_sdk | 0.6.1（本地 `src/piper_sdk`） |
| VLM 推理 | vLLM 0.21.0（独立环境 `vllm-eval`，可选） |

> ⚠️ **重要**：本项目包含大量**未提交到上游 git 的本地代码**（`src/lerobot/act_vlm_orchestrator_v2/`、`src/piper_sdk/`、`transformers/`、`src/lerobot/robots/single_piper/`、`src/lerobot/inference_new_*.py` 等）。因此**必须拷贝整个项目目录**到目标机器，不能只 `git clone` 上游仓库。

---

## 方式一：推荐 — 整目录拷贝 + 一键脚本

### 第 1 步：在源开发机上打包项目（排除大数据目录）

```bash
cd /home/ub/Videos/Evo-RL
rsync -a --exclude '.git' --exclude '.conda' --exclude 'outputs' \
      --exclude 'datasets' --exclude 'act_vlm_logs' --exclude '*.zip' \
      ./ /path/to/transfer/Evo-RL/
```

或打包成压缩包方便传输：

```bash
cd /home/ub
tar czf evo-rl-project.tar.gz \
    --exclude='.git' --exclude='.conda' --exclude='outputs' \
    --exclude='datasets' --exclude='act_vlm_logs' --exclude='*.zip' \
    Videos/Evo-RL
# 拷贝 evo-rl-project.tar.gz 到目标机器后:
# tar xzf evo-rl-project.tar.gz -C /path/to/dest
```

> `datasets/`（遥操作数据）体积较大且非运行必需，可按需单独拷贝。

### 第 2 步：在目标机器上运行部署脚本

```bash
cd /path/to/dest/Evo-RL/deploy
bash setup_evo_rl_videos.sh                # 标准部署（GPU/CUDA，自动检测网络）
```

常用参数：

```bash
bash setup_evo_rl_videos.sh --no-cuda      # 无 GPU 机器（CPU 版 PyTorch）
bash setup_evo_rl_videos.sh --with-vllm    # 同时部署 vllm-eval 环境（VLM 推理服务）
bash setup_evo_rl_videos.sh --skip-system  # 跳过 apt 系统包（无 sudo 时）
bash setup_evo_rl_videos.sh --env-name myenv  # 自定义环境名
bash setup_evo_rl_videos.sh --mirror tsinghua # 指定清华 pip + 阿里 torch 镜像
bash setup_evo_rl_videos.sh --mirror none     # 强制官方源（不走镜像）
bash setup_evo_rl_videos.sh --mirror auto     # 自动检测网络（默认，推荐）
```

> **国内网络环境**：默认 `--mirror auto` 会自动探测 `pypi.org` / `download.pytorch.org`，不可达时自动切换为**清华 pip 镜像 + 国内 PyTorch 镜像**并配置 conda 清华源，无需手动干预。也可用 `--mirror tsinghua` / `--mirror aliyun` 手动指定。
>
> **关于 PyTorch 国内镜像**：阿里云等 PyTorch 镜像（`pytorch-wheels/cu126`）是**扁平目录结构**，不提供 pip 标准索引页，因此脚本使用 `--find-links` 方式下载 torch/torchvision/torchcodec 三件套（其余依赖仍走 pip 索引）。脚本会自动探测候选镜像（阿里云 → 上交 → 南大 → 北外）中三个 wheel 是否齐全，选用第一个可用的。

脚本会自动完成：检查/安装 Miniconda → 创建 `evo-rl-videos` 环境（Python 3.10 + ffmpeg/casadi/pinocchio）→ 安装 apt 系统依赖（can-utils、ffmpeg、libgl1 等）→ 安装 CUDA 版 PyTorch → 按锁定清单安装全部 pip 依赖 → 从源码安装 lerobot / transformers / piper_sdk → 验证关键包与 CUDA。

### 第 3 步：验证

```bash
conda activate evo-rl-videos
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import lerobot, transformers, piper_sdk, qwen_vl_utils; print('OK')"
```

---

## 方式二：无项目源码（仅脚本）— 不完全推荐

如果目标机器上只有 `deploy/` 目录（没有项目源码），脚本会尝试 `git clone` 上游 `MINT-SJTU/Evo-RL` 仓库，但**缺少所有本地自定义代码**，脚本会提示后退出。此时请先按方式一拷贝完整项目。

---

## 方式三：手动一步步安装（不使用脚本）

> 若希望完全手动控制每一步，可按以下顺序执行，最终产出与一键脚本相同的 `evo-rl-videos` 环境（版本见上方「源环境信息」表）。

### 前置：拷贝项目代码

先把整个 Evo-RL 项目（含 `deploy/`、`transformers/`、`src/piper_sdk/` 等本地代码）拷贝到目标机器，打包/传输方法见[方式一第 1 步](#方式一推荐--整目录拷贝--一键脚本)。

### 第 1 步：安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y can-utils iproute2 usbutils ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 v4l-utils
```

> 无 sudo 权限时请让管理员执行，或用 `sudo -s` 提权。

### 第 2 步：安装 / 确认 conda

未安装 conda 时安装 Miniconda（内含 Python 3.10）：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
# 加入 PATH（重开终端或执行一次）
echo '. "$HOME/miniconda3/etc/profile.d/conda.sh"' >> ~/.bashrc
source ~/.bashrc
```

> 已装 Anaconda/Miniconda 则跳过；若 `conda` 不在 PATH，先执行 `source <conda目录>/etc/profile.d/conda.sh`。

### 第 3 步：创建 conda 环境（Python 3.10 + 系统级依赖）

```bash
conda create -n evo-rl-videos python=3.10 -c conda-forge -y
conda activate evo-rl-videos
conda install -n evo-rl-videos -c conda-forge -y \
    ffmpeg casadi pinocchio-python coal-python
```

> 国内网络连不上 conda-forge 时，先按文末「可选：配置镜像」配置 conda 清华源再执行。

### 第 4 步：升级 pip 并安装 PyTorch 2.7.1（CUDA 12.6）

```bash
python -m pip install --upgrade pip setuptools wheel
```

**方式 A — 官方源**（可直连 `download.pytorch.org` 时）：

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchcodec==0.5 \
    --index-url https://download.pytorch.org/whl/cu126
```

**方式 B — 国内镜像**（推荐；阿里镜像为扁平目录，必须用 `--find-links`）：

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchcodec==0.5 \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cu126
```

> 无 GPU 时改用 CPU 版：`--index-url https://download.pytorch.org/whl/cpu`，或 `--find-links https://mirrors.aliyun.com/pytorch-wheels/cpu`。
> 其余依赖（nvidia-*、filelock 等）由 pip 从配置的索引自动获取。

### 第 5 步：安装依赖锁定清单

```bash
cd <项目目录>/deploy
pip install -r requirements-evo-rl-videos.txt
```

> 国内网络可先 `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` 再执行。

### 第 6 步：从源码安装 lerobot / transformers / piper_sdk

```bash
cd <项目目录>

# 1) lerobot（项目本体，editable；依赖已在第 5 步装好，--no-deps 避免重复解析）
pip install -e . --no-deps

# 2) transformers（fix/lerobot_openpi 分支，本地目录）
cd transformers && pip install -e . --no-deps && cd ..

# 3) piper_sdk（Piper 机械臂 SDK，本地目录）
cd src/piper_sdk && pip install -e . && cd ../..
```

> 若未拷贝本地 `transformers/` 目录，可用 git 安装同一提交：
> ```bash
> pip install -e "git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0#egg=transformers" --no-deps
> ```

### 第 7 步：验证

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import lerobot, transformers, piper_sdk, qwen_vl_utils, rerun_sdk; print('OK')"
```

### 可选：配置镜像（国内网络）

**pip 清华源**（对后续所有 pip 安装生效）：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**conda 清华源**：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --set show_channel_urls yes
```

### 可选：vllm-eval 环境（VLM 推理服务）

```bash
conda create -n vllm-eval python=3.10 -c conda-forge -y
conda activate vllm-eval
pip install "vllm==0.21.0" "openai"
```

---

## VLM 推理服务（vllm-eval，可选）

ACT 编排器依赖独立的 vLLM API 服务。两种做法：

**A. 在目标机器上同时部署**（推荐）：

```bash
bash setup_evo_rl_videos.sh --with-vllm
```

**B. 使用远程/现有 vLLM 服务**：编排器通过 `VLLM_URL`（默认 `http://localhost:8000/v1`）访问，可指向任意机器。

启动 VLM 服务（模型文件需自行拷贝，如 `qwen3_vl_2b_parcel_subtask_merged`）：

```bash
conda activate vllm-eval
VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/qwen3_vl_2b_parcel_subtask_merged \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.7 --max-model-len 8192 --max-num-seqs 4 \
  --dtype auto --trust-remote-code --served-model-name qwen3-vl-2b
```

> 若目标机器驱动为 CUDA 13.x 且需要更新的 vLLM，可调整 `setup_evo_rl_videos.sh` 顶部的 `VLLM_VERSION`。

---

## 启动编排器

```bash
# 1. 确认 vLLM 服务已启动（见上）
# 2. 连接相机（top=10, wrist=4）与 Piper 机械臂（CAN）
# 3. 运行：
conda activate evo-rl-videos
bash /path/to/Evo-RL/src/lerobot/act_vlm_orchestrator_v2/start_orchestrator.sh
```

---

## 常见问题

| 问题 | 处理 |
| --- | --- |
| 报错 `NewConnectionError ... pypi.org ... 网络不可达` | 国内网络无法直连官方源，脚本默认 `--mirror auto` 会自动切换清华 pip + 阿里 torch 镜像；也可手动 `--mirror tsinghua` |
| 报错 `HTTPSConnection ... download.pytorch.org ... 网络不可达` | 同上，torch 会自动走阿里镜像 `https://mirrors.aliyun.com/pytorch-wheels/cu126` |
| 环境创建失败（conda-forge 连不上） | 配置清华源：`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge`，并给 conda 添加 `default_channels` 镜像后重试；或直接 `--mirror tsinghua` |
| 报错 `.../envs/xxx/bin/python：没有那个文件或目录` | 旧版脚本硬编码了环境路径；已优化为通过 `conda env list` 自动查询真实路径。若仍出现，请检查 `conda env list` 中环境是否存在、以及 `~/.condarc` 的 `envs_dirs` 配置 |
| 无 sudo 权限 | 加 `--skip-system`，再手动安装 `can-utils ffmpeg libgl1 libglib2.0-0` 等 |
| 安装很慢 | 已自动走镜像；若需自定义可 `export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` |
| 目标机器 GPU 驱动过旧（<12.6） | 先升级 NVIDIA 驱动；或改用 CUDA 11.8 版 torch（需同步调整依赖清单） |
| 相机/CAN 设备无权限 | 将用户加入 `dialout` 组：`sudo usermod -aG dialout $USER`，重登后生效 |
| 环境名冲突 | 使用 `--env-name` 指定其他名称 |
