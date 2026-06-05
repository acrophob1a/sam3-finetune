# 基于 SAM3 的语言指令分割微调 — 项目操作指南

> **给 Cursor 的说明**：本文档是本项目的唯一权威操作手册。读取后请按「当前进度检查」逐项核对服务器状态，从第一个未完成步骤继续执行。执行命令前务必 `cd` 到项目根目录（含 `pyproject.toml` 和 `sam3/` 的目录）。改动代码时遵循现有风格，范围尽量小；路径、GPU 数量、数据集名称须全文保持一致。

---

## 项目目标

在 SAM3 基座模型上，用 **(图像, 文本指令, 分割 mask)** 三元组做微调，使模型能根据自然语言描述分割目标物体。

## 完整流程概览

```
下载代码 → 安装环境 → 下载权重 → 准备图片 → 生成训练数据 → 修改训练配置 → 启动微调
```

## 关键路径（全文统一使用以下命名）

| 用途 | 路径 |
|------|------|
| 项目根目录 | `/root/autodl-tmp/sam3`（或你实际解压的位置） |
| SAM3 权重 | `pretrained/sam3/sam3.pt` |
| Qwen2.5-VL 权重 | `pretrained/Qwen2.5-VL-7B-Instruct` |
| 原始训练图片 | `datasets/raw_images/` |
| 生成的训练数据 | `datasets/custom0/` |
| COCO 标注文件 | `datasets/custom0/annotations.json` |
| 训练配置 | `sam3/train/configs/mydata/text_only_train.yaml` |
| 训练产出 | `workdir/<run_name>/checkpoints/` |

---

## 当前进度检查

Cursor 接手时，先运行以下命令判断从哪一步开始：

```bash
# 应在项目根目录执行
cd /root/autodl-tmp/sam3   # 改成实际路径

# 1. 环境是否就绪
python -c "import sam3; import torch; print('sam3 OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 2. 权重是否下载
ls pretrained/sam3/sam3.pt
ls pretrained/Qwen2.5-VL-7B-Instruct/config.json

# 3. 训练数据是否生成
ls datasets/custom0/annotations.json

# 4. 是否已有训练产出
ls workdir/
```

| 检查项 | 通过条件 | 对应步骤 |
|--------|----------|----------|
| 代码 | 存在 `sam3/`、`pyproject.toml` | §1 |
| 环境 | `import sam3` 无报错 | §3 |
| SAM3 权重 | `pretrained/sam3/sam3.pt` 存在 | §4.1 |
| Qwen 权重 | `pretrained/Qwen2.5-VL-7B-Instruct/` 存在 | §4.2 |
| 原始图片 | `datasets/raw_images/` 下有图片 | §5 |
| 训练数据 | `datasets/custom0/annotations.json` 存在 | §6 |
| 训练配置 | yaml 路径与 GPU 数已对齐 | §7 |
| 微调完成 | `workdir/<run_name>/checkpoints/` 有权重 | §8 |

---

## §1 下载代码仓库

自行设定路径变量，下载并解压：

```bash
your_local_dir=/root/autodl-tmp
your_root=sam3

hf download Brilliant-B/awesome-demos demo2.tar.gz --local-dir ${your_local_dir}
tar -xvf ${your_local_dir}/demo2.tar.gz -C ${your_local_dir}/${your_root}
cd ${your_local_dir}/${your_root}
```

验证：`ls pyproject.toml sam3/train/train.py`

---

## §2 AutoDL 实例选择

| 项目 | 要求 |
|------|------|
| 显存 | **48G 及以上**（单卡） |
| PyTorch | **2.7.0**（选镜像时确认） |
| 磁盘 | 建议 ≥100G（Qwen7B + SAM3 + 数据集） |

数据生成阶段（SAM3 + Qwen 同时加载）和训练阶段均建议在 48G 单卡上运行。

---

## §3 运行环境配置

在项目根目录依次执行：

```bash
python -m pip install --force-reinstall "setuptools<82"
pip install torchvision torchaudio
pip install modelscope

pip install -e .                        # 基础环境
pip install -e ".[notebooks]"           # Notebook 示例
pip install -e ".[train,dev]"           # 训练与开发

pip uninstall -y opencv-python
pip install opencv-python-headless==4.8.0.74
```

验证：

```bash
python -c "import sam3; import torch; print('OK')"
```

若 HuggingFace 下载慢，后续步骤可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com/
```

---

## §4 基座模型下载

### §4.1 SAM3 官方权重（必需）

用于生成 mask 数据、可视化，以及作为微调基座：

```bash
modelscope download --model facebook/sam3 --local_dir pretrained/sam3
```

验证：`ls -lh pretrained/sam3/sam3.pt`

### §4.2 Qwen2.5-VL 权重（二选一，用于自动生成文本标注）

| 版本 | 命令 | 适用场景 |
|------|------|----------|
| **7B（推荐）** | `hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir pretrained/Qwen2.5-VL-7B-Instruct` | 48G 单卡，速度与效果平衡 |
| 32B | `hf download Qwen/Qwen2.5-VL-32B-Instruct --local-dir pretrained/Qwen2.5-VL-32B-Instruct` | 更高质量，需更大显存 |

验证：`ls pretrained/Qwen2.5-VL-7B-Instruct/config.json`

---

## §5 准备训练图片

1. 创建目录并放入图片（jpg/png 均可）：

```bash
mkdir -p datasets/raw_images
# 将你的图片复制到 datasets/raw_images/
# 建议 50–500 张，与目标应用场景相关的图片
```

2. **修改数据生成脚本中的图片列表**

文件：`sam3/infer/generate_t2m_data.py`（约第 319–324 行）

当前硬编码了 3 张示例图，需改为读取 `datasets/raw_images/` 下所有图片。Cursor 应将其改为类似：

```python
import glob
image_list = sorted(
    glob.glob(f"{sam3_root}/datasets/raw_images/*")
)
image_list = [p for p in image_list if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
print(f"Found {len(image_list)} images")
```

或增加 `--image_dir` 命令行参数（更推荐，便于复用）。

验证：`python -c "import glob; print(len(glob.glob('datasets/raw_images/*')))"`  应 > 0

---

## §6 自动生成训练数据

流程：SAM3 分割出 mask → Qwen2.5-VL 为每个 mask 生成文本描述 → 转为 COCO 格式。

### §6.1 统一路径

确保 `sam3/scripts/data_gen/generate_data.sh` 中 `SAVE_ROOT` 为 **`datasets/custom0`**（与训练 yaml 一致）：

```bash
SAVE_ROOT=datasets/custom0
COCO_JSON_PATH=$SAVE_ROOT/annotations.json
```

若脚本中为 `custom1`，请改为 `custom0`。

### §6.2 执行数据生成

在项目根目录：

```bash
export HF_ENDPOINT=https://hf-mirror.com/   # 可选

bash sam3/scripts/data_gen/generate_data.sh
```

或手动分步执行：

```bash
python sam3/infer/generate_t2m_data.py \
    --sam_path pretrained/sam3/sam3.pt \
    --qwen_path pretrained/Qwen2.5-VL-7B-Instruct \
    --save_root datasets/custom0 \
    --batchsize 4 \
    --vlm_batchsize 4 \
    --num_pts 512 \
    --score_thresh 0.75 \
    --iou_thresh 0.1

python sam3/infer/convert_to_cocoapi.py \
    --data_path datasets/custom0 \
    --save_json_path datasets/custom0/annotations.json
```

显存不足时可减小 `--batchsize` 和 `--vlm_batchsize`（如改为 1 或 2）。

验证：

```bash
ls datasets/custom0/annotations.json
python -c "import json; d=json.load(open('datasets/custom0/annotations.json')); print('images', len(d['images']), 'annotations', len(d['annotations']))"
```

---

## §7 修改训练配置

文件：`sam3/train/configs/mydata/text_only_train.yaml`

Cursor 在启动训练前必须确认以下项：

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| `run_name` | 自定义，如 `exp-001` | 实验名称 |
| `paths.annotation_path` | `datasets/custom0/annotations.json` | 与 §6 输出一致 |
| `paths.model_path` | `pretrained/sam3/sam3.pt` | SAM3 基座 |
| `data_args.num_images` | 实际图片数量 | 不超过 raw_images 中的图片数 |
| `launcher.gpus_per_node` | **`1`** | 单卡 48G（默认是 2，需改） |
| `trainer.skip_saving_ckpts` | **`false`** | 默认 `true` 会不保存 checkpoint，必须改 |

单卡训练时 `train_batch_size: 1` 和 `resolution: 1008` 一般可直接用于 48G 显存；OOM 时可减小 `resolution` 或增大 `gradient_accumulation_steps`。

---

## §8 启动微调训练

在项目根目录：

```bash
python sam3/train/train.py \
    -c configs/mydata/text_only_train.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

说明：
- `--use-cluster 0`：本地运行，不提交 SLURM 集群
- `--num-gpus 1`：单卡训练

训练产出：
- Checkpoint：`workdir/<run_name>/checkpoints/`
- 日志：`workdir/<run_name>/logs/`

验证：`ls workdir/*/checkpoints/`

---

## §9 常见问题

| 现象 | 处理 |
|------|------|
| `import sam3` 失败 | 回到 §3 重新 `pip install -e ".[train,dev]"` |
| 找不到 `sam3.pt` | 回到 §4.1 下载权重 |
| 数据生成 CUDA OOM | 减小 `--batchsize`、`--vlm_batchsize`；或换 7B 而非 32B |
| 训练 OOM | yaml 中减小 `resolution`，或 `gradient_accumulation_steps: 2` |
| `annotations.json` 找不到 | 检查 §6 是否完成；`paths.annotation_path` 是否与 `SAVE_ROOT` 一致 |
| 训练结束无 checkpoint | 检查 `trainer.skip_saving_ckpts` 是否为 `false` |
| HuggingFace 下载慢 | `export HF_ENDPOINT=https://hf-mirror.com/` |
| opencv 冲突 | `pip uninstall -y opencv-python && pip install opencv-python-headless==4.8.0.74` |

---

## §10 项目结构速查

```
.
├── instruction.md                          # 本文件（项目操作流程）
├── EXPERIMENT_TRACE.md                     # 通用留痕规范（可复制到新项目）
├── records/                                # §11 实验留痕
│   ├── experiments.md
│   ├── data/CHANGELOG.md
│   └── results/README.md
├── configs/snapshots/                      # 实验配置快照
├── scripts/record_snapshot.sh              # 状态采集脚本
├── .cursor/rules/experiment-logging.mdc    # Cursor 留痕规则
├── pyproject.toml
├── examples/images/                        # 内置 3 张 demo 图（测试用）
├── datasets/
│   ├── raw_images/                         # §5 用户放入的原始图片
│   └── custom0/
│       └── annotations.json                # §6 生成的 COCO 标注
├── pretrained/
│   ├── sam3/sam3.pt                        # §4.1
│   └── Qwen2.5-VL-7B-Instruct/             # §4.2
├── sam3/
│   ├── infer/
│   │   ├── generate_t2m_data.py            # 数据生成主脚本（§5 需改图片路径）
│   │   └── convert_to_cocoapi.py           # 转 COCO 格式
│   ├── scripts/data_gen/generate_data.sh   # 一键数据生成（§6）
│   └── train/
│       ├── train.py                        # 训练入口（§8）
│       └── configs/mydata/text_only_train.yaml  # 训练配置（§7）
└── workdir/                                # §8 训练产出
```

---

## 给 Cursor 的任务优先级

当用户说「帮我跑起来 / 继续项目 / 看看还要做什么」时，按以下顺序处理：

1. **运行 §「当前进度检查」**，汇报已完成和未完成步骤
2. **从第一个未完成步骤继续**，不要跳步
3. **§5**：若 `generate_t2m_data.py` 仍用硬编码示例图，优先改为 `--image_dir datasets/raw_images` 或 glob 读取
4. **§6**：统一 `generate_data.sh` 与 yaml 的数据路径为 `datasets/custom0`
5. **§7**：改 yaml 中 `gpus_per_node: 1`、`skip_saving_ckpts: false`、`num_images` 为实际值
6. **§8**：启动训练并监控是否 OOM
7. 改动代码时**不要重构无关模块**；每步完成后给出验证命令输出

---

## §11 实验留痕（工作留痕 / 求职证明）

留痕规范见 **[`EXPERIMENT_TRACE.md`](EXPERIMENT_TRACE.md)**（通用，可复制到其他项目）。  
本项目已初始化：

| 文件 | 用途 |
|------|------|
| `records/experiments.md` | 实验主日志 |
| `records/data/CHANGELOG.md` | 数据集版本 |
| `records/results/README.md` | 对比图索引 |
| `configs/snapshots/` | 每次实验 yaml 快照 |
| `scripts/record_snapshot.sh` | 一键采集状态 |
| `.cursor/rules/experiment-logging.mdc` | Cursor 自动按规范写记录 |

### 实验过程中你怎么说

```
记录数据变更：[简述] + [粘贴 record_snapshot.sh 输出或统计]
记录实验 exp-001：[配置 / 结果 / 现象]
同步记录并 commit（不要 push）
```

### 推荐节奏

| 时机 | 动作 |
|------|------|
| 数据生成前后 | 「记录数据变更」→ 更新 CHANGELOG |
| 开训前 | `bash scripts/record_snapshot.sh exp-001` → 「开始 exp-001」 |
| 训练结束 | 贴日志末尾 → 「exp-001 完成」 |
| 出对比图后 | 更新 `records/results/` → 让我更新索引 |
| 里程碑 | 「同步记录并 commit」 |

### 快照脚本

```bash
bash scripts/record_snapshot.sh          # 只看当前状态
bash scripts/record_snapshot.sh exp-001  # 同时保存 yaml 快照
```
