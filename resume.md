# SAM3 语言指令分割微调 — 项目总结（简历用）

> 仓库：[github.com/acrophob1a/sam3-finetune](https://github.com/acrophob1a/sam3-finetune)  
> 留痕：`records/experiments.md` · 对比图：`records/results/exp-001/`

---

## 一句话描述

基于 Meta **SAM3**，在 TRUDI 港口场景数据集上搭建 **「自动标注 → 语言指令分割微调 → 基座对比评估」** 完整 pipeline，实现自然语言驱动的集装箱/卡车等目标分割。

---

## 项目背景与目标

- **场景**：内陆港口、集装箱堆场、货运卡车等工业视觉场景（TRUDI 数据集）
- **任务**：给定图像 + 英文文本描述（如 *"A large blue semi-trailer truck"*），输出对应实例分割 mask
- **目标**：验证 SAM3 在垂直领域数据上的可微调性，形成可复现、可留痕的实验基线（exp-001）

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 基座模型 | SAM3（ViT + 文本编码 + MaskFormer 分割头） |
| 自动标注 | SAM3 点提示分割 + **Qwen2.5-VL-7B** 文本生成 |
| 训练 | PyTorch 2.8 · CUDA · 单卡 A800 80GB · Hydra 配置 |
| 数据 | TRUDI COCO 格式 · 210 train / 70 test |
| 工程 | 数据划分脚本 · 流式 VLM 批处理 · 单卡 DDP 绕过 · 对比推理脚本 |

---

## 个人工作与贡献

1. **数据工程**
   - 解压并划分 TRUDI 官方 train/test（210 / 70 张）
   - 改造 `generate_t2m_data.py`：大图缩放（长边 1536）、mask 磁盘缓存、SAM/Qwen 分阶段释放显存，解决 OOM
   - 自动生成 **4279 条** `(图像, noun_phrase, mask)` 训练标注

2. **模型微调**
   - 配置并运行 10 epoch 文本指令分割微调（resolution 1008，batch size 1）
   - 定位并修复单卡训练 DDP unused-parameter 崩溃（`world_size=1` 跳过 DDP 包装）
   - 产出 9.4G checkpoint：`workdir/exp-001/checkpoints/checkpoint.pt`

3. **评估与可视化**
   - 编写 `scripts/compare_base_finetuned.py`，在 70 张测试集上对比基座 vs 微调
   - 生成 8 组 side-by-side 对比图及 `comparison_summary.json`

4. **实验留痕**
   - 建立 `records/` 实验日志、数据 CHANGELOG、配置快照体系（`EXPERIMENT_TRACE.md`）

---

## 关键结果（exp-001）

| 指标 | 数值 |
|------|------|
| 训练数据 | 210 图 · 4279 标注 · ~20 mask/图 |
| 标注生成耗时 | ~100 min（SAM 17min + Qwen 83min） |
| 微调耗时 | 10 epoch · ~29 min · 2100 steps |
| 测试对比 | 8 组文本 prompt · 70 张 test 集 |

**定性观察**（测试集 8 组样例）：

- 基座模型在港口专用表述（如 *semi-trailer truck*、*stack of shipping containers*）上 **6/8 样例零检出**
- 微调后在相同 prompt 下 **6/8 样例成功检出**（置信度约 0.57–0.70）
- 说明领域微调有效提升了模型对港口场景语言指令的响应能力

详细数值见 `records/results/exp-001/comparison_summary.json`。

---

## Pipeline 架构

```
TRUDI ground.zip
    ↓ setup_trudi_phase1.sh
raw_images_train (210) + raw_images_test (70)
    ↓ SAM3 点分割 + Qwen2.5-VL 描述
custom0_exp001/annotations.json (4279条)
    ↓ text_only_train.yaml · 10 epoch
workdir/exp-001/checkpoints/checkpoint.pt
    ↓ compare_base_finetuned.py
records/results/exp-001/*_compare.png
```

---

## 可写入简历的 bullet points（中英文）

**中文**

- 基于 SAM3 搭建港口场景语言指令分割微调 pipeline，完成 TRUDI 210 张训练数据自动标注（SAM3 + Qwen2.5-VL）及 10 epoch 微调
- 优化大图标注流程（分辨率缩放、流式 VLM、显存分阶段释放），将标注生成从 OOM 失败修复为稳定产出 4279 条三元组
- 修复单卡 PyTorch DDP unused-parameter 训练崩溃，编写基座/微调对比推理脚本，验证领域微调对专用文本 prompt 的检出提升

**English**

- Built an end-to-end language-guided segmentation fine-tuning pipeline on SAM3 using the TRUDI port/container dataset (210 images, 4.3K auto-labeled masks via SAM3 + Qwen2.5-VL)
- Optimized large-image annotation workflow (resize, streaming VLM batches, staged GPU memory release) to resolve OOM failures
- Fixed single-GPU DDP training crash and delivered base vs. fine-tuned inference comparisons showing improved text-prompt detection on domain-specific queries

---

## 复现命令（摘要）

```bash
# 数据划分
SAMPLE_TRAIN=0 bash scripts/setup_trudi_phase1.sh ground.zip

# 自动标注
bash sam3/scripts/data_gen/generate_data.sh

# 微调
python sam3/train/train.py -c configs/mydata/text_only_train.yaml --use-cluster 0 --num-gpus 1

# 基座 vs 微调对比
python scripts/compare_base_finetuned.py
```

完整步骤见 `instruction.md`，实验记录见 `records/experiments.md`。

---

## 后续可扩展方向

- 引入 val 集定量指标（mAP / IoU）
- 增加 epoch 或 LoRA 等高效微调对比（exp-002）
- 中文指令 / 多 prompt 联合训练
