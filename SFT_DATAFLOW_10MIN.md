# SAM3 微调数据流 — 10 分钟版

> 读完本文应能：**口述全流程**、**解释 `"object"` 从哪来**、**说出 3 个关键 shape**。  
> 深入：[SFT_DATAFLOW_LEARN.md](SFT_DATAFLOW_LEARN.md) · 规格：[SFT_DATAFLOW.md](SFT_DATAFLOW.md) · 实测：`python scripts/sft_dataflow_trace.py --run --forward`

---

## 1. 项目在干什么（30 秒）

**输入**：港口场景照片 + 一句英文  
**输出**：每个目标的 mask / 框  
**本项目做的事**：用 TRUDI 210 张图自动造标注 → 微调 SAM3 10 epoch → 对比基座 vs 微调

---

## 2. 全流程一张图（1 分钟）

```
ground.zip
   │  setup_trudi_phase1.sh
   ▼
raw_images_train/ (210)          raw_images_test/ (70，只推理)
   │
   │  generate_t2m_data.py  ← SAM3 点切 + Qwen 写 noun_phrase
   ▼
custom0_exp001/annotations.json  (4279 条)
   │
   │  train.py + text_only_train.yaml  ← 本文重点：这条训练流
   ▼
workdir/exp-001/checkpoints/checkpoint.pt
   │
   │  compare_base_finetuned.py
   ▼
records/results/exp-001/*_compare.png
```

**两个世界别混**：上面 JSON 里的 `noun_phrase` 是 Qwen 写的长句；**训练时不用它**，只用类别名 `"object"`。

---

## 3. 训练流：两阶段（1 分钟）

| 阶段 | 在哪跑 | 干什么 |
|------|--------|--------|
| **A. 数据预处理** | CPU | 读 JSON、读图、transform、拼 batch |
| **B. 模型 + loss** | GPU | 前向预测 → 和 GT 匹配 → 算 loss → 反传 |

对应代码入口：

- A：`Sam3ImageDataset.__getitem__` → `collate_fn_api`
- B：`Trainer._step` → `Sam3Image.forward` → `Sam3LossWrapper`

---

## 4. 一个样本走完全程：0000.jpg（5 分钟）

以下 shape 来自 `python scripts/sft_dataflow_trace.py --run --forward`（样本 idx=0）。

### Step 0 — 磁盘上有什么

```
datasets/raw_images_train/0000.jpg     # 原图 4624×2084
datasets/custom0_exp001/annotations.json
  └─ 该图 18 条标注，每条有 bbox、mask、noun_phrase
  └─ categories: [{ "id": 1, "name": "object" }]   ← 训练只用这个 name
```

### Step A1–A2 — JSON → 一条训练 query

```
18 个 GT 实例  +  category "object"
        │
        ▼
1 条 find_query:
  query_text = "object"
  object_ids_output = [0,1,2,...,17]    # 要同时 ground 这 18 个实例
```

代码：[coco_json_loaders.py L245](sam3/train/data/coco_json_loaders.py#L245)

### Step A3 — 读图 + transform

```
0000.jpg (4624×2084)
        │  DecodeRle → Resize/Pad → ToTensor → Normalize
        ▼
img.data          shape (3, 1008, 1008)     # 进模型的图像
objects[i].bbox   shape (1, 4) cxcywh       # 归一化框，共 N 个（约 17–18）
objects[i].segment shape (1008, 1008)       # 二值 mask
find_queries[0].query_text = "object"
```

代码：[sam3_image_dataset.py L488](sam3/train/data/sam3_image_dataset.py#L488)

### Step A4 — Collator 组 batch（B=1）

```
Datapoint (1 张图)
        │
        ▼
BatchedDatapoint:
  img_batch           (1, 3, 1008, 1008)
  find_text_batch     ["object"]
  boxes_padded        (1, N, 4)           # N = 该图 GT 数
  input_boxes         (0, 1, 4)            # 空 = 纯文本，无框 prompt
```

代码：[collator.py L137](sam3/train/data/collator.py#L137)

### Step B1–B2 — 模型前向（GPU）

```
img_batch (1,3,1008,1008)  +  text "object"
        │
        ├─ Vision backbone  → 图像特征
        ├─ Text backbone    → 文本特征
        └─ Grounding decoder → 200 个 object queries
                │
                ▼
  pred_logits   (1, 200, 1)      # 每个 query：有没有物体
  pred_boxes    (1, 200, 4)      # 每个 query：框在哪
  pred_masks    (1, 200, 288, 288)  # 低分辨率 mask（有输出）
```

代码：[sam3_image.py L527](sam3/model/sam3_image.py#L527)

**直觉**：200 个「检测槽位」在文本 `"object"` 条件下各自认领或不认领一个物体。

### Step B3–B5 — 匹配 + loss → 标量

```
200 个预测  ←── Hungarian 匹配 ──→  N≈18 个 GT
        │
        ▼
  loss_bbox   框 L1
  loss_giou   框 GIoU
  loss_ce     有没有物体（focal/BCE 风格）
  presence_loss
        │
        ▼
  core_loss   标量  →  backward
```

exp-001 **不算 mask loss**（yaml 里 `loss_fn_semantic_seg: null`）。

代码：[trainer.py L492](sam3/train/trainer.py#L492) · [sam3_loss.py L161](sam3/train/loss/sam3_loss.py#L161)

---

## 5. 数据流 shape 速查（1 分钟）

```
annotations.json
    →  Datapoint:  img (3,1008,1008),  N 个 mask
    →  Batch:      img_batch (1,3,1008,1008),  boxes_padded (1,N,4)
    →  Forward:    pred_logits (1,200,1),  pred_masks (1,200,288,288)
    →  Loss:       core_loss (标量)
```

记三个就够：**`(3,1008,1008)`** · **`(1,200,1)`** · **`(1,N,4)`**

---

## 6. 三个易错点（1 分钟）

| 误区 | 真相 |
|------|------|
| 训练用 Qwen 写的 `noun_phrase` | 训练只用 `"object"`；noun_phrase 只在 JSON 里存着 |
| 这是 LLM 预测下一个 token | 这是 DETR 式 grounding：200 queries + 框/mask 预测 |
| 有 mask 头就一定学 mask | exp-001 没配 mask loss，主要靠框监督 |

---

## 7. 10 分钟自测（30 秒）

关文档后口头回答：

1. 从 `0000.jpg` 到 `core_loss` 经过哪四个「形状节点」？  
2. `find_text_batch` 里是什么？为什么不是 truck 那句英文？  
3. A 阶段和 B 阶段分别在 CPU 还是 GPU？

**跑一条命令验证**：

```bash
cd /root/autodl-tmp/sam3
python scripts/sft_dataflow_trace.py --run --forward
```

对照输出里的 `img_batch_shape`、`find_text_batch`、`pred_logits_shape`、`core_loss`。

---

## 8. 接下来读什么

| 目标 | 文档 |
|------|------|
| 再学 1–2 小时 | [SFT_DATAFLOW_LEARN.md](SFT_DATAFLOW_LEARN.md) Day 1–4 |
| 查行号 / 风险 | [SFT_DATAFLOW.md](SFT_DATAFLOW.md) |
| 项目全貌 / 简历 | [resume.md](resume.md) |
| 实验结果 | [records/experiments.md](records/experiments.md) |
