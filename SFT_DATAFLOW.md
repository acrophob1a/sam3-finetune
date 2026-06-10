# SAM3 微调数据流 — 规格版（SFT_DATAFLOW）

> **范围**：本项目 **exp-001 实际跑通** 的 SAM3 语言 grounding 微调（非 LLM SFT、非 Pretrain、非 DPO）。  
> **实测命令**：`python scripts/sft_dataflow_trace.py --run --forward`（2026-06-06，A800）  
> **学习版**：[SFT_DATAFLOW_LEARN.md](SFT_DATAFLOW_LEARN.md) · **可运行版**：[scripts/sft_dataflow_trace.py](scripts/sft_dataflow_trace.py)

---

## 范围声明

| 阶段 | 本项目 | 说明 |
|------|--------|------|
| 自动标注（SAM3+Qwen） | ✅ 已实现 | 见 `sam3/infer/generate_t2m_data.py`，**不在本文 B 阶段训练流内** |
| **微调训练（exp-001）** | ✅ 本文主体 | 210 图 · 10 epoch · `text_only_train.yaml` |
| Pretrain | ❌ 未覆盖 | 使用 Meta 预训练 `pretrained/sam3/sam3.pt` |
| LLM SFT / DPO | ❌ 未覆盖 | 无 Causal LM loss、无 preference 数据 |

**重要**：训练时 `query_text` 来自 COCO **`categories[].name`（恒为 `"object"`）**，**不是** 标注里的 `noun_phrase`。推理对比脚本用的是自然语言 noun phrase，与训练文本分布不同（见 [records/results/exp-001/comparison_summary.json](records/results/exp-001/comparison_summary.json)）。

---

## 符号表

| 符号 | 含义 |
|------|------|
| B | batch_size（exp-001 = 1） |
| C | 图像通道数（3） |
| H, W | 输入图像高宽（pad 后 1008×1008） |
| H_m, W_m | 分割 mask 低分辨率（实测 288×288） |
| Q | Decoder object queries（O2O，200） |
| Q_o2m | DAC 一对多 queries（200） |
| N | 单张图 GT 实例数（样本 0000 实测 17–18，随 transform 过滤波动） |
| N_max | batch 内 pad 后最大 GT 数 |
| d_model | Transformer 隐藏维（256） |
| d_vit | ViT 特征维（1024，经 neck 到 256） |

---

## 最小测试样本

```python
raw_sample = {
    "image_file": "datasets/raw_images_train/0000.jpg",
    "width": 4624,
    "height": 2084,
    "category_name": "object",  # 训练 query_text
    "noun_phrase_in_json": "A large blue semi-trailer truck with \"LKW Walter\" branding.",
    # noun_phrase 写入 JSON 但 COCO_FROM_JSON 不读取
}
```

**shape 数据来源**：`python scripts/sft_dataflow_trace.py --run --forward`

| 阶段 | 关键 shape（样本 0000，idx=0） |
|------|--------------------------------|
| A3 单样本 | `img.data (3, 1008, 1008)`；GT masks `(1008, 1008)` × N |
| A4 batch | `img_batch (1, 3, 1008, 1008)`；`boxes_padded (1, N, 4)` |
| B forward | `pred_logits (1, 200, 1)`；`pred_masks (1, 200, 288, 288)` |
| B loss | `core_loss` 标量（实测 ~414.5，未训练权重随机匹配） |

---

## 调用链总览

| 组件 | 创建 | 调用 | 输入 → 输出 |
|------|------|------|-------------|
| `Sam3ImageDataset` | Hydra `trainer.data.train.dataset` | `ds[i]` | `int` → `Datapoint` |
| `collate_fn_api` | `train_args.collate_fn` | DataLoader | `List[Datapoint]` → `{all: BatchedDatapoint}` |
| `Trainer._step` | [trainer.py](sam3/train/trainer.py#L492) | `train_epoch` | batch → `core_loss` |
| `Sam3Image` | `build_sam3_image_model` | `model(batch)` | `BatchedDatapoint` → `SAM3Output` |
| `Sam3LossWrapper` | `trainer.loss.all` | `loss(find_stages, targets)` | → `dict` + `core_loss` |

---

# 阶段 A：数据预处理（CPU）

## A1 — 加载 COCO JSON

**定位**：[load_coco_and_group_by_image](sam3/train/data/coco_json_loaders.py#L37)

| | |
|--|--|
| **输入** | `datasets/custom0_exp001/annotations.json`（210 images, 4279 anns） |
| **输出** | `_raw_data: List[{image, annotations}]` 按 image 分组 |
| **目的** | 将 COCO 扁平标注变为「每图一条」训练索引 |

---

## A2 — 构造 query + GT（COCO_FROM_JSON）

**定位**：[COCO_FROM_JSON.loadQueriesAndAnnotationsFromDatapoint](sam3/train/data/coco_json_loaders.py#L153)

| | |
|--|--|
| **输入** | datapoint index `idx`（0 → 第 0 张训练图） |
| **输出** | `queries`: 长度 1；`query_text="object"`；`object_ids_output`: 长度 N（18） |
| **输出** | `annotations`: N 条；`bbox` 归一化 xywh `(4,)`；`segmentation` RLE |
| **目的** | 按 **category** 构造一条 find query（本项目仅 1 类） |

关键代码：

```245:249:sam3/train/data/coco_json_loaders.py
            query["query_text"] = (
                self._cat_idx_to_text[cat_id]
                if self.prompts is None
                else self.prompts[cat_id]
            )
```

---

## A3 — Dataset.__getitem__ + transforms

**定位**：[Sam3ImageDataset.__orig_getitem__](sam3/train/data/sam3_image_dataset.py#L488)

| | |
|--|--|
| **输入** | `idx: int` |
| **输出** | `Datapoint`：`images[0].data` → `(3, 1008, 1008)` float |
| **输出** | `find_queries[0].query_text` → `"object"` |
| **输出** | `images[0].objects[i].bbox` → `(1, 4)` cxcywh（NormalizeAPI 后） |
| **输出** | `objects[i].segment` → `(1008, 1008)` bool/uint8 |
| **目的** | 读图、解码 RLE、随机 resize/pad 到 1008、过滤空目标 |

Transform 链配置：[text_only_train.yaml](sam3/train/configs/mydata/text_only_train.yaml#L21)  
`FilterEmptyTargets` 可能导致 N 从 18→17。

---

## A4 — Collator 组 batch

**定位**：[collate_fn_api](sam3/train/data/collator.py#L137)

| 字段 | shape | 含义 |
|------|-------|------|
| `img_batch` | `(B, 3, 1008, 1008)` | 堆叠图像 |
| `find_text_batch` | `List[str]`，len=1 | 去重文本 `["object"]` |
| `find_inputs[0].input_boxes` | `(0, 1, 4)` | 无 box prompt（纯文本 grounding） |
| `find_inputs[0].input_points` | `(1, 0, 257)` | 无点 prompt |
| `find_targets[0].num_boxes` | `(B,)` → `[N]` | 每图 GT 数 |
| `find_targets[0].boxes_padded` | `(B, N_max, 4)` | pad 后 GT cxcywh |
| `find_targets[0].object_ids_padded` | `(B, N_max)` | 无效位 -1 |
| `find_targets[0].segments` | list len=N | 二值 mask |

堆叠图像：

```346:346:sam3/train/data/collator.py
    image_batch = torch.stack(img_batch)
```

**验证**：`python scripts/sft_dataflow_trace.py --run --step A4`

---

## A5 — Trainer 搬 batch 到 GPU

**定位**：[Trainer._step](sam3/train/trainer.py#L492)

```498:500:sam3/train/trainer.py
        key, batch = batch.popitem()
        batch = copy_data_to_device(batch, self.device, non_blocking=True)
        find_stages = model(batch)
```

| | |
|--|--|
| **输入** | `{"all": BatchedDatapoint}` on CPU |
| **输出** | 同上结构 on CUDA |
| **目的** | 进入模型前向 |

---

## A6 — 阶段 A 流程图（单样本 + batch）

```
annotations.json
  └─ image 0000.jpg (4624×2084) + 18 anns
       └─ A2: query_text="object", N_gt=18
            └─ A3: img (3,1008,1008), masks N×(1008,1008)
                 └─ A4: img_batch (1,3,1008,1008)
                      boxes_padded (1,N,4)
                      find_text_batch ["object"]
```

---

# 阶段 B：模型前向 + Loss（GPU）

## B1 — Vision + Text Backbone

**定位**：[Sam3Image.forward](sam3/model/sam3_image.py#L527)

| | |
|--|--|
| **输入** | `img_batch (B,3,1008,1008)`；`find_text_batch ["object"]` |
| **输出** | `backbone_out`：图像特征金字塔 + 文本 token 嵌入 |
| **目的** | 双塔编码，供 grounding decoder 使用 |

```529:535:sam3/model/sam3_image.py
        backbone_out = {"img_batch_all_stages": input.img_batch}
        backbone_out.update(self.backbone.forward_image(input.img_batch))
        ...
        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)
```

**融合点**：文本与几何 prompt 在 `_encode_prompt` 中融合（无图像 token 替换 LLM 式 mRoPE；SAM3 用 cross-attention grounding）。

---

## B2 — Grounding Decoder + 分割头

**定位**：[forward_grounding](sam3/model/sam3_image.py#L439)

| | |
|--|--|
| **输入** | `backbone_out`；`find_input`；空/几何 prompt |
| **输出** | `pred_logits (B, Q, 1)`；`pred_boxes (B, Q, 4)` |
| **输出** | `pred_logits_o2m (B, Q, 1)`（DAC） |
| **输出** | `pred_masks (B, Q, H_m, W_m)`（实测 288×288） |
| **目的** | 200 O2O + 200 O2M queries 预测框与 mask |

Q=200 来自 [model_builder](sam3/model_builder.py) `num_queries=200`，DAC 使训练时 O2M 分支有效。

---

## B3 — Hungarian 匹配

**定位**：[Sam3Image._compute_matching](sam3/model/sam3_image.py#L575) · [BinaryHungarianMatcherV2](sam3/train/matcher.py)

| | |
|--|--|
| **输入** | `pred_*` + `find_targets`（pad 后 GT） |
| **输出** | `indices`：预测 query ↔ GT 的一一匹配 |
| **目的** | DETR 式分配，供监督 loss 使用 |

---

## B4 — Loss 计算

**定位**：[Sam3LossWrapper.forward](sam3/train/loss/sam3_loss.py#L161) · [compute_loss](sam3/train/loss/sam3_loss.py#L83)

exp-001 启用的 loss（[yaml L70-86](sam3/train/configs/mydata/text_only_train.yaml#L70)）：

| Loss 类 | 键 | 权重 |
|---------|-----|------|
| [Boxes](sam3/train/loss/loss_fns.py#L523) | `loss_bbox`, `loss_giou` | 5.0, 2.0 |
| [IABCEMdetr](sam3/train/loss/loss_fns.py#L266) | `loss_ce`, `presence_loss` | 20.0, 20.0 |

**未启用**：`loss_fn_semantic_seg: null` — **mask 预测头有输出但无 mask loss**。

`IABCEMdetr.get_loss` 中 `src_logits (B, Q)`，`pad_n_queries=200` 归一化（[L391-506](sam3/train/loss/loss_fns.py#L391)）。

---

## B5 — 汇总为 core_loss

**定位**：[Sam3LossWrapper.forward](sam3/train/loss/sam3_loss.py#L192)

```python
cur_losses[CORE_LOSS_KEY] *= bs ** 0.5   # scale_by_find_batch_size=True
```

| | |
|--|--|
| **输入** | 各子 loss 标量/张量 |
| **输出** | `core_loss` 标量；Trainer backward 用此值 |
| **目的** | 加权求和 + aux/O2M + batch scale |

Trainer 提取 core_loss：[trainer.py L1112-1118](sam3/train/trainer.py#L1112)

---

## B6 — 阶段 B 流程图

```
BatchedDatapoint
  ├─ forward_image → visual features
  ├─ forward_text("object") → text features
  └─ forward_grounding
        ├─ pred_logits (1,200,1)
        ├─ pred_boxes  (1,200,4)
        ├─ pred_masks  (1,200,288,288)
        └─ Hungarian match ↔ N_gt=18
              └─ Boxes + IABCEMdetr
                    └─ core_loss scalar
```

**验证**：`python scripts/sft_dataflow_trace.py --run --forward`

---

## Shape 不匹配风险表

| # | 代码位置 | 风险 | 后果 |
|---|----------|------|------|
| 1 | [collator.py L344-345](sam3/train/data/collator.py#L344) | batch 内图像尺寸不一致 | assert 失败 |
| 2 | [coco_json_loaders.py L245](sam3/train/data/coco_json_loaders.py#L245) | 误以为用 `noun_phrase` 训练 | 实际只训练 `"object"` |
| 3 | [trainer.py L302](sam3/train/trainer.py#L302) | 单卡仍包 DDP | unused param RuntimeError（已修复：skip DDP） |
| 4 | [sam3_loss.py L75](sam3/train/loss/sam3_loss.py#L75) | 未 init process group | `all_reduce` 失败（trace 脚本需 `_ensure_dist`） |
| 5 | [loss 配置](sam3/train/configs/mydata/text_only_train.yaml#L88) | 有 mask 头无 mask loss | 分割质量仅靠 box/GIoU 间接监督 |

---

## 与 exp-001 实验对齐

| 项 | 值 |
|----|-----|
| run_name | exp-001 |
| steps/epoch | 210 |
| epochs | 10 |
| 训练日志 | `records/logs/train_exp001.log` |
| 推理对比 | `records/results/exp-001/comparison_summary.json` |
| 基座 vs 微调 | 8 组 test 样例；微调 6/8 由零检出变为有检出 |

---

## 自检题

1. 训练时 `find_text_batch` 的内容是什么？为什么 JSON 里有 `noun_phrase` 却不用？  
2. `pred_masks` 的 shape 是什么？exp-001 有没有对 mask 算 loss？  
3. 单卡训练为什么要跳过 DDP？

**验证命令**

```bash
python scripts/sft_dataflow_trace.py --doc
python scripts/sft_dataflow_trace.py --run --step A4
python scripts/sft_dataflow_trace.py --run --forward
```
