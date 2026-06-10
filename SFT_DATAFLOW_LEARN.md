# SAM3 微调数据流 — 学习版（SFT_DATAFLOW_LEARN）

> 读者：有 PyTorch 基础、第一次做视觉-语言 grounding 微调。  
> 规格版：[SFT_DATAFLOW.md](SFT_DATAFLOW.md) · 实测脚本：[scripts/sft_dataflow_trace.py](scripts/sft_dataflow_trace.py)

---

## 先建立直觉：本项目在训练什么？

不是 ChatGPT 式的「下一个 token 预测」，而是 **DETR 式 grounding**：

1. 输入一张图 + 一句文本（exp-001 里这句几乎总是 `"object"`）
2. 模型吐出 **200 个 object queries**，每个 query 预测：有没有目标、框在哪、（可选）mask 长什么样
3. 用 **Hungarian 匹配** 把 queries 和 GT 实例一一配对
4. 对配对结果算 **框 L1 + GIoU + 分类/presence loss**

数据流分两段：**A 阶段在 CPU 把 JSON 变成 batch**，**B 阶段在 GPU 前向 + 反传**。

---

## Day 1：从磁盘到 `Datapoint`（A1–A3）

### 你要搞懂的三个问题

- COCO JSON 长什么样？
- 谁把 `noun_phrase` 变成训练用的 `query_text`？
- 图像 tensor 为什么是 `(3, 1008, 1008)`？

### A1 — 打开标注文件

运行：

```bash
python scripts/sft_dataflow_trace.py --run --step A1
```

你会看到 210 张图、4279 条标注。注意 `categories` 只有一类：`name: "object"`。

**自检**：`noun_phrase` 字段存了什么？训练会不会用到它？  
→ 存了 Qwen 生成的描述；**训练不用**，见 [COCO_FROM_JSON L245](sam3/train/data/coco_json_loaders.py#L245)。

### A2 — 一条 query 代表什么？

对每张图，loader 生成 **1 条 find query**：

- `query_text = "object"`
- `object_ids_output` = 该图所有 GT 实例 id 列表（约 18 个）

这意味着：**一次 forward 要同时 ground 这张图里的所有实例**，不是「一句 noun phrase 对一个 mask」。

### A3 — Transform 做了什么？

顺序（[yaml L21-58](sam3/train/configs/mydata/text_only_train.yaml#L21)）：

1. 解码 RLE →  bitmap mask  
2. RandomResize + Pad → **1008×1008**  
3. ToTensor + Normalize（mean/std 0.5）  
4. bbox 从 xyxy 转 **cxcywh**  
5. `FilterEmptyTargets` 可能删掉极小实例 → **N 会波动**

实测：

```bash
python scripts/sft_dataflow_trace.py --run --step A3
```

### Day 1 自检

1. 样本 0000 的 `query_text` 是什么？  
2. `img.data` 的 shape？每一维含义？  
3. 为什么原图 4624×2084，tensor 却是 1008×1008？

---

## Day 2：Collator 与 batch 字典（A4–A6）

### 核心概念：`Datapoint` → `BatchedDatapoint`

- **Dataset** 返回「一条样本」的结构体  
- **Collator** 把 B 条样本拼成模型吃的 dict

### A4 — 重点字段

| 字段 | 直觉 |
|------|------|
| `img_batch` | 真正的图像 batch |
| `find_text_batch` | 去重后的文本列表（B=1 时就是 `["object"]`） |
| `find_targets[0].boxes_padded` | GT 框，不足 pad |
| `find_inputs[0].input_boxes` | 用户提供的 box prompt（本项目为空） |

```bash
python scripts/sft_dataflow_trace.py --run --step A4
```

### A5 — 谁调用 Collator？

[TorchDataset.get_loader](sam3/train/data/torch_dataset.py#L37) 创建 DataLoader，`collate_fn=collate_fn_api`。

Trainer 每个 step：

```492:505:sam3/train/trainer.py
    def _step(
        self,
        batch: BatchedDatapoint,
        model: nn.Module,
        phase: str,
    ):
        key, batch = batch.popitem()
        batch = copy_data_to_device(batch, self.device, non_blocking=True)
        find_stages = model(batch)
        ...
        loss = self._find_loss(key)(find_stages, find_targets)
```

### Day 2 串讲 checklist

- [ ] 能画出：`json → Datapoint → BatchedDatapoint → GPU`  
- [ ] 能解释 `dict_key: all` 从哪来（[yaml collate_fn](sam3/train/configs/mydata/text_only_train.yaml#L168)）  
- [ ] 知道 `input_points (1,0,257)` 表示「没有点 prompt」

---

## Day 3：模型前向（B1–B3）

### 和 LLM SFT 的区别（避免混淆）

| | LLM SFT | 本项目 SAM3 |
|--|---------|-------------|
| 文本用法 | token 序列进 LM | 文本编码 → 与视觉 cross-attn |
| 输出 | logits `(B,T,V)` | queries `(B,200,·)` |
| 监督 | next-token CE | Hungarian + box/GIoU |

### B1 — 双塔

[Sam3Image.forward](sam3/model/sam3_image.py#L527)：

- `forward_image(img_batch)` → 多尺度视觉特征  
- `forward_text(["object"])` → 文本条件向量  

**没有** Qwen 式 `image_pad` token 填进 LM 序列；融合发生在 grounding transformer 内。

### B2 — 200 个 queries 是什么？

可以把它想成「200 个可学习的检测槽位」，每个槽问：  
「在这个文本条件下，我要不要认领一个物体？框在哪？」

DAC 训练时再加 200 个 O2M queries，辅助一对多匹配（见 `pred_logits_o2m`）。

实测 shape：

```bash
python scripts/sft_dataflow_trace.py --run --forward
# pred_logits (1, 200, 1)
# pred_masks  (1, 200, 288, 288)
```

### B3 — 为什么要 Hungarian？

GT 有 N 个实例（~18），queries 有 200 个。  
必须决定「哪个 query 负责哪个 GT」，才能算 L1/GIoU。  
这是 DETR 系列的标准做法。

### Day 3 自检

1. `pred_logits` 最后一维为什么是 1 而不是类别数？  
2. mask 输出分辨率为什么比输入 1008 小？  
3. 训练时有点 prompt 吗？

---

## Day 4：Loss 与反传（B4–B6）

### exp-001 实际在优化什么？

| Loss | 作用 |
|------|------|
| `loss_bbox` | 匹配 query 的框 vs GT 框（L1, cxcywh） |
| `loss_giou` | 框形状/重叠（GIoU） |
| `loss_ce` | query 是否「有物体」（focal/BCE 风格） |
| `presence_loss` | 图像级是否存在可见 GT |

**注意**：配置里 **没有 mask loss**，尽管 `enable_segmentation=True`。

### core_loss 怎么来的？

[Sam3LossWrapper](sam3/train/loss/sam3_loss.py#L161) 把所有 step、aux layer、O2M 分支 loss 加总，再乘以 `sqrt(batch_size)`（这里 batch_size 指 `num_boxes` 的 B 维）。

Trainer 只对 `core_loss` 做 `backward`（[L1112](sam3/train/trainer.py#L1112)）。

### 单卡 DDP 坑（本项目真实踩过）

单 GPU 时若仍包 `DistributedDataParallel`，部分参数在某些 batch 不参与 loss → step 1 崩溃。  
修复：[trainer.py L302](sam3/train/trainer.py#L302) `world_size==1` 时跳过 DDP。

### Day 4 自检

1. 为什么 trace 脚本里要 `_ensure_dist()`？  
2. 10 epoch 训练日志里 `train_all_loss` ~200 和 `core_loss` 什么关系？  
3. 若你想让模型更重视 mask，该改 yaml 哪一项？

---

## Day 5：串讲 + 与推理对比闭环

### 5 分钟口述模板

1. TRUDI 210 图 → SAM3+Qwen 生成 COCO JSON（含 noun_phrase）  
2. 训练 loader **只用 category 名 `"object"`** 作 query  
3. Collator 组 `(1,3,1008,1008)` batch + pad GT  
4. Sam3Image 200 queries 预测框/ mask  
5. Hungarian + box/GIoU/CE loss → 10 epoch → checkpoint  
6. 推理脚本用 **长 noun phrase** 测基座 vs 微调（分布与训练不同，但微调仍提升港口场景检出）

### 与 exp-001 结果对照

| 资源 | 内容 |
|------|------|
| 训练 | `records/logs/train_exp001.log` |
| 对比 | `records/results/exp-001/comparison_summary.json` |
| 简历摘要 | `resume.md` |

### 最终串讲 checklist（关文档后自测）

- [ ] 从 `annotations.json` 到 `core_loss` 全链路  
- [ ] 能解释训练文本 `"object"` vs 推理 noun phrase 的差异  
- [ ] 能说出 3 个 shape：`(3,1008,1008)`、`(1,200,1)`、`(1,N,4)`  
- [ ] 能运行三条验证命令并读懂输出  

### 验证命令（必做）

```bash
python scripts/sft_dataflow_trace.py --doc
python scripts/sft_dataflow_trace.py --run
python scripts/sft_dataflow_trace.py --run --forward
```

---

## 未覆盖阶段（别编造）

| 阶段 | 状态 |
|------|------|
| SAM3 预训练 | Meta 官方权重，本项目未训练 |
| Qwen 标注生成 | 见 `generate_t2m_data.py`，属于数据工程，非训练 forward |
| DPO / RLHF | 未实现 |
| Mask loss 微调 | 配置为 null，可作 exp-002 方向 |

---

## 延伸阅读（本项目内）

- 操作手册：[instruction.md](instruction.md)  
- 实验留痕：[records/experiments.md](records/experiments.md)  
- 留痕规范：[EXPERIMENT_TRACE.md](EXPERIMENT_TRACE.md)  
- 简历版总结：[resume.md](resume.md)
