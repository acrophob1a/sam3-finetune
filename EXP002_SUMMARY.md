# exp-002 实验总结 — noun_phrase 自然语言指令微调

> 训练完成：2026-06-11 03:52 · 推理对比：2026-06-11  
> 仓库：[github.com/acrophob1a/sam3-finetune](https://github.com/acrophob1a/sam3-finetune)

---

## 一句话结论

exp-002 成功跑通 **「一条 Qwen 描述 → 一个 mask」** 的训练 pipeline，5 epoch 训练稳定（presence_acc ≈ 97%）。但在与 exp-001 **相同的 8 组短 prompt 测试** 上，exp-002 **零检出**，不如 exp-001（6/8 成功）。主要归因于 **训练文本（长 noun_phrase）与测试 prompt（短句）分布不一致**，而非 pipeline 失败。

---

## exp-001 vs exp-002 对照

| | exp-001 | exp-002 |
|--|---------|---------|
| 训练 query | `"object"`（类别名） | **Qwen `noun_phrase` 完整句** |
| 每 query 监督 | 全图所有 mask（~18） | **1 个 mask** |
| 每图 query 数 | 1 | ~8–48（avg ~20） |
| epoch | 10 | 5 |
| 训练耗时 | ~29 min | ~24 min |
| Loader | `COCO_FROM_JSON` | `COCO_FROM_JSON_NOUN_PHRASE` |
| checkpoint | `workdir/exp-001/checkpoints/checkpoint.pt` | `workdir/exp-002/checkpoints/checkpoint.pt` |

---

## 训练结果

| 指标 | 值 |
|------|-----|
| 数据 | data-v1 · 210 图 · 4279 标注（同 exp-001） |
| 最终 core_loss | **208.5**（Epoch 4 avg） |
| presence_dec_acc | **0.973** |
| checkpoint 大小 | 9.4G |
| 配置 | `text_nounphrase_train.yaml` |
| 日志 | `records/logs/train_exp002.log`（本地，未进 Git） |

训练过程无崩溃；每 step ~1.3s（约为 exp-001 的 1.6×，因每图 ~20 条 query）。

---

## 三模型推理对比（8 组 test prompt）

**命令**：

```bash
python scripts/compare_three_models.py
# 输出：records/results/exp-002/
```

**测试条件**：与 exp-001 相同 8 张 test 图 + 8 条手写英文短 prompt（非 Qwen 长句）。

### 检出数量（confidence ≥ 0.5）

| 测试图 | Prompt（缩写） | 基座 | exp-001 | **exp-002** |
|--------|----------------|------|---------|-------------|
| 0000.jpg | blue semi-trailer truck | 0 | **4** | 0 |
| 0010.jpg | blue shipping container | 3 | 2 | 0 |
| 0020.jpg | stack of containers | 0 | **1** | 0 |
| 0030.jpg | blue truck trailer | 0 | **2** | 0 |
| 0040.jpg | row of semi-trailers | 0 | **1** | 0 |
| 0050.jpg | yellow crane | 0 | **1** | 0 |
| 0060.jpg | container with logo | **6** | 0 | 0 |
| 0069.jpg | truck in port yard | 2 | 2 | 0 |
| **成功检出（>0）** | | **2/8** | **6/8** | **0/8** |

数值详情：`records/results/exp-002/comparison_summary.json`

### 对比图

三联 side-by-side（基座 | exp-001 | exp-002）：

- `records/results/exp-002/0000_*_compare.png` — exp-001 明显优于基座与 exp-002
- `records/results/exp-002/0020_*_compare.png`
- `records/results/exp-002/0040_*_compare.png`

---

## 分析与解读

### 为什么 exp-002 在短 prompt 上全灭？

1. **训练分布**：每条 query 是 Qwen 生成的 **长句**（平均 ~70 字符），且只对应 **一个具体 mask 区域**。
2. **测试分布**：8 组 prompt 是 **短类别描述**（如 *"A blue truck trailer"*），与 exp-001 评测一致，但与 exp-002 训练文本风格不同。
3. **监督粒度**：exp-002 学的是「这句话对应图中哪一块」，不是「object 类找全图所有实例」；短泛化 prompt 可能无法激活正确的文本-视觉对齐。
4. **训练量**：5 epoch vs exp-001 的 10 epoch，语言塔更新可能不足。

### exp-002 的价值（pipeline 层面）

- 验证了 `COCO_FROM_JSON_NOUN_PHRASE` loader 与 `use_noun_phrase_loader` 可稳定训练
- 语言 backbone 参与反传（`lr_language_backbone` 生效）
- 为后续 **train/test 同分布** 评估（用 JSON 里真实 `noun_phrase` 作 prompt）打下基础

### 建议下一步（exp-003 方向）

1. 推理时用 **训练集同款 noun_phrase** 作 prompt，再对比 exp-001 / exp-002
2. 增加 epoch 或混合训练（短 prompt + noun_phrase）
3. 引入 val IoU / mAP 定量指标

---

## 相关文档

| 文档 | 说明 |
|------|------|
| [SFT_DATAFLOW_EXP002_LEARN.md](SFT_DATAFLOW_EXP002_LEARN.md) | 数据流学习版 |
| [SFT_DATAFLOW_EXP002.md](SFT_DATAFLOW_EXP002.md) | 数据流规格版 |
| [records/experiments.md](records/experiments.md) | 实验留痕 |
| [records/results/README.md](records/results/README.md) | 对比图索引 |

---

## 复现

```bash
# 训练（已完成）
python sam3/train/train.py -c configs/mydata/text_nounphrase_train.yaml --use-cluster 0 --num-gpus 1

# 三模型对比推理
python scripts/compare_three_models.py
```
