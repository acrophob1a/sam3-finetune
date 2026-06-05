# 实验日志

> 规范见 [`EXPERIMENT_TRACE.md`](../EXPERIMENT_TRACE.md)。每次实验追加一条，不覆盖历史。
>
> **实验编号**：`exp-001`, `exp-002`, …  
> **数据版本**：见 [`data/CHANGELOG.md`](data/CHANGELOG.md)

---

## 索引

| 实验 | 日期 | 状态 | 数据版本 | 摘要 |
|------|------|------|----------|------|
| — | — | — | — | 尚无实验记录 |

---

<!-- 在下方追加实验条目，最新实验放在索引表更新后、旧条目之上 -->

## exp-001 — baseline — （待开始）

**状态**：未开始

**动机**：（填写：例如「验证完整 pipeline：数据生成 → 微调 → 推理对比」）

**数据版本**：data-v0（见 CHANGELOG）

**配置**：
| 项 | 值 |
|----|-----|
| run_name | try-0 |
| config | `sam3/train/configs/mydata/text_only_train.yaml` |
| num_images | （填实际数量） |
| gpus | 1 |
| max_epochs | 20 |
| resolution | 1008 |

**配置快照**：`configs/snapshots/exp-001.yaml`（开跑前生成）

**命令**：
```bash
python sam3/train/train.py \
    -c configs/mydata/text_only_train.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

**结果**：（训练结束后填写）
- 产出路径：`workdir/<run_name>/checkpoints/`
- 关键指标/现象：

**观察与结论**：

**下一步**：

---
