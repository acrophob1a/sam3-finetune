# 实验结果可视化索引

> 对比图、推理样例等放本目录子文件夹，例如 `exp-001/`、`exp-002/`。  
> 规范见 [`EXPERIMENT_TRACE.md`](../../EXPERIMENT_TRACE.md)。

## 命名建议

```
records/results/exp-{NNN}/{描述}_{baseline|finetuned|compare}.png
```

## 索引表

| 文件/目录 | 实验 | 说明 |
|-----------|------|------|
| `exp-001/comparison_summary.json` | exp-001 | 8 组 test 对比数值摘要 |
| `exp-001/*_compare.png` | exp-001 | 基座 vs 微调 side-by-side（8 张） |
| `exp-001/*_baseline.png` | exp-001 | SAM3 基座推理（8 张） |
| `exp-001/*_finetuned.png` | exp-001 | exp-001 微调推理（8 张） |
| `exp-002/comparison_summary.json` | exp-002 | 三模型对比数值（基座 / exp-001 / exp-002） |
| `exp-002/*_compare.png` | exp-002 | 三联 side-by-side（8 张） |
| `exp-002/*_{baseline,exp001,exp002}.png` | exp-002 | 单模型推理 overlay（24 张） |

## exp-002 对比摘要（2026-06-11）

与 exp-001 相同 8 组 test 短 prompt，confidence ≥ 0.5：

| 测试图 | 基座 | exp-001 | exp-002 |
|--------|------|---------|---------|
| 0000.jpg | 0 | 4 | 0 |
| 0010.jpg | 3 | 2 | 0 |
| 0020.jpg | 0 | 1 | 0 |
| 0030.jpg | 0 | 2 | 0 |
| 0040.jpg | 0 | 1 | 0 |
| 0050.jpg | 0 | 1 | 0 |
| 0060.jpg | 6 | 0 | 0 |
| 0069.jpg | 2 | 2 | 0 |
| **合计成功** | 2/8 | **6/8** | 0/8 |

完整分析见 [EXP002_SUMMARY.md](../../EXP002_SUMMARY.md)。推荐展示：`0000_*_compare.png`（exp-001 明显优于 exp-002）。

## exp-001 对比摘要（2026-06-06）

| 测试图 | Prompt（缩写） | 基座检出 | 微调检出 |
|--------|----------------|----------|----------|
| 0000.jpg | blue semi-trailer truck | 0 | 4 |
| 0010.jpg | blue shipping container | 3 | 2 |
| 0020.jpg | stack of shipping containers | 0 | 1 |
| 0030.jpg | blue truck trailer | 0 | 2 |
| 0040.jpg | row of blue semi-trailers | 0 | 1 |
| 0050.jpg | yellow crane | 0 | 1 |
| 0060.jpg | container with logo | 6 | 0 |
| 0069.jpg | truck in port yard | 2 | 2 |

推荐展示：`0000_*_compare.png`、`0020_*_compare.png`、`0040_*_compare.png`（微调明显优于基座）

## 说明

- 小体积对比图已提交 Git，便于 portfolio / 简历展示
- 含隐私的原图勿提交；可放脱敏样例或面试时本地演示
- 大权重、checkpoint 路径写在 `records/experiments.md`，不进 Git
