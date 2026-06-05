# 实验结果可视化索引

> 对比图、推理样例等放本目录子文件夹，例如 `exp-001/`、`exp-002/`。  
> 规范见 [`EXPERIMENT_TRACE.md`](../../EXPERIMENT_TRACE.md)。

## 命名建议

```
records/results/exp-{NNN}/{描述}_{baseline|finetuned}.png
```

示例：
- `exp-001/truck_baseline.png`
- `exp-001/truck_finetuned.png`

## 索引表

| 文件/目录 | 实验 | 说明 |
|-----------|------|------|
| — | — | 尚无结果文件 |

## 说明

- 小体积对比图可提交 Git，便于 portfolio
- 含隐私的原图勿提交；可放脱敏样例或面试时本地演示
- 大权重、checkpoint 路径写在 `records/experiments.md`，不进 Git
