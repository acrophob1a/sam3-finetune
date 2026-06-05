# 数据集变更日志

> 规范见 [`EXPERIMENT_TRACE.md`](../../EXPERIMENT_TRACE.md)。每次数据变动追加一条，不覆盖历史。

---

## 索引

| 版本 | 日期 | 摘要 | 关联实验 |
|------|------|------|----------|
| data-v0 | — | 初始占位，尚无真实数据 | — |

---

## data-v0 — （项目初始化）

**变更类型**：其他

**摘要**：留痕结构已创建，训练数据尚未生成。

**详情**：
- 原始图片路径：`datasets/raw_images/`（待放入）
- 生成数据路径：`datasets/custom0/`
- COCO 标注：`datasets/custom0/annotations.json`（待生成）
- 数据生成脚本：`bash sam3/scripts/data_gen/generate_data.sh`

**关联实验**：（待补）

**验证命令**：
```bash
# 原始图片数量
ls datasets/raw_images 2>/dev/null | wc -l

# 标注统计（生成后）
python -c "
import json, os
p = 'datasets/custom0/annotations.json'
if os.path.exists(p):
    d = json.load(open(p))
    print('images:', len(d.get('images', [])), 'annotations:', len(d.get('annotations', [])))
else:
    print('annotations.json 尚未生成')
"
```

---

<!-- 在下方追加新版本：data-v1, data-v2, ... -->
