# 数据集变更日志

> 规范见 [`EXPERIMENT_TRACE.md`](../../EXPERIMENT_TRACE.md)。每次数据变动追加一条，不覆盖历史。

---

## 索引

| 版本 | 日期 | 摘要 | 关联实验 |
|------|------|------|----------|
| data-v0 | 2026-06-05 | 初始占位，尚无真实数据 | — |
| data-v1 | 2026-06-06 | TRUDI 210 train / 70 test 划分完成，标注生成中 | exp-001 |

---

## data-v0 — 2026-06-05

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
ls datasets/raw_images 2>/dev/null | wc -l
```

---

## data-v1 — 2026-06-06

**变更类型**：新增

**摘要**：上传 TRUDI（`ground.zip`），按官方 train/test 划分；210 张训练图用于 SAM3+Qwen 自动标注。

**详情**：
- 数据来源：`ground.zip`（TRUDI 内陆港口数据集，733 实例 / 350 图含 val）
- 原始解压路径：`datasets/TRUDI_raw/ground/`
- 训练图片：`datasets/raw_images_train/` — **210 张**（官方 train 全量）
- 测试图片：`datasets/raw_images_test/` — **70 张**（官方 test，不参与标注生成）
- val 集：`datasets/TRUDI_raw/ground/val/` — **70 张**（本次未使用）
- 生成数据路径：`datasets/custom0_exp001/`
- COCO 标注：`datasets/custom0_exp001/annotations.json`（**已生成**）
- 划分脚本：`SAMPLE_TRAIN=0 bash scripts/setup_trudi_phase1.sh ground.zip`
- 关键参数：`score_thresh=0.75`, `iou_thresh=0.1`, `num_pts=256`, `max_image_size=1536`, `batchsize=1`, `vlm_batchsize=2`
- 参数说明：TRUDI 原图约 4624×2084，需缩放长边至 1536；Qwen 阶段 mask 落盘 + 流式 batch 避免 RAM OOM
- **生成结果**（2026-06-06）：210 张图 → **4279 条** mask+文本标注，耗时约 100 分钟

**关联实验**：exp-001

**验证命令**：
```bash
echo -n "train: "; ls datasets/raw_images_train | wc -l   # 210
echo -n "test:  "; ls datasets/raw_images_test  | wc -l   # 70
python -c "
import json, os
p = 'datasets/custom0_exp001/annotations.json'
if os.path.exists(p):
    d = json.load(open(p))
    print('images:', len(d.get('images', [])), 'annotations:', len(d.get('annotations', [])))
else:
    print('annotations.json 尚未生成')
"
```

---

<!-- 在下方追加新版本：data-v2, data-v3, ... -->
