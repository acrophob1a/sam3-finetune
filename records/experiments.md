# 实验日志

> 规范见 [`EXPERIMENT_TRACE.md`](../EXPERIMENT_TRACE.md)。每次实验追加一条，不覆盖历史。
>
> **实验编号**：`exp-001`, `exp-002`, …  
> **数据版本**：见 [`data/CHANGELOG.md`](data/CHANGELOG.md)

---

## 索引

| 实验 | 日期 | 状态 | 数据版本 | 摘要 |
|------|------|------|----------|------|
| exp-001 | 2026-06-06 | 进行中 | data-v1 | TRUDI 210 张，港口场景语言指令分割基线 |

---

<!-- 在下方追加实验条目，最新实验放在索引表更新后、旧条目之上 -->

## exp-001 — TRUDI 港口场景基线微调 — 2026-06-06

**状态**：进行中（标注已完成，待开训）

**动机**：验证完整 pipeline——TRUDI 数据划分 → SAM3+Qwen 自动生成 (图像, 文本, mask) 三元组 → SAM3 语言指令分割微调 → 测试集推理对比。作为后续 exp-002/003 的基线。

**数据版本**：data-v1（见 [`data/CHANGELOG.md`](data/CHANGELOG.md)）

**配置**：
| 项 | 值 |
|----|-----|
| run_name | exp-001 |
| config | `sam3/train/configs/mydata/text_only_train.yaml` |
| annotation_path | `datasets/custom0_exp001/annotations.json` |
| base_model | `pretrained/sam3/sam3.pt` |
| num_images | 210（train） |
| test_images | 70（`raw_images_test`，仅推理） |
| gpus | 1（A800 80GB） |
| max_epochs | 20 |
| resolution | 1008 |
| train_batch_size | 1 |
| skip_saving_ckpts | false |

**配置快照**：`configs/snapshots/exp-001.yaml`

**代码改动**（2026-06-06）：
- `sam3/infer/generate_t2m_data.py`：新增 `--image_dir`、`--max_image_size`；分割前缩放长边至 1536；SAM 完成后释放显存再加载 Qwen
- `sam3/scripts/data_gen/generate_data.sh`：`SAVE_ROOT=datasets/custom0_exp001`，`batchsize=1`，`num_pts=256`
- `text_only_train.yaml`：`run_name=exp-001`，`num_images=210`，`gpus_per_node=1`，`skip_saving_ckpts=false`

**标注生成日志**：`records/logs/data_gen_exp001.log`

**命令**：
```bash
# 数据划分（已完成）
SAMPLE_TRAIN=0 bash scripts/setup_trudi_phase1.sh ground.zip

# 标注生成（已完成 2026-06-06）
export HF_ENDPOINT=https://hf-mirror.com/
bash sam3/scripts/data_gen/generate_data.sh
# 结果：210 images → 4279 annotations，耗时约 100 分钟

# 微调（下一步）
python sam3/train/train.py \
    -c configs/mydata/text_only_train.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

**结果**：（部分完成）
- 标注路径：`datasets/custom0_exp001/annotations.json`
- 标注规模：**210 张图 / 4279 条标注**（平均每图 ~20 个 mask）
- 生成耗时：约 100 分钟（SAM3 分割 17min + Qwen 文本 83min）
- checkpoint 路径：`workdir/exp-001/checkpoints/`（待训练）
- 对比图路径：`records/results/exp-001/`（待生成）

**观察与结论**：（待填）

**下一步**：
1. ~~等待标注生成完成~~ ✅
2. 启动 exp-001 微调训练
3. 在 `raw_images_test/` 上跑基座 vs 微调对比图

---
