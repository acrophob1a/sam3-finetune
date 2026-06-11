# 实验日志

> 规范见 [`EXPERIMENT_TRACE.md`](../EXPERIMENT_TRACE.md)。每次实验追加一条，不覆盖历史。
>
> **实验编号**：`exp-001`, `exp-002`, …  
> **数据版本**：见 [`data/CHANGELOG.md`](data/CHANGELOG.md)

---

## 索引

| 实验 | 日期 | 状态 | 数据版本 | 摘要 |
|------|------|------|----------|------|
| exp-002 | 2026-06-06 | 完成 | data-v1 | noun_phrase 逐条监督，自然语言指令对齐 |
| exp-001 | 2026-06-06 | 完成 | data-v1 | TRUDI 210 张，query=`object` 基线 |

---

<!-- 在下方追加实验条目，最新实验放在索引表更新后、旧条目之上 -->

## exp-002 — noun_phrase 自然语言指令微调 — 2026-06-06

**状态**：完成（5 epoch，2026-06-11 03:52）

**动机**：对齐项目初衷——每条 Qwen `noun_phrase` 作为独立 text query，一句描述 → 一个 mask，而非 exp-001 的 `"object"` 粗粒度监督。

**数据版本**：data-v1（同 `custom0_exp001/annotations.json`，4279 条 noun_phrase）

**配置**：
| 项 | 值 |
|----|-----|
| run_name | exp-002 |
| config | `sam3/train/configs/mydata/text_nounphrase_train.yaml` |
| loader | `COCO_FROM_JSON_NOUN_PHRASE`（`use_noun_phrase_loader: true`） |
| base_model | `pretrained/sam3/sam3.pt` |
| max_epochs | **5**（每图 ~20 query，较 exp-001 更重；省磁盘/时间） |
| checkpoint | 仅最终（`save_freq: 0`）≈ 9.4G |
| 数据盘 | `/root/autodl-tmp` 可用 ~57G |

**配置快照**：`configs/snapshots/exp-002.yaml`

**代码改动**：
- `coco_json_loaders.py`：新增 `COCO_FROM_JSON_NOUN_PHRASE`
- `sam3_image_dataset.py`：`use_noun_phrase_loader` 开关
- `scripts/run_exp002_train_shutdown.sh`：训练完成后关机

**命令**：
```bash
bash scripts/run_exp002_train_shutdown.sh
# 或
python sam3/train/train.py -c configs/mydata/text_nounphrase_train.yaml --use-cluster 0 --num-gpus 1
```

**预期**：每图 ~8–48 条 query（avg ~20），每条 `object_ids_output=[1]`；`find_text_batch` 为去重后的 noun_phrase 列表。

**数据流文档**：[`SFT_DATAFLOW_EXP002.md`](../SFT_DATAFLOW_EXP002.md) · [`SFT_DATAFLOW_EXP002_LEARN.md`](../SFT_DATAFLOW_EXP002_LEARN.md)

**结果**：
- checkpoint：`workdir/exp-002/checkpoints/checkpoint.pt`（9.4G）
- 训练耗时：约 **24 分钟**（5 epoch × 210 steps）
- 最终 loss：Epoch 4 avg ~208（core_loss 208.5）
- presence_dec_acc：~0.97
- 训练日志：`records/logs/train_exp002.log`
- 自动关机：**失败**（容器无 systemd/shutdown，需在 AutoDL 控制台手动关机）

**观察与结论**：
- noun_phrase loader 训练稳定；每 step ~1.3s（约为 exp-001 的 1.6×）
- **三模型推理对比**（2026-06-11，8 组与 exp-001 相同短 prompt）：基座 2/8 · exp-001 **6/8** · exp-002 **0/8** 成功检出
- exp-002 在短 prompt 上全灭，主因 train（长 Qwen noun_phrase）与 test（手写短句）分布不一致；pipeline 本身验证通过
- 对比图：`records/results/exp-002/`（8 组三联 compare + `comparison_summary.json`）
- 推理日志：`records/logs/infer_exp002_compare.log`（本地）
- 总结文档：[EXP002_SUMMARY.md](../EXP002_SUMMARY.md)

**下一步**：
1. ~~训练~~ ✅
2. ~~三模型推理对比~~ ✅
3. （可选）用 JSON 内真实 `noun_phrase` 作 test prompt 再评 exp-002
4. （可选）exp-003：混合 prompt 或更长训练

---

## exp-001 — TRUDI 港口场景基线微调 — 2026-06-06

**状态**：完成（10 epoch，2026-06-06 18:56）

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
| max_epochs | 10 |
| resolution | 1008 |
| train_batch_size | 1 |
| skip_saving_ckpts | false |

**配置快照**：`configs/snapshots/exp-001.yaml`

**代码改动**（2026-06-06）：
- `sam3/infer/generate_t2m_data.py`：新增 `--image_dir`、`--max_image_size`；分割前缩放长边至 1536；SAM 完成后释放显存再加载 Qwen
- `sam3/scripts/data_gen/generate_data.sh`：`SAVE_ROOT=datasets/custom0_exp001`，`batchsize=1`，`num_pts=256`
- `text_only_train.yaml`：`run_name=exp-001`，`num_images=210`，`gpus_per_node=1`，`skip_saving_ckpts=false`，`max_epochs=10`
- `sam3/train/trainer.py`：单卡（`world_size=1`）跳过 DDP 包装，修复 step 1 的 unused-parameter 报错

**训练日志**：`records/logs/train_exp001.log`  
**数据流文档**（2026-06-06）：`SFT_DATAFLOW.md` · `SFT_DATAFLOW_LEARN.md` · `SFT_DATAFLOW_10MIN.md`

**命令**：
```bash
# 数据划分（已完成）
SAMPLE_TRAIN=0 bash scripts/setup_trudi_phase1.sh ground.zip

# 标注生成（已完成 2026-06-06）
export HF_ENDPOINT=https://hf-mirror.com/
bash sam3/scripts/data_gen/generate_data.sh
# 结果：210 images → 4279 annotations，耗时约 100 分钟

# 微调（2026-06-06 18:27 重启）
python sam3/train/train.py \
    -c configs/mydata/text_only_train.yaml \
    --use-cluster 0 \
    --num-gpus 1
# 首次启动在 batch 1 因 DDP unused params 崩溃；单卡跳过 DDP 后正常

# 基座 vs 微调对比推理（2026-06-06 19:08）
python scripts/compare_base_finetuned.py
# 8 组 test 样例 → records/results/exp-001/
```

**结果**：
- 标注路径：`datasets/custom0_exp001/annotations.json`
- 标注规模：**210 张图 / 4279 条标注**（平均每图 ~20 个 mask）
- 生成耗时：约 100 分钟（SAM3 分割 17min + Qwen 文本 83min）
- checkpoint：`workdir/exp-001/checkpoints/checkpoint.pt`（9.4G）
- 训练耗时：约 29 分钟（10 epoch × 210 steps）
- 最终 loss：Epoch 0 avg ~198 → Epoch 9 avg ~240（core_loss 240.2）
- 对比图路径：`records/results/exp-001/`（8 组 baseline/finetuned/compare，见 `comparison_summary.json`）
- 推理日志：`records/logs/infer_exp001_compare.log`（约 66s，8×2 次推理）

**观察与结论**：
- 单卡训练需跳过 DDP 包装；10 epoch 基线微调已完成
- 测试集 8 组样例：基座 6/8 零检出 → 微调后 6/8 成功响应港口专用文本 prompt（置信度 ~0.57–0.70）
- 微调模型对训练域语言表述更敏感，部分通用 prompt 表现与基座互有胜负，后续可加 val 定量评估

**下一步**：
1. ~~等待标注生成完成~~ ✅
2. ~~启动 exp-001 微调训练~~ ✅
3. ~~在 `raw_images_test/` 上跑基座 vs 微调对比图~~ ✅
4. （可选）val 集定量指标、exp-002 参数对比

---
