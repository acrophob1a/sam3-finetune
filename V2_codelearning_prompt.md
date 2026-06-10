# 代码学习 / 数据流拆解 · 通用提示词

基于**本项目的完整代码实现**，详细拆解【预训练 / 微调 / SFT / DPO】等阶段的数据流。

**默认读者**：有 PyTorch 基础、第一次做 VLM SFT；先直觉 + 实测，再堆公式；不默认读者已熟悉 mRoPE、masked_scatter、Collator 等。

---

## 0. 文档产出与源码导航（必须）

产出物应**可点击、可运行、可自测**，禁止只写「见 xxx.py」而不给行号。

### 0.1 三件套分工

| 产出 | 文件名（本项目示例） | 定位 |
|------|---------------------|------|
| **规格版** | `SFT_DATAFLOW.md` | shape 速查、精确行号、风险表 |
| **学习版** | `SFT_DATAFLOW_LEARN.md` | Day 1–N 路线、概念解释、自检题 |
| **可运行版** | `scripts/sft_dataflow_trace.py` | `--doc` / `--run` / `--step Ax` 实测 |

三份文档的 **步骤编号（如 A1–B6）应对齐**。

### 0.2 可点击链接规范（缺一不可）

- **Markdown 链接**：`[符号名](相对路径/文件.py#L行号)`  
  例：`[SlakeVQADataset.__getitem__](finetuning/dataset/slake_vqa_dataset.py#L90)`
- **代码引用块**：长片段用 IDE 可跳转格式  
  ` ```startLine:endLine:相对路径 `（单独占一行，不加 list 前缀）
- **禁止**：仅写文件名、无 `#L行号`、无链接的「见 tsv_dataset.py」
- **transformers 等外部库**：给 `site-packages/.../modeling_xxx.py` 路径 + 行号，或官方 GitHub 链接

### 0.3 可运行验证

每个大阶段末尾给出至少一条命令，例如：

```bash
python scripts/sft_dataflow_trace.py --doc --step A6
python scripts/sft_dataflow_trace.py --run
python scripts/sft_dataflow_trace.py --run --forward
```

**所有 shape 必须在该命令输出或断点实测中可核对**，不能仅靠手算。

---

## 1. 范围声明（必须）

- 只拆解本项目**实际实现并跑通**的阶段。
- **未实现阶段**（如 Pretrain、DPO/RL）单独写「未覆盖」，不编造数据流。
- 本项目示例：仅 **SFT 监督微调**；基座 Qwen2.5-VL-3B-Instruct；Pretrain / DPO 不在 SLAKE 主流程中。

---

## 2. 全流程边界（必须）

从**最原始输入数据格式**到**最终 loss 标量**，严格分为两阶段：

| 阶段 | 运行位置 | 典型入口 | 典型出口 |
|------|---------|---------|---------|
| **A. 数据预处理** | CPU | `Dataset.__getitem__` | `Collator.__call__` 输出的 batch |
| **B. 模型前向** | GPU | `model(**batch)` / `forward` | `logits` + `loss` |

两阶段分别引用项目中对应的代码文件和函数，不得混为一谈。

---

## 3. 每一步的五要素（缺一不可）

每一个操作步骤都必须包含：

1. **可点击的精确代码定位**（文件 + 行号 + 函数名；见 §0.2）
2. **输入**的精确 shape + **具体内容示例**（tensor / dict / 字符串均可）
3. **输出**的精确 shape + **具体内容示例**
4. **每个维度的含义**（如 `[B,T,C]` 中 B=batch_size, T=序列长度；`(117,)` 是 1D 而非 1×117）
5. **本步在项目中的核心目的**（一句话）

---

## 4. 最小测试样本（必填，贯穿全文）

使用**一个最小、完整、可实测**的样本；所有 shape 计算与文档描述必须基于该样本。

### 4.1 模板（VLM / 本项目）

```python
raw_sample = {
    "img_name": "xmlab0/source.jpg",           # 相对 image_root
    "question": "What modality is used to take this image?",
    "answer": "MRI",
    "q_lang": "en",
}
# 图片路径：datasets/SLAKE/imgs/xmlab0/source.jpg
# PIL 尺寸：(256, 256) RGB
# 实测（--run）：input_ids (117,); N_img=81; pixel_values (324, 1176); grid_thw [1,18,18]
```

### 4.2 其他项目类型（替换时保留「实测字段」）

```python
# LLM 示例
raw_sample = {"text": "今天天气很好", "labels": "..."}

# 视频示例
raw_sample = {"video": "...", "caption": "...", "frames": 8}
```

文档中须注明：**shape 数据来源**（`--run` / 日志 / 断点），而非纯推导。

---

## 5. 符号表（文档开头固定一节）

全文统一符号，至少包含：

| 符号 | 含义 |
|------|------|
| B | batch_size |
| T | 单条文本 token 数 |
| T_max | batch 内 pad 后最大序列长度 |
| V | 词表大小 |
| H | LLM hidden_size |
| P | Vision 输入 patch 数 |
| N_img | 文本序列中 `<\|image_pad\|>` 个数 |
| C_patch | 每个 patch 展平维（如 1176 = 2×14×14×3） |

新符号首次出现时必须解释；与 `grid_thw [T,H,W]` 等专有名词保持一致。

---

## 6. 多模态 / 多输入专项（必须）

1. **先分模态**：文本、图像（及视频若有）各自从原始输入到 tensor 的独立数据流。
2. **再写融合点**：精确到**文件 + 行号**；融合前 / 后 shape；融合方式（拼接 / cross-attention / masked_scatter / 加权等）。
3. **本项目示例**：Vision Tower → `image_embeds (81,2048)` → `masked_scatter` 填入 `input_ids==151655` 的 81 个槽位；shape 不变 `(B,T,H)`。

---

## 7. 前向传播分模块拆解（必须）

从 `forward` 入口开始，按核心模块逐步写 shape 变化，直至 logits 与 loss，例如：

- embed_tokens
- Vision Tower + Merger
- 多模态融合
- Decoder ×N
- lm_head
- shift + CrossEntropy（ignore_index）

外部库模块（transformers）同样给可点击路径或行号。

---

## 8. 不可跳过的中间步骤（必须）

包括但不限于：

- 特殊 token、Chat Template、`<image>` → vision placeholder
- padding / truncation（含 Collator pad 值：pad_token_id vs IGNORE_INDEX=-100）
- 归一化、resize、patchify
- `attention_mask` / causal mask / label mask
- labels 偏移（因果 LM shift）与 `-100` 掩码
- **position_ids / mRoPE** 与 `grid_thw` 的关系（若为多模态模型）

---

## 9. Dataset / Collator / Trainer 调用链（必须）

文档须单独说明「谁创建、谁调用、输入输出 shape」，例如：

| 组件 | 创建位置 | 调用方式 | 输入 → 输出 |
|------|---------|---------|-------------|
| `Dataset` | `BUILDER.build(cfg.train_dataset)` | `ds[i]` → `__getitem__` | 索引 → **单条** dict |
| `Collator` | `BUILDER.build(cfg.data_collator)` | `collator(instances)` → `__call__` | **B 条** list → **1 个** batch dict |
| `Trainer` | `train.py` | 内部 DataLoader 调 Collator | batch → `model(**batch)` |

避免只讲 tensor 不讲调用关系。

---

## 10. 流程图与风险（必须）

### 10.1 纯文本 shape 流程图

- **至少两条**：单样本（Dataset 输出）+ batch（Collator 输出）
- 从原始 JSON/图像到 loss 标量，逐步标注 shape

### 10.2 Shape 不匹配风险表

| # | 代码位置（可点击） | 风险描述 | 后果 |
|---|-------------------|---------|------|
| 1 | … | … | … |

### 10.3 与实验记录对齐（可选但推荐）

- 训练：run_name、steps、effective batch、关键超参
- 评估：指标、样本数、base vs fine-tuned 对比

---

## 11. 自检题与验证命令（必须）

- 每个大 Part 或 Day 末尾：**1–3 道自检题**（可先自答再对照文档）。
- 每 Part 至少 **1 条** §0.3 风格的可运行验证命令。
- 学习版文档须含**串讲 checklist**（关闭文档后能否口述全流程）。

---

## 使用说明

1. 将本文作为 Prompt 交给 AI 或自用 checklist，生成 / 更新 `SFT_DATAFLOW.md`、`SFT_DATAFLOW_LEARN.md`、`sft_dataflow_trace.py`。
2. 生成后自检：随机抽 3 个步骤，链接能否 Cmd/Ctrl+Click 跳转；`--run` 输出是否与文档 shape 一致。
3. 相关文档索引：
   - [`SFT_DATAFLOW.md`](SFT_DATAFLOW.md) — 规格版
   - [`SFT_DATAFLOW_LEARN.md`](SFT_DATAFLOW_LEARN.md) — 学习版
   - [`scripts/sft_dataflow_trace.py`](scripts/sft_dataflow_trace.py) — 可运行版
   - [`docs/ML_PRACTITIONER_HANDBOOK.md`](docs/ML_PRACTITIONER_HANDBOOK.md) — PyTorch / 面试补充

---

*提示词版本：v2 · 2026-06 · 含可点击源码、三件套产出、调用链、符号表、自检与范围声明*
