# 实验留痕指南（通用）

> **给 Cursor 的说明**：本文档是**任意 ML/实验类项目**的留痕权威规范。读取后须遵守全文约定。当用户提到「记录实验」「数据变更」「留痕」「同步记录」「commit」等关键词时，按本文档执行，无需用户重复解释格式。
>
> **使用方式**：将本文件放在新项目根目录；可选复制 §8 中的 Cursor Rule 到 `.cursor/rules/experiment-logging.mdc`。项目若有专属操作手册（如 `instruction.md`），留痕格式以本文档为准，业务流程以项目手册为准。

---

## 1. 目的

为求职者/学习者建立**可验证、可复现、可追问**的实验记录，证明：

- 实验真实跑过（有日志、配置、时间线）
- 决策有理由（参数为何这样选）
- 过程可追溯（数据版本 ↔ 实验编号 ↔ 产出物）

**禁止**：编造指标、伪造日志、夸大贡献。

---

## 2. 标准目录结构

在新项目根目录创建：

```
<project-root>/
├── EXPERIMENT_TRACE.md          # 本文件（通用留痕规范）
├── instruction.md               # 可选：项目专属操作流程
├── records/
│   ├── experiments.md           # 实验主日志（核心）
│   ├── data/
│   │   └── CHANGELOG.md         # 数据集版本变更
│   └── results/
│       └── README.md            # 可视化结果索引
├── configs/
│   └── snapshots/               # 每次实验的配置快照
└── scripts/
    └── record_snapshot.sh       # 可选：一键采集当前状态
```

**原则**：

- `records/` 必须进 Git（文本记录）
- 大文件（权重、原始数据、checkpoint）**不进 Git**，在记录中写路径与获取方式
- 对比图等小体积产出可放 `records/results/`，是否提交由 `.gitignore` 决定

---

## 3. 文件格式

### 3.1 `records/data/CHANGELOG.md` — 数据版本

每次数据集变动追加一条（**不覆盖历史**）：

```markdown
## data-v{N} — {YYYY-MM-DD}

**变更类型**：新增 | 删除 | 重新生成 | 清洗 | 其他

**摘要**：一句话说明改了什么

**详情**：
- 来源/路径：`...`
- 规模：{N} 张图 / {M} 条标注 / ...
- 关键参数：`score_thresh=0.75`, ...

**关联实验**：exp-00X, exp-00Y（可后补）

**验证命令**：
\`\`\`bash
# 例如统计标注数量
\`\`\`
```

### 3.2 `records/experiments.md` — 实验主日志

每次实验一条，状态：`进行中` | `已完成` | `失败` | `已放弃`

```markdown
## exp-{NNN} — {简短标题} — {YYYY-MM-DD}

**状态**：进行中

**动机**：为什么要做这次实验

**数据版本**：data-v{N}

**配置**：
| 项 | 值 |
|----|-----|
| run_name | ... |
| ... | ... |

**配置快照**：`configs/snapshots/exp-{NNN}.yaml`（或等价文件）

**命令**：
\`\`\`bash
# 实际执行的命令
\`\`\`

**结果**：（训练结束后填写）
- 产出路径：`...`
- 关键指标/现象：...

**观察与结论**：

**下一步**：

---
```

### 3.3 `records/results/README.md` — 结果索引

```markdown
| 文件 | 实验 | 说明 |
|------|------|------|
| exp-001_sample01.png | exp-001 | 基座 vs 微调对比 |
```

---

## 4. 触发语与 Cursor 行为

用户说下列内容时，Cursor **必须**更新对应文件，**不得**只在聊天里总结：

| 用户意图 | 触发示例 | Cursor 必须做 |
|----------|----------|---------------|
| 数据变更 | 「记录数据变更：…」 | 追加 `data/CHANGELOG.md` 新版本 |
| 开始实验 | 「开始 exp-003：…」 | 改配置；写快照到 `configs/snapshots/`；在 `experiments.md` 开新条，状态「进行中」 |
| 实验结束 | 「exp-003 完成，日志：…」 | 更新该条：结果、观察、状态「已完成」 |
| 出图/评估 | 「对比图在 records/results/exp-003/」 | 更新 `results/README.md` 与实验条目 |
| 里程碑提交 | 「同步记录并 commit」（未说 push 则不 push） | 核对记录与代码一致 → 写 commit message → commit |
| Resume 描述 | 「写简历项目描述」 | 基于 `experiments.md` 真实条目生成，不夸大 |

**格式要求**：

- 追加写入，不删除历史（除非用户明确要求修正错误）
- 日期用 `YYYY-MM-DD`
- 实验编号 `exp-001` 递增；数据版本 `data-v1` 递增
- 改训练/实验配置时**同步**保存快照

---

## 5. Git 提交规范

- **仅用户明确要求时**才 `git commit` / `git push`
- Commit message 建议：`exp-003: 动词 + 简要说明`
  - 例：`exp-003: train on data-v2 with score_thresh 0.75`
- 一次 commit 对应一个逻辑单元（一次实验配置 / 一次数据版本 / 一次文档同步）
- 勿提交：`.env`、密钥、私有原始数据、超大权重

---

## 6. 禁止事项

- 编造 loss、指标、训练时长
- 用「训练成功」代替具体现象
- 覆盖/删除实验历史（除用户要求纠错）
- 未经要求自动 push
- 在简历/总结中Claim未做过的改动（如「自研模型架构」）

---

## 7. 新项目初始化清单

Cursor 在新项目中读取本文件后，若 `records/` 不存在，应询问用户是否创建 starter kit：

- [ ] `records/experiments.md`（含说明与占位）
- [ ] `records/data/CHANGELOG.md`（含 data-v0 占位）
- [ ] `records/results/README.md`
- [ ] `configs/snapshots/.gitkeep`
- [ ] `scripts/record_snapshot.sh`（按项目路径定制）
- [ ] `.cursor/rules/experiment-logging.mdc`（见 §8）

---

## 8. Cursor Rule 模板（复制到 `.cursor/rules/experiment-logging.mdc`）

```markdown
---
description: 实验留痕：记录数据变更、实验日志、配置快照与 commit
alwaysApply: true
---

# 实验留痕助手

权威规范见项目根目录 `EXPERIMENT_TRACE.md`。

## 触发时必做
- 「记录数据变更」→ 追加 records/data/CHANGELOG.md
- 「开始 exp-XXX」→ 配置快照 + experiments.md 新条目（进行中）
- 「exp-XXX 完成」→ 补全结果与结论（已完成）
- 「同步记录并 commit」→ 核对后 commit；未要求则不 push

## 原则
- 追加不覆盖；不编造指标；改配置必留快照
- 大权重/私有数据不进 Git
```

---

## 9. 三条万能触发语（给用户）

```
记录数据变更：[简述] + [统计数据或粘贴命令输出]
记录实验 exp-XXX：[配置 / 结果 / 现象]
同步记录并 commit（不要 push）
```

配合 `@EXPERIMENT_TRACE.md` 或 `@records/experiments.md` 使用。
