#!/usr/bin/env bash
# 采集当前项目状态，输出可粘贴到 Cursor 的留痕摘要
# 用法：bash scripts/record_snapshot.sh [exp-id]
# 示例：bash scripts/record_snapshot.sh exp-001

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXP_ID="${1:-}"
DATE="$(date +%Y-%m-%d)"
TIME="$(date +%H:%M:%S)"

echo "========== 实验快照 ${DATE} ${TIME} =========="
echo "项目根目录: $ROOT"
echo ""

# Git 信息（若在 git 仓库中）
if git rev-parse --git-dir >/dev/null 2>&1; then
  echo "--- Git ---"
  echo "branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
  echo "dirty:  $(git status --porcelain 2>/dev/null | head -5 | wc -l | tr -d ' ') files changed (showing max 5)"
  git status --porcelain 2>/dev/null | head -5 || true
  echo ""
fi

# 原始图片
echo "--- 原始图片 (datasets/raw_images) ---"
if [[ -d datasets/raw_images ]]; then
  RAW_COUNT=$(find datasets/raw_images -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) 2>/dev/null | wc -l | tr -d ' ')
  echo "count: $RAW_COUNT"
else
  echo "目录不存在"
fi
echo ""

# COCO 标注
echo "--- 训练标注 (datasets/custom0/annotations.json) ---"
ANN="datasets/custom0/annotations.json"
if [[ -f "$ANN" ]]; then
  python3 -c "
import json
d = json.load(open('$ANN'))
print('images:', len(d.get('images', [])))
print('annotations:', len(d.get('annotations', [])))
print('categories:', len(d.get('categories', [])))
"
else
  echo "尚未生成"
fi
echo ""

# 权重
echo "--- 模型权重 ---"
for p in pretrained/sam3/sam3.pt pretrained/Qwen2.5-VL-7B-Instruct/config.json; do
  if [[ -e "$p" ]]; then
    echo "OK  $p"
  else
    echo "MISS $p"
  fi
done
echo ""

# 训练产出
echo "--- 训练产出 (workdir) ---"
if [[ -d workdir ]]; then
  find workdir -name '*.pt' -o -name 'checkpoint*' 2>/dev/null | head -10 || echo "(无 checkpoint 或目录为空)"
else
  echo "workdir/ 不存在"
fi
echo ""

# 配置快照
if [[ -n "$EXP_ID" ]]; then
  SNAPSHOT_DIR="configs/snapshots"
  SRC="sam3/train/configs/mydata/text_only_train.yaml"
  DEST="$SNAPSHOT_DIR/${EXP_ID}.yaml"
  mkdir -p "$SNAPSHOT_DIR"
  if [[ -f "$SRC" ]]; then
    cp "$SRC" "$DEST"
    echo "--- 配置快照 ---"
    echo "已保存: $DEST"
  else
    echo "WARN: 未找到 $SRC，跳过快照"
  fi
  echo ""
fi

echo "--- 粘贴给 Cursor ---"
echo "请根据以上快照更新留痕记录。"
if [[ -n "$EXP_ID" ]]; then
  echo "实验编号: $EXP_ID"
fi
echo "（可将本段输出直接发给 Cursor：「记录实验 ${EXP_ID:-}，快照如下：…」）"
echo "=========================================="
