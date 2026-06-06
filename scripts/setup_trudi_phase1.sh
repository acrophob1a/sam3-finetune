#!/usr/bin/env bash
# 阶段①：解压 TRUDI，按官方 train/test 划分复制到 raw_images_train / raw_images_test
#
# 用法（在项目根目录）：
#   # 方式 A：上传 zip 后
#   bash scripts/setup_trudi_phase1.sh datasets/TRUDI.zip
#
#   # 方式 B：已手动解压到 datasets/TRUDI_raw/
#   bash scripts/setup_trudi_phase1.sh
#
# 可选：只从官方 train 中抽样 N 张用于 exp-001 快速验证
#   SAMPLE_TRAIN=100 bash scripts/setup_trudi_phase1.sh datasets/TRUDI.zip

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ZIP_PATH="${1:-}"
RAW_DIR="datasets/TRUDI_raw"
TRAIN_DIR="datasets/raw_images_train"
TEST_DIR="datasets/raw_images_test"
SAMPLE_TRAIN="${SAMPLE_TRAIN:-0}"   # 0 = 使用全部官方 train
SEED=42

pick_split_dir() {
  local base="$1"
  local kind="$2"  # train or test
  for d in "$base/$kind" "$base/${kind}ing" "$base/val" "$base/validation"; do
    if [ -d "$d" ]; then
      echo "$d"
      return 0
    fi
  done
  return 1
}

collect_images() {
  local dir="$1"
  python - <<'PY' "$dir"
import glob, sys
dir = sys.argv[1]
ext = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG')
print("\n".join(sorted(p for p in glob.glob(f"{dir}/**/*", recursive=True)
                      if p.lower().endswith(ext))))
PY
}

echo "=== TRUDI 阶段① setup ==="
echo "root: $ROOT"

if [ -n "$ZIP_PATH" ]; then
  if [ ! -f "$ZIP_PATH" ]; then
    echo "ERROR: zip not found: $ZIP_PATH"
    exit 1
  fi
  echo "解压: $ZIP_PATH -> $RAW_DIR"
  rm -rf "$RAW_DIR"
  mkdir -p "$RAW_DIR"
  unzip -q "$ZIP_PATH" -d "$RAW_DIR"
fi

if [ ! -d "$RAW_DIR" ] || [ -z "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
  echo "ERROR: $RAW_DIR 为空。请先上传 TRUDI 压缩包，例如："
  echo "  datasets/TRUDI.zip"
  exit 1
fi

# 若 zip 解压后多一层目录，自动下探
BASE="$RAW_DIR"
while [ "$(find "$BASE" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ] && \
      [ "$(find "$BASE" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 0 ]; do
  BASE="$(find "$BASE" -mindepth 1 -maxdepth 1 -type d | head -1)"
done

TRAIN_SRC="$(pick_split_dir "$BASE" train || true)"
TEST_SRC="$(pick_split_dir "$BASE" test || true)"

if [ -z "$TRAIN_SRC" ] || [ -z "$TEST_SRC" ]; then
  echo "ERROR: 未找到官方 train/ 或 test/ 目录。当前结构："
  find "$RAW_DIR" -maxdepth 3 -type d | head -30
  exit 1
fi

echo "官方 train: $TRAIN_SRC"
echo "官方 test:  $TEST_SRC"

mkdir -p "$TRAIN_DIR" "$TEST_DIR"
find "$TRAIN_DIR" "$TEST_DIR" -type f -delete 2>/dev/null || true

python - <<PY
import glob, os, random, shutil

train_src = "$TRAIN_SRC"
test_src = "$TEST_SRC"
train_dst = "$TRAIN_DIR"
test_dst = "$TEST_DIR"
sample_train = int("$SAMPLE_TRAIN")
seed = int("$SEED")
ext = ('.jpg', '.jpeg', '.png', '.webp')

def collect(root):
    out = []
    for p in glob.glob(root + "/**/*", recursive=True):
        if p.lower().endswith(ext):
            out.append(p)
    return sorted(out)

train_imgs = collect(train_src)
test_imgs = collect(test_src)

if sample_train > 0 and len(train_imgs) > sample_train:
    random.seed(seed)
    train_imgs = sorted(random.sample(train_imgs, sample_train))

for i, p in enumerate(train_imgs):
    shutil.copy2(p, os.path.join(train_dst, f"{i:04d}{os.path.splitext(p)[1].lower()}"))
for i, p in enumerate(test_imgs):
    shutil.copy2(p, os.path.join(test_dst, f"{i:04d}{os.path.splitext(p)[1].lower()}"))

print(f"train -> {train_dst}: {len(train_imgs)}")
print(f"test  -> {test_dst}: {len(test_imgs)}")
PY

echo ""
echo "=== 验证 ==="
echo -n "raw_images_train: "; ls "$TRAIN_DIR" | wc -l
echo -n "raw_images_test:  "; ls "$TEST_DIR" | wc -l
echo "阶段①完成。官方划分：train 用于后续生成标注，test 仅用于推理对比。"
