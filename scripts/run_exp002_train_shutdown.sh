#!/bin/bash
# exp-002: noun_phrase per-query training, then shutdown.
set -euo pipefail
cd /root/autodl-tmp/sam3

LOG=records/logs/train_exp002.log
mkdir -p records/logs workdir/exp-002

echo "=== exp-002 train start $(date -Iseconds) ===" | tee "$LOG"
echo "Disk before train:" | tee -a "$LOG"
df -h /root/autodl-tmp | tee -a "$LOG"

python sam3/train/train.py \
    -c configs/mydata/text_nounphrase_train.yaml \
    --use-cluster 0 \
    --num-gpus 1 2>&1 | tee -a "$LOG"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== train exit code: $EXIT_CODE at $(date -Iseconds) ===" | tee -a "$LOG"

if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "Training failed, NOT shutting down." | tee -a "$LOG"
    exit "$EXIT_CODE"
fi

if [[ -f workdir/exp-002/checkpoints/checkpoint.pt ]]; then
    ls -lh workdir/exp-002/checkpoints/checkpoint.pt | tee -a "$LOG"
else
    echo "WARNING: checkpoint not found" | tee -a "$LOG"
fi

echo "Disk after train:" | tee -a "$LOG"
df -h /root/autodl-tmp | tee -a "$LOG"

echo "Shutting down in 30 seconds..." | tee -a "$LOG"
sleep 30
/sbin/shutdown -h now || /usr/sbin/shutdown -h now || poweroff
