#!/usr/bin/env bash
set -euo pipefail

# 16GB GPU 前提の控えめな pilot 設定。
# train2017 が揃ったあとに，まずこの設定で quality 改善の有無を見る。

python3 stage1_geometry_finetune/train_text_prompt_finetune.py \
  --data-root "${DATA_ROOT:?set DATA_ROOT}" \
  --sam3-checkpoint "${SAM3_CHECKPOINT:?set SAM3_CHECKPOINT}" \
  --stage1-checkpoint "${STAGE1_CHECKPOINT:?set STAGE1_CHECKPOINT}" \
  --student-backbone "${STUDENT_BACKBONE:-repvit_m1_1}" \
  --student-text-encoder-type "${STUDENT_TEXT_ENCODER_TYPE:-MobileCLIP-S1}" \
  --output-dir "${OUTPUT_DIR:-output/text_prompt_finetune_train_pilot}" \
  --split "${SPLIT:-train}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --epochs "${EPOCHS:-1}" \
  --num-samples "${NUM_SAMPLES:-2048}" \
  --lr "${LR:-5e-6}" \
  --weight-decay "${WEIGHT_DECAY:-1e-2}" \
  --embedding-weight "${EMBEDDING_WEIGHT:-0.0015}" \
  --score-weight "${SCORE_WEIGHT:-0.25}" \
  --gt-mask-weight "${GT_MASK_WEIGHT:-0.5}" \
  --box-weight "${BOX_WEIGHT:-0.2}" \
  --save-every "${SAVE_EVERY:-1}" \
  --train-neck \
  --amp \
  --device "${DEVICE:-cuda}"
