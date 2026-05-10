#!/usr/bin/env bash
set -euo pipefail

# text prompt 品質を保つための prompt-in-the-loop 蒸留。
# full SAM3 teacher checkpoint を前提にする。

python3 stage1_geometry_finetune/train_text_prompt_finetune.py \
  --data-root "${DATA_ROOT:?set DATA_ROOT}" \
  --sam3-checkpoint "${SAM3_CHECKPOINT:?set SAM3_CHECKPOINT}" \
  --stage1-checkpoint "${STAGE1_CHECKPOINT:-}" \
  --student-backbone "${STUDENT_BACKBONE:-tiny_vit_21m}" \
  --student-text-encoder-type "${STUDENT_TEXT_ENCODER_TYPE:-MobileCLIP-S1}" \
  --output-dir "${OUTPUT_DIR:-output/text_prompt_finetune}" \
  --split "${SPLIT:-train}" \
  --img-size "${IMG_SIZE:-1008}" \
  --embed-dim "${EMBED_DIM:-1024}" \
  --embed-size "${EMBED_SIZE:-72}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --epochs "${EPOCHS:-1}" \
  --num-samples "${NUM_SAMPLES:-128}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-2}" \
  --embedding-weight "${EMBEDDING_WEIGHT:-0.0015}" \
  --score-weight "${SCORE_WEIGHT:-1.0}" \
  --gt-mask-weight "${GT_MASK_WEIGHT:-0.5}" \
  --box-weight "${BOX_WEIGHT:-0.2}" \
  --save-every "${SAVE_EVERY:-1}" \
  --device "${DEVICE:-cuda}" \
  ${DISABLE_PROMPT_VARIANTS:+--disable-prompt-variants} \
  ${TRAIN_NECK:+--train-neck} \
  ${AMP_ENABLE:+--amp}
