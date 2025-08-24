# This script runs all training scripts to create the PromptableDeTR model. It starts training the 
# Aligner model first, and then, gets the best checkpoint generate durining training to instanciate 
# the PromptableDeTR model and train it.
#!/bin/bash

# General params.
SEED=${SEED:-"42"}

# Dataset params.
TRAIN_DATASET_DIR=${TRAIN_DATASET_DIR}
VALID_DATASET_DIR=${VALID_DATASET_DIR:-"${TRAIN_DATASET_DIR}"}
IMAGE_DIR=${IMAGE_DIR}

# Model params.
VOCAB_FILE=${VOCAB_FILE}
IMG_ENC_WEIGHT=${IMG_ENC_WEIGHT}
TXT_ENC_WEIGHT=${TXT_ENC_WEIGHT}

# Training params.
BATCH_SIZE=${BATCH_SIZE:-"32"}
LR=${LR:-"1e-4"}
LR_FACTOR=${LR_FACTOR:-"0.1"}
WARMUP_STEPS=${WARMUP_STEPS:-"500"}
FROZEN_STEPS=${FROZEN_STEPS:-"5000"}
MAX_ITER=${MAX_ITER:-"50000"}
EVAL_INTERVAL=${EVAL_INTERVAL:-"100"}
LOG_INTERVAL=${LOG_INTERVAL:-"10"}
OVERFIT_THRESHOLD=${OVERFIT_THRESHOLD:-"1e-3"}
OVERFIT_PATIENCE=${OVERFIT_PATIENCE:-"5"}

EXP_DIR=${EXP_DIR:-"./promptable_detr_exp"}

PRESENCE_WEIGHT=${PRESENCE_WEIGHT:-"1.0"}
L1_WEIGHT=${L1_WEIGHT:-"1.0"}

N_JOINER_LAYERS=${N_JOINER_LAYERS:-"6"}

python install_utilities.py
python train.py \
        --train-dataset-dir $TRAIN_DATASET_DIR \
        --valid-dataset-dir $VALID_DATASET_DIR \
        --image-dir $IMAGE_DIR \
        --shuffle \
        --vocab-file $VOCAB_FILE \
        --imgw $IMG_ENC_WEIGHT \
        --txtw $TXT_ENC_WEIGHT \
        --num-joiner-layers $N_JOINER_LAYERS \
        --max-iter $MAX_ITER \
        --batch-size $BATCH_SIZE \
        --log-interval $LOG_INTERVAL \
        --eval-interval $EVAL_INTERVAL \
        --lr $LR \
        --lr-factor $LR_FACTOR \
        --warmup-steps $WARMUP_STEPS \
        --frozen-steps $FROZEN_STEPS \
        --overfit-threshold $OVERFIT_THRESHOLD \
        --overfit-patience $OVERFIT_PATIENCE \
        --exp-dir $EXP_DIR \
        --presence-weight $PRESENCE_WEIGHT \
        --l1-weight $L1_WEIGHT \
        --seed $SEED

# Get the best model.
BEST_MODEL=$(ls "$EXP_DIR"/*-best-*.pth | sort -t '-' -k5,5nr | head -1)

# Delete other ckpts to free space.
for ckpt in "$EXP_DIR"/*.pth; do
        if [ "$ckpt" != "$BEST_MODEL" ]; then
                rm "$ckpt"
        fi
done

echo "The model has been trained."
echo "- Checkpoint of the best model : $BEST_MODEL"
