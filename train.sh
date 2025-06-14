# This script runs all training scripts to create the PromptableDeTR model. It starts training the 
# Aligner model first, and then, gets the best checkpoint generate durining training to instanciate 
# the PromptableDeTR model and train it.
#!/bin/bash

# General params.
SEED=${SEED:-"42"}

# Dataset params.
TRAIN_DATASET_DIR=${DATASET_DIR}
VALID_DATASET_DIR=${VALID_DATASET_DIR:-"${DATASET_DIR}"}
IMAGE_DIR=${IMAGE_DIR}
MASK_RATIO=${MASK_RATIO:-"0.8"}

# Model params.
VOCAB_FILE=${VOCAB_FILE}
IMG_ENC_WEIGHT=${IMG_ENC_WEIGHT}
TXT_ENC_WEIGHT=${TXT_ENC_WEIGHT}
NUM_JOINER_LAYERS=${NUM_JOINER_LAYERS:-"4"}

# Training params.
MAX_ITER=${MAX_ITER:-"10000"}
BATCH_SIZE=${BATCH_SIZE:-"32"}
LR=${LR:-"1e-4"}
EVAL_INTERVAL=${EVAL_INTERVAL:-"100"}
LOG_INTERVAL=${LOG_INTERVAL:-"10"}
OVERFIT_THRESHOLD=${OVERFIT_THRESHOLD:-"1e-3"}
OVERFIT_PATIENTE=${OVERFIT_PATIENTE:-"5"}

EXP_DIR=${EXP_DIR:-"./exp_detector"}

GIOU_WEIGHT=${GIOU_WEIGHT:-"1.0"}
PRESENCE_WEIGHT=${PRESENCE_WEIGHT:-"1.0"}
L1_WEIGHT=${L1_WEIGHT:-"1.0"}

python install_utilities.py
python train.py \
        --train-dataset-dir $DATASET_DIR \
        --valid-dataset-dir $VALID_DATASET_DIR \
        --image-dir $IMAGE_DIR \
        --shuffle \
        --vocab-file $VOCAB_FILE \
        --imgw $IMG_ENC_WEIGHT \
        --txtw $TXT_ENC_WEIGHT \
        --image-size $IMG_SIZE \
        --image-tokens 400 100 \
        --emb-dim 128 \
        --proj-dim 512 \
        --emb-dropout-rate 0.1 \
        --heads 8 \
        --ff-dim 2048 \
        --num-joiner-layers $NUM_JOINER_LAYERS \
        --max-iter $MAX_ITER \
        --batch-size $BATCH_SIZE \
        --log-interval $LOG_INTERVAL \
        --eval-interval $EVAL_INTERVAL \
        --lr $LR \
        --overfit-threshold $OVERFIT_THRESHOLD \
        --overfit-patience $OVERFIT_PATIENTE \
        --exp-dir $EXP_DIR \
        --giou-weight $GIOU_WEIGHT \
        --presence-weight $PRESENCE_WEIGHT \
        --l1-weight $L1_WEIGHT

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
