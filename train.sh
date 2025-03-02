# This script runs all training scripts to create the PromptableDeTR model. It starts training the 
# Aligner model first, and then, gets the best checkpoint generate durining training to instanciate 
# the PromptableDeTR model and train it.
#!/bin/bash

# General params.
SEED=${SEED:-"42"}

# Dataset params.
DATASET_DIR=${DATASET_DIR}
IMAGE_DIR=${IMAGE_DIR}
VALID_SPLIT=${VALID_SPLIT:-"0.2"}
MASK_RATIO=${MASK_RATIO:-"0.8"}

# Model params.
VOCAB_FILE=${VOCAB_FILE}
IMG_ENC_WEIGHT=${IMG_ENC_WEIGHT}
TXT_ENC_WEIGHT=${TXT_ENC_WEIGHT}
IMG_SIZE="640" # Mandatory
IMG_TOKENS="[1600, 400, 100]" # Mandatory
EMB_DIM="128" # Mandatory
PROJ_DIM="512" # Mandatory
HEADS="8" # Mandatory
FF_DIM="2048" # Mandatory
NUM_JOINER_LAYERS=${NUM_JOINER_LAYERS:-"4"}

# Training params.
MAX_ITER=${MAX_ITER:-"10000"}
BATCH_SIZE=${BATCH_SIZE:-"32"}
LR=${LR:-"1e-4"}
EVAL_INTERVAL=${EVAL_INTERVAL:-"100"}
LOG_INTERVAL=${LOG_INTERVAL:-"10"}
OVERFIT_THRESHOLD=${OVERFIT_THRESHOLD:-"1e-3"}
OVERFIT_PATIENTE=${OVERFIT_PATIENTE:-"5"}

ALINGER_EXP_DIR="./exp_aligner"
DETECTOR_EXP_DIR="./exp_detector"

GIOU_WEIGHT=${GIOU_WEIGHT:-"1.0"}
PRESENCE_WEIGHT=${PRESENCE_WEIGHT:-"1.0"}
L1_WEIGHT=${L1_WEIGHT:-"1.0"}

python install_utilities.py

python train_aligner.py \
        --dataset-dir $DATASET_DIR \
        --image-dir $IMAGE_DIR \
        --aligner \
        --valid-split $VALID_SPLIT \
        --mask-ratio $MASK_RATIO \
        --shuffle \
        --vocab-file $VOCAB_FILE \
        --imgw $IMG_ENC_WEIGHT \
        --txtw $TXT_ENC_WEIGHT \
        --image-size $IMG_SIZE \
        --image-tokens 1600 400 100 \
        --emb-dim 128 \
        --proj-dim 512 \
        --emb-dropout-rate 0.1 \
        --heads 8 \
        --ff-dim 2048 \
        --num-joiner-layers $NUM_JOINER_LAYERS \
        --max-iter $MAX_ITER \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --overfit-threshold $OVERFIT_THRESHOLD \
        --overfit-patience $OVERFIT_PATIENTE \
        --exp-dir $ALINGER_EXP_DIR

# Get the base model weight.
BASE_MODEL_WEIGHT=$(ls "$ALINGER_EXP_DIR"/*-best-*.pth | sort -t '-' -k4,4nr | head -1)

python train_detector.py \
        --dataset-dir $DATASET_DIR \
        --image-dir $IMAGE_DIR \
        --valid-split $VALID_SPLIT \
        --shuffle \
        --vocab-file $VOCAB_FILE \
        --bmw $BASE_MODEL_WEIGHT \
        --image-size $IMG_SIZE \
        --image-tokens 1600 400 100 \
        --emb-dim 128 \
        --proj-dim 512 \
        --emb-dropout-rate 0.1 \
        --heads 8 \
        --ff-dim 2048 \
        --num-joiner-layers $NUM_JOINER_LAYERS \
        --max-iter $MAX_ITER \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --overfit-threshold $OVERFIT_THRESHOLD \
        --overfit-patience $OVERFIT_PATIENTE \
        --exp-dir $DETECTOR_EXP_DIR \
        --giou-weight $GIOU_WEIGHT \
        --presence-weight $PRESENCE_WEIGHT \
        --l1-weight $L1_WEIGHT

BEST_MODEL=$(ls $DETECTOR_EXP_DIR/*-best-*.pth | sort -t '-' -k4,4nr | head -1)
echo "All models has been trained."
echo "- Checkpoint of the best model : $BEST_MODEL"
