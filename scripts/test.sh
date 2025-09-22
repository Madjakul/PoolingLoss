#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

CHECKPOINT_PATH="/path/to/your/logs/your-run-name/checkpoints/epoch=0.ckpt"
CONFIG_PATH=$PROJECT_ROOT/configs/test.yml
LOGS_DIR=$PROJECT_ROOT/logs/test

# --------------------------------------------------------------------------------------

# Path to the pre-processed dataset (should be the same one used for training)
PROCESSED_DS_PATH=/scratch/$USER/Datasets/msmarco/facebookai-roberta-base/padding
# CACHE_DIR=$PROJECT_ROOT/../cache/
NUM_PROC=16

# **************************************************************************************

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at '$CHECKPOINT_PATH'"
    echo "Please update the CHECKPOINT_PATH variable in scripts/test.sh"
    exit 1
fi

mkdir -p "$LOGS_DIR" || true
mkdir -p "$PROJECT_ROOT/tmp/" || true

cmd=(python3 "$PROJECT_ROOT/test.py"
    --config_path "$CONFIG_PATH"
    --logs_dir "$LOGS_DIR"
    --processed_ds_path "$PROCESSED_DS_PATH"
    --checkpoint_path "$CHECKPOINT_PATH")

if [[ -v CACHE_DIR ]]; then
    mkdir -p "$CACHE_DIR" || true
    cmd+=(--cache_dir "$CACHE_DIR")
fi

if [[ -v NUM_PROC ]]; then
    cmd+=(--num_proc "$NUM_PROC")
fi

"${cmd[@]}"
