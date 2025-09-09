#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=$PROJECT_ROOT/configs/train.yml
LOGS_DIR=$PROJECT_ROOT/logs
PROCESSED_DS_PATH=/scratch/$USER/Datasets/allnli/facebookai-roberta-base/padding # /scratch/$USER/Datasets/pooling-loss-v2/facebookai-roberta-base/padding

# --------------------------------------------------------------------------------------

# INDIVIDUAL_PROCESSED_PATHS=(
# "/scratch/$USER/Datasets/allnli/facebookai-roberta-base/padding"
# "/scratch/$USER/Datasets/msmarco/facebookai-roberta-base/padding"
# "/scratch/$USER/Datasets/booksum/facebookai-roberta-base/padding"
# )
# CACHE_DIR=$PROJECT_ROOT/../cache/
NUM_PROC=48
#

# **************************************************************************************

mkdir -p "$LOGS_DIR" || true
mkdir -p "$PROJECT_ROOT/tmp/" || true

if [[ $SLURM_JOB_ID != "" ]]; then
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "SLURM_NNODES: $SLURM_NNODES"
    echo "SLURM_NTASKS: $SLURM_NTASKS"
    echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
    echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    torchrun \
        --standalone \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$SLURM_GPUS_ON_NODE \
        "$PROJECT_ROOT/train.py" \
        --config_path "$CONFIG_PATH" \
        --logs_dir "$LOGS_DIR" \
        --processed_ds_path "$PROCESSED_DS_PATH" \
        ${INDIVIDUAL_PROCESSED_PATHS:+--individual_processed_paths "${INDIVIDUAL_PROCESSED_PATHS[@]}"} \
        ${CACHE_DIR:+--cache_dir "$CACHE_DIR"} \
        ${CHECKPOINT_DIR:+--checkpoint_dir "$CHECKPOINT_DIR"} \
        ${NUM_PROC:+--num_proc "$NUM_PROC"}
else
    cmd=()
    cmd=(python3 "$PROJECT_ROOT/train.py"
        --config_path "$CONFIG_PATH"
        --logs_dir "$LOGS_DIR"
        --processed_ds_path "$PROCESSED_DS_PATH")

    if [[ -v INDIVIDUAL_PROCESSED_PATHS ]]; then
        cmd+=(--individual_processed_paths "${INDIVIDUAL_PROCESSED_PATHS[@]}")
    fi

    if [[ -v CACHE_DIR ]]; then
        mkdir -p "$CACHE_DIR" || true
        cmd+=(--cache_dir "$CACHE_DIR")
    fi

    if [[ -v NUM_PROC ]]; then
        cmd+=(--num_proc "$NUM_PROC")
    fi

    "${cmd[@]}"
fi
