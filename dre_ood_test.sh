#!/bin/bash
# OS_DRE OOD Detection test script (test_only mode)
# Usage:
#   bash dre_ood_test.sh [RUN_MODE] [DATA_MODE] [GPU_LIST]
#     RUN_MODE : neural | score | all        # default: all
#     DATA_MODE: cifar10 | cifar100 | both   # default: cifar10
#     GPU_LIST : space-separated GPU id list # default: "0"
#
# Examples:
#   1) Run CIFAR-10 score methods (three score methods) on GPU 0 in test_only mode:
#        bash dre_ood_test.sh score cifar10 "0"
#      Or with nohup (default style without extra shell logs):
#        nohup bash dre_ood_test.sh score cifar10 "0" > /dev/null 2>&1 &
#   2) Run CIFAR-10/100 neural + score methods sequentially on GPU 3:
#        bash dre_ood_test.sh all both "3"
#      Or with nohup (no extra .log from the shell script itself):
#        nohup bash dre_ood_test.sh all both "3" > /dev/null 2>&1 &
#
# Notes:
#   - This script only performs test_only evaluation (no training).
#   - All experiments run sequentially on a single GPU (the first GPU in GPU_LIST),
#     which is important for fair timing comparison.
#   - Each python process writes its stdout/stderr to an individual .log file under
#     results/fdiv_estimation/ood_detection/test_logs/, and these logs are ignored by git.

# Common configuration (should stay consistent with dre_ood_train.sh)
EPOCHS=1000
LR=0.001
SAMPLE_NOISE_STD=0.005
OOD_TRAIN=random
OOD_USE_IMGLIST=True
IMAGE_BACKBONE_NORM=groupnorm

# Batch sizes used in test_only mode (for dataloader configuration).
# For fair timing comparison, we enforce the SAME batch size across methods
# within each dataset (neural vs score share the same batch size).
# You can change these numbers, but keep them identical across methods for
# a given dataset.
BATCH_SIZE_CIFAR10=512
BATCH_SIZE_CIFAR100=768

# Neural-specific: backbone=resnet18
NEURAL_IMAGE_BACKBONE=resnet18

# Score-specific: backbone=unet (consistent with training)
SCORE_IMAGE_BACKBONE=unet
PATH_TYPE=trigonometric
T_MODE=lognorm
CONDITION=True
JOINT=True
BRIDGE=1

# Random seeds (same as training)
SEEDS=(1 2 3407)

# Method lists (must stay consistent with dre_ood_train.sh)
NEURAL_METHODS=(nce pw chisq infonce logistic)
SCORE_METHODS=(dre_infty d3re osdre)

# First argument: neural | score | all
RUN_MODE="${1:-all}"

# Second argument: dataset mode cifar10 | cifar100 | both
DATA_MODE="${2:-cifar10}"
declare -a DATASETS=()
case "$DATA_MODE" in
    cifar10)
        DATASETS=(cifar10)
        ;;
    cifar100)
        DATASETS=(cifar100)
        ;;
    both)
        DATASETS=(cifar10 cifar100)
        ;;
    *)
        echo "Usage: $0 [neural|score|all] [cifar10|cifar100|both] [\"GPU_LIST\"]"
        exit 1
        ;;
esac

# Third argument: GPU_LIST, only the first GPU will be used
DEFAULT_GPU_LIST="0"
GPU_LIST_INPUT="${3:-$DEFAULT_GPU_LIST}"
read -r -a gpus <<< "$GPU_LIST_INPUT"
GPU_ID="${gpus[0]}"

# Build experiment list: "dataset,sub_method,subsub_method,seed"
declare -a experiments=()
case "$RUN_MODE" in
    neural)
        for dataset in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for m in "${NEURAL_METHODS[@]}"; do
                    experiments+=("${dataset},neural,${m},${seed}")
                done
            done
        done
        ;;
    score)
        for dataset in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for m in "${SCORE_METHODS[@]}"; do
                    experiments+=("${dataset},score,${m},${seed}")
                done
            done
        done
        ;;
    all)
        for dataset in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for m in "${NEURAL_METHODS[@]}"; do
                    experiments+=("${dataset},neural,${m},${seed}")
                done
                for m in "${SCORE_METHODS[@]}"; do
                    experiments+=("${dataset},score,${m},${seed}")
                done
            done
        done
        ;;
    *)
        echo "Usage: $0 [neural|score|all] [cifar10|cifar100|both] [\"GPU_LIST\"]"
        exit 1
        ;;
esac

LOG_ROOT="results/fdiv_estimation/ood_detection/test_logs"
mkdir -p "$LOG_ROOT"
# Single aggregated log file for all python runs in this script invocation.
# Bash-level messages go to stdout/stderr; only python outputs are appended here.
RUN_LOG_FILE="${LOG_ROOT}/test_${DATA_MODE}_${RUN_MODE}_gpu${GPU_ID}.log"

echo "============================================"
echo "OS_DRE OOD Detection TEST_ONLY — RUN_MODE=$RUN_MODE"
echo "============================================"
echo "Seeds: ${SEEDS[*]}"
echo "Data mode: $DATA_MODE, datasets: ${DATASETS[*]}"
echo "Neural methods (backbone=$NEURAL_IMAGE_BACKBONE): ${NEURAL_METHODS[*]}"
echo "Score methods (backbone=$SCORE_IMAGE_BACKBONE): ${SCORE_METHODS[*]}"
echo "Total experiments: ${#experiments[@]}"
echo "Using single GPU: $GPU_ID"
echo "Python outputs will be appended to: $RUN_LOG_FILE"
echo "============================================"

total=${#experiments[@]}
idx=0

for exp in "${experiments[@]}"; do
    IFS=',' read -r dataset sub_method subsub_method seed <<< "$exp"
    idx=$((idx + 1))

    # Choose batch_size based only on dataset (same across methods for fairness)
    if [ "$dataset" = "cifar100" ]; then
        bs=$BATCH_SIZE_CIFAR100
    else
        bs=$BATCH_SIZE_CIFAR10
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ($idx/$total) Running: data=$dataset, sub_method=$sub_method, subsub_method=$subsub_method, seed=$seed, gpu=$GPU_ID"

    if [ "$sub_method" = "neural" ]; then
        python main.py fdiv_estimation \
            --subtask ood_detection \
            --sub_method neural \
            --subsub_method "$subsub_method" \
            --sample_noise_std $SAMPLE_NOISE_STD \
            --ood_use_imglist $OOD_USE_IMGLIST \
            --ood_in_dist "$dataset" \
            --ood_train $OOD_TRAIN \
            --image_backbone $NEURAL_IMAGE_BACKBONE \
            --image_backbone_norm $IMAGE_BACKBONE_NORM \
            --epochs $EPOCHS \
            --batch_size $bs \
            --lr $LR \
            --gpu_id $GPU_ID \
            --seed $seed \
            --test_only \
            >> "$RUN_LOG_FILE" 2>&1
    else
        python main.py fdiv_estimation \
            --subtask ood_detection \
            --sub_method score \
            --subsub_method "$subsub_method" \
            --condition $CONDITION \
            --path_type $PATH_TYPE \
            --t_mode $T_MODE \
            --joint $JOINT \
            --bridge \
            --sample_noise_std $SAMPLE_NOISE_STD \
            --ood_use_imglist $OOD_USE_IMGLIST \
            --ood_in_dist "$dataset" \
            --ood_train $OOD_TRAIN \
            --image_backbone $SCORE_IMAGE_BACKBONE \
            --image_backbone_norm $IMAGE_BACKBONE_NORM \
            --epochs $EPOCHS \
            --batch_size $bs \
            --lr $LR \
            --gpu_id $GPU_ID \
            --seed $seed \
            --test_only \
            >> "$RUN_LOG_FILE" 2>&1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: data=$dataset, sub_method=$sub_method, subsub_method=$subsub_method, seed=$seed"
    echo ""
done

echo "============================================"
echo "All $total test_only experiments finished. Logs in: $LOG_ROOT"
echo "============================================"

