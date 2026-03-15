#!/bin/bash
# OS_DRE OOD Detection unified script
# Usage:
#   bash dre_ood.sh [MODE] [RUN_MODE] [DATA_MODE] [GPU_LIST] [SEEDS]
#     MODE     : train | test               # default: test
#     RUN_MODE : neural | score | all       # default: all
#     DATA_MODE: cifar10 | cifar100 | both  # default: cifar10
#     GPU_LIST : space-separated GPU id list # default: "0" (test) or "0 1 2 3" (train)
#     SEEDS    : space-separated seed list   # default: "1 2 3407"
#
# Examples:
#   1) Test mode (default) - run CIFAR-10 score methods on GPU 0:
#        bash dre_ood.sh test score cifar10 "0"
#   2) Test with single seed (seed=1 only):
#        bash dre_ood.sh test score cifar10 "0" "1"
#   3) Train with specific seeds (seeds=1 and 2):
#        bash dre_ood.sh train neural cifar100 "2 3 4 5" "1 2"
#   4) Train all methods on multiple GPUs with all seeds:
#        bash dre_ood.sh train all both "0 1 2 3"
#
# Notes:
#   - train mode: parallel execution with GPU scheduling (multi-GPU support)
#   - test mode: sequential execution on single GPU (for fair timing comparison)
#   - test_only flag is automatically added in test mode

# ============================================
# First argument: MODE (train | test), default: test
# ============================================
MODE="${1:-test}"
if [ "$MODE" != "train" ] && [ "$MODE" != "test" ]; then
    echo "Error: MODE must be 'train' or 'test', got '$MODE'"
    echo "Usage: $0 [train|test] [neural|score|all] [cifar10|cifar100|both] [\"GPU_LIST\"] [\"SEEDS\"]"
    exit 1
fi

# ============================================
# Common configuration (consistent across modes)
# ============================================
EPOCHS=1000
LR=0.001
SAMPLE_NOISE_STD=0.005
OOD_TRAIN=random
OOD_USE_IMGLIST=True
IMAGE_BACKBONE_NORM=groupnorm

# Neural-specific: backbone=resnet18
NEURAL_IMAGE_BACKBONE=resnet18

# Score-specific: backbone=unet
SCORE_IMAGE_BACKBONE=unet
PATH_TYPE=trigonometric
T_MODE=lognorm
CONDITION=True
JOINT=True
BRIDGE=1

# Method lists
NEURAL_METHODS=(nce pw chisq infonce logistic)
SCORE_METHODS=(dre_infty d3re osdre)

# ============================================
# Mode-specific configuration
# ============================================
# Batch sizes (CIFAR-10 aligns with CIFAR-100)
BATCH_SIZE_CIFAR10=768
BATCH_SIZE_CIFAR100=768

# GPU settings
if [ "$MODE" = "train" ]; then
    DEFAULT_GPU_LIST="0 1 2 3"
    CHECK_INTERVAL=20
    LAUNCH_WAIT=60
else
    DEFAULT_GPU_LIST="0"
fi

# Train-specific: max tasks per GPU
MAX_TASKS_PER_GPU=2

# ============================================
# Parse remaining arguments
# ============================================
# Second argument: neural | score | all
RUN_MODE="${2:-all}"

# Third argument: dataset mode cifar10 | cifar100 | both
DATA_MODE="${3:-cifar10}"

# Fourth argument: GPU_LIST
GPU_LIST_INPUT="${4:-$DEFAULT_GPU_LIST}"

# Fifth argument: SEEDS (optional, default: "1 2 3407")
DEFAULT_SEEDS="1 2 3407"
SEEDS_INPUT="${5:-$DEFAULT_SEEDS}"
read -r -a SEEDS <<< "$SEEDS_INPUT"

# Validate RUN_MODE
case "$RUN_MODE" in
    neural|score|all) ;;
    *)
        echo "Error: RUN_MODE must be 'neural', 'score', or 'all', got '$RUN_MODE'"
        echo "Usage: $0 [train|test] [neural|score|all] [cifar10|cifar100|both] [\"GPU_LIST\"] [\"SEEDS\"]"
        exit 1
        ;;
esac

# Parse DATA_MODE
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
        echo "Error: DATA_MODE must be 'cifar10', 'cifar100', or 'both', got '$DATA_MODE'"
        echo "Usage: $0 [train|test] [neural|score|all] [cifar10|cifar100|both] [\"GPU_LIST\"] [\"SEEDS\"]"
        exit 1
        ;;
esac

# Parse GPU list
read -r -a gpus <<< "$GPU_LIST_INPUT"

# ============================================
# Train mode: GPU scheduling functions
# ============================================
if [ "$MODE" = "train" ]; then
    # Adjust MAX_TASKS_PER_GPU based on RUN_MODE
    # Note: score methods use more memory than neural methods
    # - neural: 2 tasks per GPU
    # - score: 1 task per GPU
    case "$RUN_MODE" in
        neural)
            MAX_TASKS_PER_GPU=2
            ;;
        score)
            MAX_TASKS_PER_GPU=1
            ;;
        all)
            MAX_TASKS_PER_GPU=2
            ;;
    esac

    get_running_tasks() {
        local gpu_id=$1
        nvidia-smi --id="$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l
    }

    find_available_gpu() {
        for gpu_id in "${gpus[@]}"; do
            running_tasks=$(get_running_tasks "$gpu_id")
            if [ "$running_tasks" -lt "$MAX_TASKS_PER_GPU" ]; then
                echo "$gpu_id"
                return 0
            fi
        done
        return 1
    }
fi

# ============================================
# Helper functions for running experiments
# ============================================
run_neural() {
    local dataset=$1
    local seed=$2
    local subsub_method=$3
    local gpu_id=$4
    local mode=$5

    # Choose batch_size based on dataset
    if [ "$dataset" = "cifar100" ]; then
        bs=$BATCH_SIZE_CIFAR100
    else
        bs=$BATCH_SIZE_CIFAR10
    fi

    if [ "$mode" = "train" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting neural: data=$dataset, seed=$seed, subsub_method=$subsub_method, batch_size=$bs, gpu_id=$gpu_id"
        nohup python main.py fdiv_estimation \
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
            --gpu_id $gpu_id \
            --seed $seed \
            > /dev/null 2>&1 &
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched neural: data=$dataset, seed=$seed, $subsub_method, gpu=$gpu_id, pid=$!"
    else
        # test mode: sequential execution with logging
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
            --gpu_id $gpu_id \
            --seed $seed \
            --test_only \
            >> "$RUN_LOG_FILE" 2>&1
    fi
}

run_score() {
    local dataset=$1
    local seed=$2
    local subsub_method=$3
    local gpu_id=$4
    local mode=$5

    # Choose batch_size based on dataset
    if [ "$dataset" = "cifar100" ]; then
        bs=$BATCH_SIZE_CIFAR100
    else
        bs=$BATCH_SIZE_CIFAR10
    fi

    if [ "$mode" = "train" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting score: data=$dataset, seed=$seed, subsub_method=$subsub_method, batch_size=$bs, gpu_id=$gpu_id"
        nohup python main.py fdiv_estimation \
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
            --gpu_id $gpu_id \
            --seed $seed \
            > /dev/null 2>&1 &
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched score: data=$dataset, seed=$seed, $subsub_method, gpu=$gpu_id, pid=$!"
    else
        # test mode: sequential execution with logging
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
            --gpu_id $gpu_id \
            --seed $seed \
            --test_only \
            >> "$RUN_LOG_FILE" 2>&1
    fi
}

# ============================================
# Build experiment list
# ============================================
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
esac

# ============================================
# Setup logging (test mode only)
# ============================================
if [ "$MODE" = "test" ]; then
    LOG_ROOT="results/fdiv_estimation/ood_detection/test_logs"
    mkdir -p "$LOG_ROOT"
    GPU_ID="${gpus[0]}"
    SEEDS_STR=$(echo "${SEEDS[@]}" | tr ' ' '_')
    RUN_LOG_FILE="${LOG_ROOT}/test_${DATA_MODE}_${RUN_MODE}_seed${SEEDS_STR}_gpu${GPU_ID}.log"
fi

# ============================================
# Print configuration
# ============================================
echo "============================================"
echo "OS_DRE OOD Detection — MODE=$MODE, RUN_MODE=$RUN_MODE"
echo "============================================"
echo "Seeds: ${SEEDS[*]}"
echo "Data mode: $DATA_MODE, datasets: ${DATASETS[*]}"
echo "Batch sizes: CIFAR-10=$BATCH_SIZE_CIFAR10, CIFAR-100=$BATCH_SIZE_CIFAR100"
echo "Neural methods (backbone=$NEURAL_IMAGE_BACKBONE): ${NEURAL_METHODS[*]}"
echo "Score methods (backbone=$SCORE_IMAGE_BACKBONE): ${SCORE_METHODS[*]}"
echo "Total experiments: ${#experiments[@]}"
if [ "$MODE" = "train" ]; then
    echo "GPUs: ${gpus[*]} (max $MAX_TASKS_PER_GPU per GPU)"
else
    echo "Using single GPU: $GPU_ID (sequential execution for fair timing)"
    echo "Python outputs will be appended to: $RUN_LOG_FILE"
fi
echo "============================================"

# ============================================
# Main execution loop
# ============================================
total=${#experiments[@]}

if [ "$MODE" = "train" ]; then
    # ========== TRAIN MODE: parallel execution with GPU scheduling ==========
    completed=0

    for exp in "${experiments[@]}"; do
        IFS=',' read -r dataset sub_method subsub_method seed <<< "$exp"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for available GPU..."
        while true; do
            available_gpu=$(find_available_gpu)
            [ -n "$available_gpu" ] && break
            sleep $CHECK_INTERVAL
        done

        if [ "$sub_method" = "neural" ]; then
            run_neural "$dataset" "$seed" "$subsub_method" "$available_gpu" "train"
        else
            run_score "$dataset" "$seed" "$subsub_method" "$available_gpu" "train"
        fi

        completed=$((completed + 1))
        echo "Progress: $completed/$total launched. Waiting ${LAUNCH_WAIT}s..."
        sleep $LAUNCH_WAIT
        echo ""
    done

    echo "============================================"
    echo "All $total experiments launched. Monitor: nvidia-smi"
    echo "============================================"

else
    # ========== TEST MODE: sequential execution ==========
    idx=0

    for exp in "${experiments[@]}"; do
        IFS=',' read -r dataset sub_method subsub_method seed <<< "$exp"
        idx=$((idx + 1))

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ($idx/$total) Running: data=$dataset, sub_method=$sub_method, subsub_method=$subsub_method, seed=$seed, gpu=$GPU_ID"

        if [ "$sub_method" = "neural" ]; then
            run_neural "$dataset" "$seed" "$subsub_method" "$GPU_ID" "test"
        else
            run_score "$dataset" "$seed" "$subsub_method" "$GPU_ID" "test"
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: data=$dataset, sub_method=$sub_method, subsub_method=$subsub_method, seed=$seed"
        echo ""
    done

    echo "============================================"
    echo "All $total test_only experiments finished. Logs in: $LOG_ROOT"
    echo "============================================"
fi