#!/bin/bash
# OS_DRE OOD Detection 启动脚本
# 用法:
#   bash dre_ood_train.sh [RUN_MODE] [DATA_MODE] [GPU_LIST]
#     RUN_MODE : neural | score | all        # 默认 all
#     DATA_MODE: cifar10 | cifar100 | both   # 默认 cifar10
#     GPU_LIST : 以空格分隔的 GPU id 列表   # 默认 "0 1 2 3"
#
# 示例:
#   1) 只跑 CIFAR-10 的 score（三种方法），用 GPU 0,1:
#        bash dre_ood_train.sh score cifar10 "0 1"
#   2) 跑 CIFAR-100 的 neural（batch=1024），用 GPU 2,3,4,5:
#        bash dre_ood_train.sh neural cifar100 "2 3 4 5"
#   3) 后台不写 log 跑 CIFAR-100 的 score（batch=768），用 GPU 0,1,2,3:
#        nohup bash dre_ood_train.sh score cifar100 "0 1 2 3" > /dev/null 2>&1 &
#
# Neural: image_backbone=resnet18.
# Score : image_backbone=unet, path_type=trigonometric, joint, bridge 等与标准一致.

# 公共配置
EPOCHS=1000
LR=0.001
SAMPLE_NOISE_STD=0.005
OOD_IN_DIST=cifar10   # 默认数据集（单数据集模式下使用）
OOD_TRAIN=random
OOD_USE_IMGLIST=True
IMAGE_BACKBONE_NORM=groupnorm
CHECK_INTERVAL=20
LAUNCH_WAIT=60

# 不同方法/数据集的 batch_size（默认先按 CIFAR-10 配置，后面再根据 DATA_MODE 调整）
BATCH_SIZE_NEURAL=512
BATCH_SIZE_SCORE=512

# Neural 专用: backbone=resnet18
NEURAL_IMAGE_BACKBONE=resnet18

# Score 专用: backbone=unet，与标准命令一致
SCORE_IMAGE_BACKBONE=unet
PATH_TYPE=trigonometric
T_MODE=lognorm
CONDITION=True
JOINT=True
BRIDGE=1

# 消融种子（neural/score 都可用）
SEEDS=(1 2 3407)

# GPUs（默认 0 1 2 3，可以通过第三个参数覆盖）
DEFAULT_GPU_LIST="0 1 2 3"
GPU_LIST_INPUT="${3:-$DEFAULT_GPU_LIST}"
read -r -a gpus <<< "$GPU_LIST_INPUT"
MAX_TASKS_PER_GPU=1   # 具体值稍后根据 RUN_MODE / DATA_MODE 调整

# 方法列表
NEURAL_METHODS=(nce pw chisq infonce logistic)
SCORE_METHODS=(dre_infty d3re osdre)

# 第一个参数: neural | score | all
RUN_MODE="${1:-all}"

# 第二个参数: 数据集模式 cifar10 | cifar100 | both
DATA_MODE="${2:-cifar10}"
declare -a DATASETS=()
case "$DATA_MODE" in
    cifar10)
        DATASETS=(cifar10)
        # CIFAR-10：neural/score 都用 512
        BATCH_SIZE_NEURAL=512
        BATCH_SIZE_SCORE=512
        ;;
    cifar100)
        DATASETS=(cifar100)
        # CIFAR-100：你指定的配置
        BATCH_SIZE_NEURAL=1024   # neural
        BATCH_SIZE_SCORE=768     # score
        ;;
    both)
        DATASETS=(cifar10 cifar100)
        # 混合跑时，在 run_* 里再根据 dataset 细分
        BATCH_SIZE_NEURAL=512
        BATCH_SIZE_SCORE=512
        ;;
    *)
        echo "Usage: $0 [neural|score|all] [cifar10|cifar100|both] [\"GPU_LIST\"]"
        echo "  第一个参数: 选择方法类型 (默认 all)"
        echo "  第二个参数: 选择数据集 (默认 cifar10)"
        echo "  第三个参数: GPU 列表，例如 \"0 1 2 3\"（默认 \"${DEFAULT_GPU_LIST}\")"
        exit 1
        ;;
esac

# 根据 RUN_MODE / DATA_MODE 设置每卡并发数：
# - CIFAR-100 + score：一张卡跑一个方法
# - CIFAR-100 + neural：一张卡跑两个方法
case "$RUN_MODE" in
    neural)
        if [ "$DATA_MODE" = "cifar100" ]; then
            MAX_TASKS_PER_GPU=2
        else
            MAX_TASKS_PER_GPU=2
        fi
        ;;
    score)
        if [ "$DATA_MODE" = "cifar100" ]; then
            MAX_TASKS_PER_GPU=1
        else
            MAX_TASKS_PER_GPU=2
        fi
        ;;
    all)
        # all 情况下两类方法混合，为简单起见仍设为 2
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

run_neural() {
    local dataset=$1
    local seed=$2
    local subsub_method=$3
    local gpu_id=$4
    # 根据数据集选择 batch_size（特别是 CIFAR-100）
    local bs=$BATCH_SIZE_NEURAL
    if [ "$dataset" = "cifar100" ]; then
        bs=1024
    fi
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
}

run_score() {
    local dataset=$1
    local seed=$2
    local subsub_method=$3
    local gpu_id=$4
    # 根据数据集选择 batch_size（特别是 CIFAR-100）
    local bs=$BATCH_SIZE_SCORE
    if [ "$dataset" = "cifar100" ]; then
        bs=768
    fi
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
}

# 根据 RUN_MODE 和 DATA_MODE 构建实验列表: "dataset,sub_method,subsub_method,seed"
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
        echo "  neural: 只跑 neural density ratio (backbone=resnet18), seeds=${SEEDS[*]}"
        echo "  score:  只跑 score methods (backbone=unet), seeds=${SEEDS[*]}"
        echo "  all:    跑 neural + score (默认)"
        echo "  第二个参数选择数据集: cifar10 / cifar100 / both"
        echo "  第三个参数选择 GPU: 例如 \"0 1 2 3\"（默认 \"${DEFAULT_GPU_LIST}\")"
        exit 1
        ;;
esac

# Main
echo "============================================"
echo "OS_DRE OOD Detection — RUN_MODE=$RUN_MODE"
echo "============================================"
echo "Seeds: ${SEEDS[*]}"
echo "Data mode: $DATA_MODE, datasets: ${DATASETS[*]}"
echo "Batch size (neural): $BATCH_SIZE_NEURAL (CIFAR-100 override: 1024)"
echo "Batch size (score):  $BATCH_SIZE_SCORE (CIFAR-100 override: 768)"
echo "Neural methods (backbone=$NEURAL_IMAGE_BACKBONE): ${NEURAL_METHODS[*]}"
echo "Score methods (backbone=$SCORE_IMAGE_BACKBONE): ${SCORE_METHODS[*]}"
echo "Total experiments: ${#experiments[@]}"
echo "GPUs: ${gpus[*]} (max $MAX_TASKS_PER_GPU per GPU)"
echo "============================================"

total=${#experiments[@]}
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
        run_neural "$dataset" "$seed" "$subsub_method" "$available_gpu"
    else
        run_score "$dataset" "$seed" "$subsub_method" "$available_gpu"
    fi

    completed=$((completed + 1))
    echo "Progress: $completed/$total launched. Waiting ${LAUNCH_WAIT}s..."
    sleep $LAUNCH_WAIT
    echo ""
done

echo "============================================"
echo "All $total experiments launched. Monitor: nvidia-smi"
