#!/bin/bash
# OS_DRE OOD Detection: 通过第一个参数选择跑 neural / score / all
# 用法: bash dre_ood_train.sh [neural|score|all]   # 可选后续参数暂不用于覆盖 seeds
# Neural: image_backbone=resnet18.  Score: image_backbone=unet, path_type=trigonometric, joint, bridge 等与标准一致.
# 后台不写 log: nohup bash dre_ood_train.sh neural > /dev/null 2>&1 &

# 公共配置
EPOCHS=1000
BATCH_SIZE=512
LR=0.001
SAMPLE_NOISE_STD=0.005
OOD_IN_DIST=cifar10
OOD_TRAIN=random
OOD_USE_IMGLIST=True
IMAGE_BACKBONE_NORM=groupnorm
CHECK_INTERVAL=20
LAUNCH_WAIT=60

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

# GPUs
gpus=(1 2)
MAX_TASKS_PER_GPU=2

# 方法列表
NEURAL_METHODS=(nce pw chisq gamma infonce logistic)
SCORE_METHODS=(osdre osdre_bridge)

# 第一个参数: neural | score | all
RUN_MODE="${1:-all}"

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
    local seed=$1
    local subsub_method=$2
    local gpu_id=$3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting neural: seed=$seed, subsub_method=$subsub_method, gpu_id=$gpu_id"
    nohup python main.py fdiv_estimation \
        --subtask ood_detection \
        --sub_method neural \
        --subsub_method "$subsub_method" \
        --sample_noise_std $SAMPLE_NOISE_STD \
        --ood_use_imglist $OOD_USE_IMGLIST \
        --ood_in_dist $OOD_IN_DIST \
        --ood_train $OOD_TRAIN \
        --image_backbone $NEURAL_IMAGE_BACKBONE \
        --image_backbone_norm $IMAGE_BACKBONE_NORM \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --gpu_id $gpu_id \
        --seed $seed \
        > /dev/null 2>&1 &
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched neural: seed=$seed, $subsub_method, gpu=$gpu_id, pid=$!"
}

run_score() {
    local seed=$1
    local subsub_method=$2
    local gpu_id=$3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting score: seed=$seed, subsub_method=$subsub_method, gpu_id=$gpu_id"
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
        --ood_in_dist $OOD_IN_DIST \
        --ood_train $OOD_TRAIN \
        --image_backbone $SCORE_IMAGE_BACKBONE \
        --image_backbone_norm $IMAGE_BACKBONE_NORM \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --gpu_id $gpu_id \
        --seed $seed \
        > /dev/null 2>&1 &
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched score: seed=$seed, $subsub_method, gpu=$gpu_id, pid=$!"
}

# 根据 RUN_MODE 构建实验列表: "seed,sub_method,subsub_method"
declare -a experiments=()
case "$RUN_MODE" in
    neural)
        for seed in "${SEEDS[@]}"; do
            for m in "${NEURAL_METHODS[@]}"; do
                experiments+=("${seed},neural,${m}")
            done
        done
        ;;
    score)
        for seed in "${SEEDS[@]}"; do
            for m in "${SCORE_METHODS[@]}"; do
                experiments+=("${seed},score,${m}")
            done
        done
        ;;
    all)
        for seed in "${SEEDS[@]}"; do
            for m in "${NEURAL_METHODS[@]}"; do
                experiments+=("${seed},neural,${m}")
            done
            for m in "${SCORE_METHODS[@]}"; do
                experiments+=("${seed},score,${m}")
            done
        done
        ;;
    *)
        echo "Usage: $0 [neural|score|all]"
        echo "  neural: 只跑 neural density ratio (backbone=resnet18), seeds=${SEEDS[*]}"
        echo "  score:  只跑 score methods (backbone=unet), seeds=${SEEDS[*]}"
        echo "  all:    跑 neural + score (默认)"
        exit 1
        ;;
esac

# Main
echo "============================================"
echo "OS_DRE OOD Detection — RUN_MODE=$RUN_MODE"
echo "============================================"
echo "Seeds: ${SEEDS[*]}"
echo "Neural methods (backbone=$NEURAL_IMAGE_BACKBONE): ${NEURAL_METHODS[*]}"
echo "Score methods (backbone=$SCORE_IMAGE_BACKBONE): ${SCORE_METHODS[*]}"
echo "Total experiments: ${#experiments[@]}"
echo "GPUs: ${gpus[*]}"
echo "============================================"

total=${#experiments[@]}
completed=0

for exp in "${experiments[@]}"; do
    IFS=',' read -r seed sub_method subsub_method <<< "$exp"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for available GPU..."
    while true; do
        available_gpu=$(find_available_gpu)
        [ -n "$available_gpu" ] && break
        sleep $CHECK_INTERVAL
    done

    if [ "$sub_method" = "neural" ]; then
        run_neural "$seed" "$subsub_method" "$available_gpu"
    else
        run_score "$seed" "$subsub_method" "$available_gpu"
    fi

    completed=$((completed + 1))
    echo "Progress: $completed/$total launched. Waiting ${LAUNCH_WAIT}s..."
    sleep $LAUNCH_WAIT
    echo ""
done

echo "============================================"
echo "All $total experiments launched. Monitor: nvidia-smi"
