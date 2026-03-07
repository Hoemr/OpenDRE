#!/bin/bash
# OS_DRE OOD Detection Experiments - Parallel Run
# Neural: nce, pw, chisq, gamma  |  Score: osdre, osdre+bridge
# One process per GPU. GPUs 2-9.
# RUN: nohup bash dre_ood_train.sh > dre_ood_train.log 2>&1 &

# Configuration
EPOCHS=1000
BATCH_SIZE=512
LR=0.001
SAMPLE_NOISE_STD=0.005
CHECK_INTERVAL=20
LAUNCH_WAIT=60

# GPUs 2-9, one task per GPU
gpus=(2 3 4 5 6 7 8 9)
MAX_TASKS_PER_GPU=1

# Neural subsub_methods: nce, pw, chisq, gamma
NEURAL_METHODS=(nce pw chisq gamma)

# Score: osdre (no bridge), osdre_bridge (with --bridge)
SCORE_METHODS=(osdre osdre_bridge)

# Build experiment list: "sub_method,subsub_method" or "score,osdre_bridge" for bridge
declare -a experiments=()
for m in "${NEURAL_METHODS[@]}"; do
    experiments+=("neural,$m")
done
for m in "${SCORE_METHODS[@]}"; do
    experiments+=("score,$m")
done

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

run_experiment() {
    local sub_method=$1
    local subsub_method=$2
    local gpu_id=$3

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: sub_method=$sub_method, subsub_method=$subsub_method, gpu_id=$gpu_id"

    if [ "$sub_method" = "score" ]; then
        local bridge_flag=""
        [ "$subsub_method" = "osdre_bridge" ] && bridge_flag="--bridge"
        local actual_method="osdre"
        nohup python main.py fdiv_estimation \
            --subtask ood_detection \
            --sub_method score \
            --subsub_method "$actual_method" \
            --condition True \
            --path_type VP \
            --t_mode lognorm \
            --joint False \
            $bridge_flag \
            --sample_noise_std $SAMPLE_NOISE_STD \
            --ood_use_imglist True \
            --ood_in_dist cifar10 \
            --ood_train selfgen \
            --image_backbone resnet18 \
            --image_backbone_norm groupnorm \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --gpu_id $gpu_id \
            > /dev/null 2>&1 &
    else
        nohup python main.py fdiv_estimation \
            --subtask ood_detection \
            --sub_method neural \
            --subsub_method "$subsub_method" \
            --sample_noise_std $SAMPLE_NOISE_STD \
            --ood_use_imglist True \
            --ood_in_dist cifar10 \
            --ood_train selfgen \
            --image_backbone resnet18 \
            --image_backbone_norm groupnorm \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --gpu_id $gpu_id \
            > /dev/null 2>&1 &
    fi

    local pid=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched: sub_method=$sub_method, subsub_method=$subsub_method, gpu_id=$gpu_id, pid=$pid"
}

# Main
echo "============================================"
echo "OS_DRE OOD Detection - Neural (nce,pw,chisq,gamma) + Score (osdre, osdre+bridge)"
echo "============================================"
echo "Neural methods: ${NEURAL_METHODS[*]}"
echo "Score methods: ${SCORE_METHODS[*]}"
echo "Config: cifar10, selfgen, resnet18, groupnorm, epochs=$EPOCHS, batch=$BATCH_SIZE, lr=$LR"
echo "Total experiments: ${#experiments[@]}"
echo "GPUs: ${gpus[*]}"
echo "============================================"

total=${#experiments[@]}
completed=0

for exp in "${experiments[@]}"; do
    IFS=',' read -r sub_method subsub_method <<< "$exp"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for available GPU..."
    while true; do
        available_gpu=$(find_available_gpu)
        [ -n "$available_gpu" ] && break
        sleep $CHECK_INTERVAL
    done

    run_experiment "$sub_method" "$subsub_method" "$available_gpu"
    completed=$((completed + 1))
    echo "Progress: $completed/$total launched. Waiting ${LAUNCH_WAIT}s..."
    sleep $LAUNCH_WAIT
    echo ""
done

echo "============================================"
echo "All $total experiments launched. Monitor: nvidia-smi"
