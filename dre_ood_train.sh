#!/bin/bash
# OS_DRE OOD Detection Experiments - Parallel Run Script
# Supports configurable DRE methods and in-distribution datasets
# Automatically iterates over all combinations of DRE methods, in-dist datasets, and base types

# Configuration
EPOCHS=1000
BATCH_SIZE=512
LR=0.01
NOISE_STD=0.1
ANNEAL_ALPHA_INIT=0.142857
ANNEAL_GAMMA=1.0
MAX_TASKS_PER_GPU=1  # Each GPU runs one task at a time
CHECK_INTERVAL=20  # Check for available GPU every 20 seconds
LAUNCH_WAIT=30  # Wait 30 seconds after launching before checking for next GPU

# Optional GPU list (GPUs 0-9)
gpus=(2 3 4 5 6 7 8 9)

# DRE methods to run (can be multiple, space-separated)
# Options: logistic, nce, infonce, rkl, hellinger, kl, gamma, ew, pw, kulsif
# Note: These correspond to subsub_method in OS_DRE (neural sub_method)
DRE_METHODS="${1:-logistic nce}"
# Example: DRE_METHODS="logistic nce rkl"

# In-distribution datasets to run
# Options: mnist, fashion_mnist, cifar10, cifar100
IN_DIST_DATASETS="${2:-mnist fashion_mnist}"
# Example: IN_DIST_DATASETS="mnist fashion_mnist cifar10"

# Base types to run (local, general, universal)
BASE_TYPES="${3:-universal}"
# Example: BASE_TYPES="universal" or BASE_TYPES="local general universal"

# Image backbone
# Options: resnet18, resnet34, resnet50, wideresnet28_10
# Note: resnet18 for mnist/fashion_mnist, wideresnet28_10 for cifar10/cifar100
IMAGE_BACKBONE="${4:-resnet18}"
# Example: IMAGE_BACKBONE="resnet18 wideresnet28_10"

# Parse DRE methods into array (handle both quoted and unquoted input)
if [[ "$DRE_METHODS" == *" "* ]]; then
    IFS=' ' read -ra DRE_METHODS_ARRAY <<< "$DRE_METHODS"
else
    DRE_METHODS_ARRAY=("$DRE_METHODS")
fi

# Parse in-dist datasets into array
if [[ "$IN_DIST_DATASETS" == *" "* ]]; then
    IFS=' ' read -ra IN_DIST_ARRAY <<< "$IN_DIST_DATASETS"
else
    IN_DIST_ARRAY=("$IN_DIST_DATASETS")
fi

# Parse base types into array
if [[ "$BASE_TYPES" == *" "* ]]; then
    IFS=' ' read -ra BASE_TYPES_ARRAY <<< "$BASE_TYPES"
else
    BASE_TYPES_ARRAY=("$BASE_TYPES")
fi

# Parse image backbone into array
if [[ "$IMAGE_BACKBONE" == *" "* ]]; then
    IFS=' ' read -ra IMAGE_BACKBONE_ARRAY <<< "$IMAGE_BACKBONE"
else
    IMAGE_BACKBONE_ARRAY=("$IMAGE_BACKBONE")
fi

# Generate all experiment combinations
declare -a experiments=()

for dre_method in "${DRE_METHODS_ARRAY[@]}"; do
    for in_dist in "${IN_DIST_ARRAY[@]}"; do
        for base_type in "${BASE_TYPES_ARRAY[@]}"; do
            for backbone in "${IMAGE_BACKBONE_ARRAY[@]}"; do
                experiments+=("$in_dist,$base_type,$dre_method,$backbone")
            done
        done
    done
done

# Track running experiments
declare -A running_experiments

# Function to get number of running tasks on a GPU
get_running_tasks() {
    local gpu_id=$1
    nvidia-smi --id="$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l
}

# Function to find an available GPU
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

# Function to run a single experiment
run_experiment() {
    local in_dist=$1
    local base_type=$2
    local dre_method=$3
    local backbone=$4
    local gpu_id=$5
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: in_dist=$in_dist, base_type=$base_type, dre_method=$dre_method, backbone=$backbone, gpu_id=$gpu_id"
    
    # Run with nohup - output goes to the log file created by the Python script
    nohup conda run -n osdre python main.py fdiv_estimation \
        --subtask ood_detection \
        --ood_in_dist "$in_dist" \
        --ood_base_type "$base_type" \
        --ood_noise_std $NOISE_STD \
        --ood_anneal_alpha_init $ANNEAL_ALPHA_INIT \
        --ood_anneal_gamma $ANNEAL_GAMMA \
        --sub_method neural \
        --subsub_method "$dre_method" \
        --image_backbone "$backbone" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --gpu_id $gpu_id \
        > /dev/null 2>&1 &
    
    local pid=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched: in_dist=$in_dist, base_type=$base_type, dre_method=$dre_method, backbone=$backbone, gpu_id=$gpu_id, pid=$pid"
}

# Main execution
echo "============================================"
echo "OS_DRE OOD Detection Experiments - Parallel Run"
echo "============================================"
echo "DRE Methods (subsub_method): ${DRE_METHODS}"
echo "In-distribution Datasets: ${IN_DIST_DATASETS}"
echo "Base Types: ${BASE_TYPES}"
echo "Image Backbone: ${IMAGE_BACKBONE}"
echo "Total experiments: ${#experiments[@]}"
echo "Available GPUs: ${gpus[@]}"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "============================================"
echo ""
echo "Usage: ./dre_ood_train.sh [dre_methods] [in_dist_datasets] [base_types] [image_backbone]"
echo "Example: ./dre_ood_train.sh 'logistic nce rkl' 'mnist fashion_mnist' 'universal' 'resnet18'"
echo "Example: ./dre_ood_train.sh logistic mnist fashion_mnist universal resnet18"
echo "Available DRE methods: logistic, nce, infonce, rkl, hellinger, kl, gamma, ew, pw, kulsif"
echo "Available datasets: mnist, fashion_mnist, cifar10, cifar100"
echo "Available base types: local, general, universal"
echo "Available backbones: resnet18, resnet34, resnet50, wideresnet28_10"
echo ""

# Calculate total experiments
total_experiments=${#experiments[@]}
completed=0

# Process each experiment
for exp in "${experiments[@]}"; do
    IFS=',' read -r in_dist base_type dre_method backbone <<< "$exp"
    
    # Wait for an available GPU
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for available GPU..."
    while true; do
        available_gpu=$(find_available_gpu)
        if [ -n "$available_gpu" ]; then
            break
        fi
        sleep $CHECK_INTERVAL
    done
    
    # Run the experiment on the available GPU
    run_experiment "$in_dist" "$base_type" "$dre_method" "$backbone" "$available_gpu"
    
    completed=$((completed + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Progress: $completed/$total_experiments experiments launched"
    echo "Waiting ${LAUNCH_WAIT}s for the process to initialize..."
    sleep $LAUNCH_WAIT
    echo ""
done

echo "============================================"
echo "All $total_experiments experiments launched!"
echo "Use 'nvidia-smi' to monitor GPU usage"
echo "Logs are saved in ./results/ directory"
echo "============================================"
