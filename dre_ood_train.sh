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

# Neural DRE methods to run (can be multiple, space-separated)
# Options: logistic, nce, infonce, rkl, hellinger, kl, gamma, ew, pw, kulsif
# Note: These correspond to subsub_method in OS_DRE (neural sub_method)
NEURAL_DRE_METHODS="${1:-logistic nce pw}"
# Example: NEURAL_DRE_METHODS="logistic nce rkl"

# Score-based OOD methods (subsub_method)
# Options: dre_infty, d3re
SCORE_METHODS="${5:-dre_infty d3re}"

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

# Which sub_method(s) to launch
# Options: neural, score
SUB_METHODS="${6:-neural score}"

# Parse neural DRE methods into array (handle both quoted and unquoted input)
if [[ "$NEURAL_DRE_METHODS" == *" "* ]]; then
    IFS=' ' read -ra NEURAL_DRE_METHODS_ARRAY <<< "$NEURAL_DRE_METHODS"
else
    NEURAL_DRE_METHODS_ARRAY=("$NEURAL_DRE_METHODS")
fi

# Parse score methods into array
if [[ "$SCORE_METHODS" == *" "* ]]; then
    IFS=' ' read -ra SCORE_METHODS_ARRAY <<< "$SCORE_METHODS"
else
    SCORE_METHODS_ARRAY=("$SCORE_METHODS")
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

# Parse sub_methods into array
if [[ "$SUB_METHODS" == *" "* ]]; then
    IFS=' ' read -ra SUB_METHODS_ARRAY <<< "$SUB_METHODS"
else
    SUB_METHODS_ARRAY=("$SUB_METHODS")
fi

# Generate all experiment combinations
declare -a experiments=()

for sub_method in "${SUB_METHODS_ARRAY[@]}"; do
    case "$sub_method" in
        neural)
            for dre_method in "${NEURAL_DRE_METHODS_ARRAY[@]}"; do
                for in_dist in "${IN_DIST_ARRAY[@]}"; do
                    for base_type in "${BASE_TYPES_ARRAY[@]}"; do
                        for backbone in "${IMAGE_BACKBONE_ARRAY[@]}"; do
                            experiments+=("$sub_method,$in_dist,$base_type,$dre_method,$backbone")
                        done
                    done
                done
            done
            ;;
        score)
            for score_method in "${SCORE_METHODS_ARRAY[@]}"; do
                for in_dist in "${IN_DIST_ARRAY[@]}"; do
                    # For score mode, prioritize UNet backbone
                    experiments+=("$sub_method,$in_dist,NA,$score_method,unet")
                done
            done
            ;;
        *)
            echo "Warning: Unsupported sub_method '$sub_method', skip."
            ;;
    esac
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
    local sub_method=$1
    local in_dist=$2
    local base_type=$3
    local dre_method=$4
    local backbone=$5
    local gpu_id=$6

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: sub_method=$sub_method, in_dist=$in_dist, base_type=$base_type, subsub_method=$dre_method, backbone=$backbone, gpu_id=$gpu_id"

    if [ "$sub_method" = "score" ]; then
        # Requested score command style:
        # python main.py ... --sub_method score --subsub_method dre_infty/d3re --condition True
        # Plus explicitly prioritize UNet backbone.
        nohup conda run -n osdre python main.py fdiv_estimation \
            --subtask ood_detection \
            --ood_in_dist "$in_dist" \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --sub_method score \
            --subsub_method "$dre_method" \
            --condition True \
            --image_backbone unet \
            --gpu_id $gpu_id \
            > /dev/null 2>&1 &
    else
        # Keep existing neural behavior unchanged
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
    fi

    local pid=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched: sub_method=$sub_method, in_dist=$in_dist, subsub_method=$dre_method, gpu_id=$gpu_id, pid=$pid"
}

# Main execution
echo "============================================"
echo "OS_DRE OOD Detection Experiments - Parallel Run"
echo "============================================"
echo "Sub Methods: ${SUB_METHODS}"
echo "Neural DRE Methods (subsub_method): ${NEURAL_DRE_METHODS}"
echo "Score Methods (subsub_method): ${SCORE_METHODS}"
echo "In-distribution Datasets: ${IN_DIST_DATASETS}"
echo "Base Types: ${BASE_TYPES}"
echo "Image Backbone: ${IMAGE_BACKBONE}"
echo "Total experiments: ${#experiments[@]}"
echo "Available GPUs: ${gpus[@]}"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "============================================"
echo ""
echo "Usage: ./dre_ood_train.sh [neural_methods] [in_dist_datasets] [base_types] [image_backbone] [score_methods] [sub_methods]"
echo "Example: ./dre_ood_train.sh 'logistic nce rkl' 'mnist fashion_mnist' 'universal' 'resnet18' 'dre_infty d3re' 'neural score'"
echo "Neural methods: logistic, nce, infonce, rkl, hellinger, kl, gamma, ew, pw, kulsif"
echo "Score methods: dre_infty, d3re"
echo "Available datasets: mnist, fashion_mnist, cifar10, cifar100"
echo "Available base types: local, general, universal"
echo "Available backbones: resnet18, resnet34, resnet50, wideresnet28_10"
echo ""

# Calculate total experiments
total_experiments=${#experiments[@]}
completed=0

# Process each experiment
for exp in "${experiments[@]}"; do
    IFS=',' read -r sub_method in_dist base_type dre_method backbone <<< "$exp"
    
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
    run_experiment "$sub_method" "$in_dist" "$base_type" "$dre_method" "$backbone" "$available_gpu"
    
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
