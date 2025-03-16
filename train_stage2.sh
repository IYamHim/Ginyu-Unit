#!/bin/bash

# Namek Stage 2 Training Script
# This script runs the intermediate GRPO training for the stonk prediction model
# with a balanced 50% UP / 50% DOWN dataset for easier prediction
# Owner: ./install_AI

# Create output directory if it doesn't exist
mkdir -p outputs_namek_stage2

# Define parameters
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH="2084Collective/deepstock-sp500-companies-with-info-and-user-prompt"
OUTPUT_DIR="./outputs_namek_stage2"
TRAIN_EPOCHS=1
MAX_STEPS=10000
BATCH_SIZE=2
MAX_SAMPLES=10000
STAGE1_CHECKPOINT="./outputs_namek_stage1"

echo "=== Starting Namek Stage 2 (Piccolo) GRPO Training ==="
echo "Owner: ./install_AI"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Starting from stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo "Training epochs: $TRAIN_EPOCHS"
echo "Max steps: $MAX_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Max samples: $MAX_SAMPLES"
echo "=================================================="
echo "Using balanced dataset with 50% UP and 50% DOWN examples for easier training"
echo "=================================================="

# Run the training script
python Piccolo.py \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $TRAIN_EPOCHS \
    --max_steps $MAX_STEPS \
    --max_samples $MAX_SAMPLES \
    --per_device_train_batch_size $BATCH_SIZE \
    --use_pretrained_checkpoint $STAGE1_CHECKPOINT \
    --debug

echo "=== Stage 2 Training complete! ==="
echo "The model is now ready for Stage 3 training with Kakarot.py" 