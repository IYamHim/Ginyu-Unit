#!/bin/bash

# Namek Stage 3 Training Script
# This script runs the advanced GRPO training for the stonk prediction model
# Owner: ./install_AI

# Create output directory if it doesn't exist
mkdir -p outputs_namek_stage3

# Define parameters
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH="2084Collective/deepstock-sp500-companies-with-info-and-user-prompt"
OUTPUT_DIR="./outputs_namek_stage3"
TRAIN_EPOCHS=1
MAX_STEPS=20000  # Increased to handle the full dataset (estimated size)
BATCH_SIZE=1
STAGE2_CHECKPOINT="./outputs_namek_stage2"  # Path to the stage 2 model

echo "=== Starting Namek Stage 3 (Kakarot) Advanced GRPO Training ==="
echo "Owner: ./install_AI"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Starting from stage 2 checkpoint: $STAGE2_CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo "Training epochs: $TRAIN_EPOCHS"
echo "Max steps: $MAX_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Using FULL DATASET (no sample limit)"
echo "=================================================="

# Run the training script
python Kakarot.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $TRAIN_EPOCHS \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --use_pretrained_checkpoint $STAGE2_CHECKPOINT \
    --debug

echo "=== Training complete! ==="
echo "Run ./test_inference.sh to test the model" 