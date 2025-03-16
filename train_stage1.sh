#!/bin/bash

# Namek Stage 1 Training Script
# This script runs the initial SFT training for the stonk prediction model
# Owner: ./install_AI

# Create output directory if it doesn't exist
mkdir -p outputs_namek_stage1

# Define parameters
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH="2084Collective/deepstock-sp500-companies-with-info-and-user-prompt"
OUTPUT_DIR="./outputs_namek_stage1"
TRAIN_EPOCHS=1
MAX_STEPS=1000
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
MAX_SAMPLES=5000  # Initial training with 5000 samples

echo "=== Starting Namek Stage 1 (Krillon) SFT Training ==="
echo "Owner: ./install_AI"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Training epochs: $TRAIN_EPOCHS"
echo "Max steps: $MAX_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Max samples: $MAX_SAMPLES"
echo "=================================================="

# Run the training script
python Krillon.py \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $TRAIN_EPOCHS \
    --max_steps $MAX_STEPS \
    --max_samples $MAX_SAMPLES \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE

echo "=== Stage 1 Training complete! ==="
echo "The model is now ready for Stage 2 training" 