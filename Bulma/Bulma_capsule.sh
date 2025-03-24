#!/bin/bash

# Set variables
NUM_EXAMPLES=1800
DOWN_PERCENT=0.60
OUTPUT_DIR="down_enhanced_dataset"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
FINAL_MODEL="output/down_enhanced_1.5b_model"
EPOCHS=10
BATCH_SIZE=4
GRADIENT_ACCUM=4
LEARNING_RATE=2e-5
LORA_R=24
LORA_ALPHA=48
DOWN_WEIGHT=2.0
UP_WEIGHT=1.0

echo "==== DOWN-Enhanced Model Training Pipeline ===="
echo "Step 1: Generating dataset with $DOWN_PERCENT DOWN bias ($NUM_EXAMPLES examples)"

# Generate the DOWN-biased dataset
python Bulma_generator.py \
  --num_examples $NUM_EXAMPLES \
  --output_dir $OUTPUT_DIR \
  --down_percent $DOWN_PERCENT

if [ $? -ne 0 ]; then
  echo "Error generating dataset. Exiting."
  exit 1
fi

echo "Step 2: Training model with DOWN-weighted loss"

# Train the model with directional weighted loss
python Bulma_trainer.py \
  --dataset "$OUTPUT_DIR/down_enhanced_dataset.jsonl" \
  --base_model $BASE_MODEL \
  --output_dir $FINAL_MODEL \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --gradient_accumulation_steps $GRADIENT_ACCUM \
  --down_weight $DOWN_WEIGHT \
  --up_weight $UP_WEIGHT

if [ $? -ne 0 ]; then
  echo "Error training model. Exiting."
  exit 1
fi

echo "Step 3: Testing model performance"

# Test the model
python Bulma_radar.py \
  --test_all \
  --model_path $FINAL_MODEL \
  --days_ago 1 \
  --temperature 0.1 \
  --output down_enhanced_1.5b_predictions.csv

echo "==== Pipeline Complete ===="
echo "Dataset: $OUTPUT_DIR/down_enhanced_dataset.jsonl"
echo "Model: $FINAL_MODEL"
echo "Results: down_enhanced_1.5b_predictions.csv" 