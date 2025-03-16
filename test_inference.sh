#!/bin/bash

# Test both original and new models
# Owner: ./install_AI

# Original model path
ORIGINAL_MODEL_PATH="../outputs_up_down_stage3_v2"
# New model path
NEW_MODEL_PATH="./outputs_namek_stage3/checkpoint-30"

# Function to test a model
test_model() {
  local model_path=$1
  local model_name=$2
  
  echo "========================================================"
  echo "Testing $model_name model at $model_path"
  echo "Owner: ./install_AI"
  echo "========================================================"
  
  # Microsoft test (bullish)
  echo "Testing Microsoft stonk prediction (bullish)..."
  python Vegeta.py \
    --model_path $model_path \
    --ticker "MSFT" \
    --company_name "Microsoft Corporation" \
    --sector "Technology" \
    --industry "Software" \
    --current_price 425.52 \
    --previous_price 422.86 \
    --revenue 211900000000 \
    --net_income 72361000000 \
    --eps 9.71 \
    --pe_ratio 33.1 \
    --rsi 65.2 \
    --macd 2.1 \
    --moving_avg_50 410.5 \
    --moving_avg_200 390.2 \
    --news "Microsoft announces new AI features for Office apps" \
           "Cloud revenue grows 22% in Q2" \
           "Microsoft partners with OpenAI for new project" \
           "Analysts raise price target for MSFT stonk"
  
  # Tesla test (bearish)
  echo -e "\n\nTesting Tesla stonk prediction (bearish)..."
  python Vegeta.py \
    --model_path $model_path \
    --ticker "TSLA" \
    --company_name "Tesla, Inc." \
    --sector "Automotive" \
    --industry "Electric Vehicles" \
    --current_price 251.52 \
    --previous_price 265.28 \
    --revenue 96773000000 \
    --net_income 15292000000 \
    --eps 4.30 \
    --pe_ratio 118.9 \
    --rsi 39.8 \
    --macd -1.2 \
    --moving_avg_50 260.1 \
    --moving_avg_200 240.5 \
    --news "Tesla faces production challenges in Berlin factory" \
           "Q2 deliveries miss analyst expectations" \
           "Competition intensifies in EV market" \
           "Tesla cuts prices in key markets to boost demand"
}

# Test the original pre-trained model
test_model "$ORIGINAL_MODEL_PATH" "Original"

# Check if a new model exists, and test it
if [ -d "$NEW_MODEL_PATH" ]; then
  test_model "$NEW_MODEL_PATH" "Newly Trained"
else
  echo "New model checkpoint not found. Training may not have been run yet."
fi 