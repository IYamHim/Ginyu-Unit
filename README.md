# Codename Gi-Unit

A deep learning project for stonk market prediction using transformer models.

## Owner
**./install_AI**

## Files

- **Krillon.py**: Stage 1 SFT (Supervised Fine-Tuning) script for initial model training
- **Piccolo.py**: Stage 2 GRPO training script with balanced dataset (50% UP, 50% DOWN)
- **Kakarot.py**: Stage 3 RLHF training script for the model
- **Vegeta.py**: Inference script for making predictions with trained models
- **Over9Thousand.py**: Core library with model training and utility functions
- **train_stage1.sh**: Shell script for stage 1 SFT training
- **train_stage2.sh**: Shell script for stage 2 GRPO training with balanced dataset
- **train_stage3.sh**: Shell script for stage 3 GRPO training
- **test_inference.sh**: Shell script for testing model predictions

## Training Process

The Gi-Unit project uses a three-stage training approach:

1. **Stage 1 (Krillon)**: Supervised Fine-Tuning (SFT) to teach the model the proper format and basic prediction capabilities
2. **Stage 2 (Piccolo)**: Intermediate GRPO training with a balanced dataset (50% UP, 50% DOWN) for easier prediction learning
3. **Stage 3 (Kakarot)**: Advanced GRPO training with the full dataset and enhanced rewards for prediction accuracy and analysis quality

This progressive training approach ensures the model learns the proper format first, then gradually improves its prediction accuracy and reasoning capabilities.

## Reward System

The Namek project implements a progressive reward system across its three training stages:

### Stage 1 (Krillon)
- Uses standard supervised fine-tuning without explicit rewards
- Focuses on teaching the model proper format and basic prediction capabilities

### Stage 2 (Piccolo)
- Moderate rewards for correct predictions (+3.5)
- Balanced penalties for incorrect predictions (-1.5)
- Format compliance rewards for proper structure
- Percentage accuracy rewards based on prediction precision

### Stage 3 (Kakarot)
- Asymmetric rewards favoring DOWN predictions:
  - Correct UP prediction: **+3.8**
  - Correct DOWN prediction: **+4.0** (higher to favor safer predictions)
- Stronger penalties for incorrect predictions (-2.0)
- Sophisticated analysis quality rewards for:
  - Mentioning specific financial metrics
  - Including technical indicators
  - Providing balanced analysis
- Minimum reward floor ensuring correct predictions always receive positive reinforcement

This progressive system guides the model from basic format compliance to sophisticated financial analysis and accurate predictions, with a slight bias toward safer DOWN predictions in uncertain market conditions.

## Usage

### Training

You can train the model using either the Python script directly or the provided shell script:

#### Using the Python script:

```bash
python Kakarot.py \
    --dataset_path "2084Collective/deepstock-sp500-companies-with-info-and-user-prompt" \
    --output_dir outputs_namek \
    --max_samples 5000 \
    --debug
```

#### Using the shell script:

```bash
./train_stage3.sh
```

#### Training Parameters:

- `--dataset_path`: Path to the dataset (default: "2084Collective/deepstock-sp500-companies-with-info-and-user-prompt")
- `--output_dir`: Directory to save model checkpoints (default: "outputs_namek")
- `--max_samples`: Maximum number of samples to use for training (default: None, uses all samples)
- `--debug`: Enable verbose logging to see full model responses
- `--use_pretrained_checkpoint`: Path to a checkpoint to continue training from
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--max_steps`: Maximum number of training steps (default: 10)
- `--per_device_train_batch_size`: Batch size per device (default: 1)

### Inference (Vegeta)

You can run inference using either the Python script directly or the provided test script:

#### Using the Python script:

```bash
python Vegeta.py \
    --model_path "path/to/trained/model" \
    --ticker "MSFT" \
    --company_name "Microsoft Corporation" \
    --current_price 425.52 \
    --previous_price 422.86 \
    --sector "Technology" \
    --industry "Software" \
    --revenue 211900000000 \
    --net_income 72361000000 \
    --eps 9.71 \
    --pe_ratio 33.1 \
    --rsi 65.2 \
    --macd 2.1 \
    --moving_avg_50 410.5 \
    --moving_avg_200 390.2 \
    --news "Microsoft announces new AI features" "Cloud revenue grows 22% in Q2"
```

#### Using the test script:

```bash
./test_inference.sh
```

The test script will run predictions for both Microsoft (bullish case) and Tesla (bearish case) using the pre-trained model.

#### Inference Parameters:

- `--model_path`: Path to the trained model (required)
- `--ticker`: Stonk ticker symbol (required)
- `--company_name`: Company name (required)
- `--current_price`: Current stonk price (required)
- `--previous_price`: Previous stonk price (required)
- `--sector`: Company sector
- `--industry`: Company industry
- `--revenue`: Company revenue
- `--net_income`: Company net income
- `--eps`: Earnings per share
- `--pe_ratio`: Price to earnings ratio
- `--rsi`: RSI value
- `--macd`: MACD value
- `--moving_avg_50`: 50-day moving average
- `--moving_avg_200`: 200-day moving average
- `--news`: Recent news headlines (multiple can be provided)

## Model Output Format

The model generates predictions in a structured format:

```
<think>
Key Factors:
1. [First key observation]
2. [Second key observation]
3. [Third key observation]

Analysis:
[Detailed analysis of technical indicators, financial metrics, and market sentiment]
</think>

<answer>direction: up change: X.X%</answer>
```

## Features

- Real-time stonk market prediction
- Detailed analysis of market factors
- Direction (UP/DOWN) and percentage change predictions
- Reward-based training for improved accuracy
- Verbose logging for model response analysis

## Model Architecture

The project uses the Qwen2.5-1.5B-Instruct model fine-tuned on S&P 500 company data with reinforcement learning from human feedback (RLHF) techniques.

### Stage 2 Training (Piccolo)

```bash
python Piccolo.py \
    --dataset_path "2084Collective/deepstock-sp500-companies-with-info-and-user-prompt" \
    --output_dir outputs_namek_stage2 \
    --max_samples 1000 \
    --use_pretrained_checkpoint outputs_namek_stage1 \
    --debug
```

#### Using the shell script:

```bash
./train_stage2.sh
```

#### Piccolo Parameters:

- `--model_name`: Base model to use (default: "Qwen/Qwen2.5-1.5B-Instruct")
- `--dataset_path`: Path to the dataset (default: "2084Collective/deepstock-sp500-companies-with-info-and-user-prompt")
- `--output_dir`: Directory to save model checkpoints (default: "outputs_namek_stage2")
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--max_steps`: Maximum number of training steps (default: 10000)
- `--max_samples`: Maximum number of samples to use (default: 10000)
- `--per_device_train_batch_size`: Batch size per device (default: 2)
- `--use_pretrained_checkpoint`: Path to a checkpoint to continue training from (default: outputs_namek_stage1)
- `--debug`: Enable verbose logging to see full model responses

### Stage 3 Training (Kakarot)

```bash
python Kakarot.py \
    --dataset_path "2084Collective/deepstock-sp500-companies-with-info-and-user-prompt" \
    --output_dir outputs_namek \
    --debug
```

#### Training Parameters:

- `--dataset_path`: Path to the dataset (default: "2084Collective/deepstock-sp500-companies-with-info-and-user-prompt")
- `--output_dir`: Directory to save model checkpoints (default: "outputs_namek")
- `--max_samples`: Maximum number of samples to use for training (default: None, uses all samples)
- `--debug`: Enable verbose logging to see full model responses
- `--use_pretrained_checkpoint`: Path to a checkpoint to continue training from
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--max_steps`: Maximum number of training steps (default: 20000)
- `--per_device_train_batch_size`: Batch size per device (default: 1)

## Acknowledgements

- The Qwen team for the base model
- Special thanks to Lukas Nel the creator of the 2084Collective/deepstock-sp500-companies-with-info-and-user-prompt dataset for the Stonk-Trainer
- The PyTorch and Hugging Face communities 
