# Bulma Stock Prediction Model

This repository contains a two-stage stock prediction model, starting with basic SFT training and then enhancing DOWN prediction accuracy.

## Training Stages

### Stage 1: Basic SFT Training (`Bulma_Senzu_Bean.py`)
- Balanced dataset with equal UP/DOWN distribution
- Basic thinking process with technical and fundamental analysis
- Smaller LoRA parameters (r=16, alpha=32)
- Shorter training duration (3 epochs)
- Focus on general directional prediction skills

### Stage 2: Enhanced DOWN Training (`Bulma_generator.py` & `Bulma_trainer.py`)
- 60% DOWN-biased dataset
- Enhanced thinking process with ambiguous examples
- Larger LoRA parameters (r=24, alpha=48)
- Longer training (10 epochs)
- Custom loss function with 2x weighting for DOWN predictions

## Key Components

1. **First Stage Training**: `Bulma_Senzu_Bean.py`
   - Basic SFT training setup for foundational model capabilities
   - Balanced dataset generation for general prediction skills
   - Standard LoRA configuration for efficient training
   - Equal weighting for UP/DOWN predictions

2. **Dataset Generator**: `Bulma_generator.py`
   - Creates training data with 60% DOWN bias
   - Includes ambiguous examples with mixed signals (25% of DOWN examples)
   - Balanced distribution across energy sector stocks

3. **Enhanced Trainer**: `Bulma_trainer.py`
   - Custom loss function with 2x weighting for DOWN predictions
   - LoRA fine-tuning parameters (r=24, alpha=48 for Qwen model)
   - Device compatibility handling

4. **Training Capsule**: `Bulma_capsule.sh`
   - End-to-end training script 
   - Configurable parameters for dataset size, model selection, and training
   - Includes testing phase

5. **Model Radar**: `Bulma_radar.py`
   - Evaluates model on real-world data
   - Produces accuracy metrics for UP/DOWN predictions
   - Saves predictions to CSV for analysis

6. **Blueprint**: `Bulma_blueprint.md`
   - Comprehensive strategy for enhancing model performance
   - Technical details on implementation

## Usage

### Stage 1: Basic Training

#### Data Generation
```bash
python Bulma_Senzu_Bean.py \
  --mode generate \
  --num_examples 1000 \
  --output_file base_training_data.json
```
Parameters:
- `--mode generate`: Generate training data
- `--num_examples`: Number of examples to generate (default: 1000)
- `--output_file`: Where to save the generated data

Example output:
```json
[
  {
    "text": "<think>\nKey Factors:\n- RSI indicator at 32.5 shows weakening momentum\n- Trading volume decreased by 15.3%\n- Recent price decline of 2.1%\n[...more factors...]\n\nAnalysis:\nBased on technical indicators...\n\nMy final prediction is DOWN with an estimated change of 1.85%\n</think>\n\n<answer>direction: down change: 1.85%</answer>"
  },
  // More examples...
]
```

#### Model Training
```bash
python Bulma_Senzu_Bean.py \
  --mode train \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset_path "base_training_data.json" \
  --output_dir "output/base_model" \
  --num_epochs 3 \
  --batch_size 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --learning_rate 2e-5 \
  --max_seq_length 2048 \
  --gradient_accumulation_steps 4
```
Parameters:
- `--mode train`: Train the model
- `--base_model`: Base model to fine-tune
- `--dataset_path`: Path to training data
- `--output_dir`: Where to save the model
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size per GPU
- `--lora_r`: LoRA rank
- `--lora_alpha`: LoRA alpha
- `--learning_rate`: Learning rate
- `--max_seq_length`: Maximum sequence length
- `--gradient_accumulation_steps`: Gradient accumulation steps

Example output:
```
Training started...
Epoch 1/3: 100%|██████████| 250/250 [12:34<00:00, 3.02it/s]
Loss: 0.1234
Epoch 2/3: 100%|██████████| 250/250 [12:31<00:00, 3.04it/s]
Loss: 0.0856
Epoch 3/3: 100%|██████████| 250/250 [12:33<00:00, 3.03it/s]
Loss: 0.0654
Saving model...
Training completed!
```

#### Model Testing
```bash
python Bulma_Senzu_Bean.py \
  --mode test \
  --model_path "output/base_model" \
  --test_tickers "AAPL,MSFT,GOOGL" \
  --output_file "base_model_predictions.csv" \
  --days_ago 1 \
  --num_predictions 10
```
Parameters:
- `--mode test`: Test the model
- `--model_path`: Path to trained model
- `--test_tickers`: Comma-separated list of tickers to test
- `--output_file`: Where to save predictions
- `--days_ago`: How many days ago to start testing (default: 1)
- `--num_predictions`: Number of predictions per ticker (default: 10)

Example output:
```csv
date,ticker,predicted_direction,predicted_change,actual_direction,actual_change,correct
2024-03-24,AAPL,UP,1.23,UP,1.45,True
2024-03-24,MSFT,DOWN,0.85,DOWN,0.92,True
2024-03-24,GOOGL,UP,1.56,DOWN,0.34,False
```

### Additional Usage Scenarios

#### 1. Cross-Validation Training
```bash
python Bulma_Senzu_Bean.py \
  --mode train \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset_path "base_training_data.json" \
  --output_dir "output/base_model" \
  --num_epochs 3 \
  --cross_validation \
  --num_folds 5
```

#### 2. Sector-Specific Testing
```bash
python Bulma_Senzu_Bean.py \
  --mode test \
  --model_path "output/base_model" \
  --sector "TECHNOLOGY" \
  --num_tickers 10 \
  --output_file "tech_predictions.csv"
```

#### 3. Historical Backtesting
```bash
python Bulma_Senzu_Bean.py \
  --mode test \
  --model_path "output/base_model" \
  --test_tickers "AAPL,MSFT,GOOGL" \
  --start_date "2024-01-01" \
  --end_date "2024-03-24" \
  --output_file "backtest_results.csv"
```

#### 4. Real-Time Testing
```bash
python Bulma_Senzu_Bean.py \
  --mode test \
  --model_path "output/base_model" \
  --test_tickers "AAPL,MSFT,GOOGL" \
  --realtime \
  --interval "1h" \
  --output_file "realtime_predictions.csv"
```

### Stage 2: Enhanced Training
1. Configure the parameters in `Bulma_capsule.sh`
2. Make the script executable: `chmod +x Bulma_capsule.sh`
3. Run the training pipeline: `./Bulma_capsule.sh`

## Training Parameters

### Stage 1 (Base Training)
- Base model: Qwen2.5-1.5B-Instruct
- Dataset: Balanced UP/DOWN examples
- Epochs: 3
- Batch size: 4
- LoRA: r=16, alpha=32
- Equal weighting for all predictions
- Focus on foundational prediction capabilities

### Stage 2 (Enhanced Training)
- Base model: Qwen2.5-1.5B-Instruct
- Dataset: 1,800 examples (60% DOWN-biased)
- Epochs: 10
- Batch size: 4
- LoRA: r=24, alpha=48
- DOWN prediction weight: 2.0

## Changing Models

To use a different base model, modify the `BASE_MODEL` parameter in `Bulma_capsule.sh`:

```bash
# For 3B model
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
FINAL_MODEL="output/down_enhanced_3b_model"
BATCH_SIZE=2
GRADIENT_ACCUM=8

# For 1.5B model
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
FINAL_MODEL="output/down_enhanced_1.5b_model"
BATCH_SIZE=4
GRADIENT_ACCUM=4

# For 7B model
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
FINAL_MODEL="output/down_enhanced_7b_model"
BATCH_SIZE=1
GRADIENT_ACCUM=16
```

When changing models, remember to:
1. Update the `FINAL_MODEL` path to reflect the model size
2. Adjust batch size and gradient accumulation based on available GPU memory
3. Consider adjusting LoRA parameters based on model size

## Expected Results

### Stage 1 (Base Model)
- Balanced UP/DOWN accuracy (55-65%)
- General directional understanding
- Basic technical analysis capabilities
- Foundation for further enhancement

### Stage 2 (Enhanced Model)
- TBD