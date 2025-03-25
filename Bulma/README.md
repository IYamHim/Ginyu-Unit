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
```bash
python Bulma_Senzu_Bean.py \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset_path "base_training_data.json" \
  --output_dir "output/base_model" \
  --num_epochs 3 \
  --batch_size 4 \
  --lora_r 16 \
  --lora_alpha 32
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
- Improved DOWN prediction accuracy (60-70%)
- Overall directional accuracy of 70-75%
- Better handling of ambiguous market conditions

Note: Training requires a GPU with sufficient VRAM for running the specified model. 