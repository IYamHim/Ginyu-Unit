# DOWN-Enhanced Stock Prediction Model

This repository contains a specialized stock prediction model focusing on improving DOWN prediction accuracy for energy sector stocks.

## Key Components

1. **Dataset Generator**: `Bulma_generator.py`
   - Creates training data with 60% DOWN bias
   - Includes ambiguous examples with mixed signals (25% of DOWN examples)
   - Balanced distribution across energy sector stocks

2. **Model Trainer**: `Bulma_trainer.py`
   - Custom loss function with 2x weighting for DOWN predictions
   - LoRA fine-tuning parameters (r=24, alpha=48 for Qwen model)
   - Device compatibility handling

3. **Training Capsule**: `Bulma_capsule.sh`
   - End-to-end training script 
   - Configurable parameters for dataset size, model selection, and training
   - Includes testing phase

4. **Model Radar**: `Bulma_radar.py`
   - Evaluates model on real-world data
   - Produces accuracy metrics for UP/DOWN predictions
   - Saves predictions to CSV for analysis

5. **Blueprint**: `Bulma_blueprint.md`
   - Comprehensive strategy for enhancing model performance
   - Technical details on implementation

## Usage

1. Configure the parameters in `Bulma_capsule.sh`
2. Make the script executable: `chmod +x Bulma_capsule.sh`
3. Run the training pipeline: `./Bulma_capsule.sh`

## Training Parameters

The script is configured to use:
- Qwen2.5-1.5B-Instruct model
- 1,800 examples (60% DOWN-biased)
- 10 training epochs
- Batch size of 4 with gradient accumulation of 4
- LoRA parameters: r=24, alpha=48
- DOWN prediction weight: 2.0 (2x emphasis on DOWN examples)

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
3. Consider reducing LoRA rank for smaller models or increasing it for larger ones

## Expected Results

- Improved DOWN prediction accuracy (target: 60-70%)
- Overall directional accuracy of 70-75%
- Better handling of ambiguous market conditions

Note: Training the model requires a GPU with sufficient VRAM for running the specified model. 