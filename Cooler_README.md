# Cooler_Qwen3_14b

A highly optimized QLoRA fine-tuning script for Qwen3-14B focused on financial market prediction with options trading recommendations.

## Overview

Cooler_Qwen3_14b transforms the Qwen3-14B model into a trading assistant through reinforcement learning with custom reward mechanisms. The model is designed to:

- Analyze market data to make precise directional predictions (UP/DOWN/STRONG_UP/STRONG_DOWN)
- Provide actionable options trading strategies with specific strike prices and expirations
- Explain reasoning with both technical analysis and fundamental context
- Adapt to different timeframes (hourly, daily, weekly, monthly)
- Manage position sizing based on confidence level

## Technical Implementation

- **QLoRA 4-bit Quantization**: Enables efficient fine-tuning of 14B parameter models
- **Reinforcement Learning**: Custom reward function optimizes for prediction accuracy and trading performance
- **Bankroll Management**: Simulates trading outcomes with dynamic capital allocation
- **Memory Optimization**: Gradient accumulation and chunked processing for efficient training
- **Colab Compatibility**: Designed to run on Google Colab with limited VRAM

## Requirements

- **Hardware**: NVIDIA GPU with at least 40GB VRAM (A100 or equivalent)
- **Software**:
```
torch
transformers>=4.31.0
peft>=0.4.0
bitsandbytes>=0.40.0
accelerate>=0.20.0
datasets
tqdm
pandas
```

Even with 4-bit quantization, the model's memory footprint requires high-end GPU resources due to the 14B parameter size.

## Setup

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Cooler_Qwen3_14b.git
cd Cooler_Qwen3_14b
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Google Colab

The script includes built-in Colab detection and will automatically:
- Mount Google Drive
- Install required packages
- Configure the environment

## Data Format

Training data should be in JSONL format with two key files:
- `with_thinking.jsonl`: Samples that include step-by-step reasoning
- `without_thinking.jsonl`: Samples with direct predictions

Each sample should contain:
```json
{
  "messages": [{"role": "user", "content": "Analyze XYZ stock and predict movement"}],
  "thinking_structured": {
    "prediction": "UP",
    "reasoning": "Price is above 200-day MA with increasing volume...",
    "historical_context": "Stock has shown positive momentum for 3 weeks..."
  },
  "ticker": "XYZ",
  "datetime_str": "2023-05-01 14:30:00",
  "future_prices": [145.20, 146.30, 147.10, 145.80, 148.20],
  "current_price": 145.20
}
```

## Configuration

Key parameters that can be modified in the script:

```python
# Model configuration
model_name = "Qwen/Qwen3-14B"  # Base model
output_dir = "qwen3_14b_memory_optimized_lora"  # Output path

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Alpha scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05
)

# Training parameters
batch_size = 1
num_epochs = 1
learning_rate = 2e-5
```

## Usage

### Running Training

1. Place your data files in the appropriate location:
```
/content/drive/MyDrive/Big_Data/with_thinking.jsonl
/content/drive/MyDrive/Big_Data/without_thinking.jsonl
```

2. Run the script:
```bash
python SS1.py
```

3. For custom data paths:
```python
# Modify these lines in the script
with_thinking_path = '/path/to/with_thinking.jsonl'
without_thinking_path = '/path/to/without_thinking.jsonl'
```

### Training Monitoring

The script provides detailed logs during training:
- Input/output samples
- Reward calculations and components
- Token-level loss information
- Bankroll management metrics

### Output

The trained model is saved to the specified output directory with:
- LoRA adapter weights
- Tokenizer files
- Configuration

## Inference

Load the fine-tuned model for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", 
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "qwen3_14b_memory_optimized_lora")

# Create inference function
def get_prediction(ticker, timeframe="hourly"):
    prompt = f"Analyze {ticker} with {timeframe} chart. Predict direction and options strategy."
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example use
prediction = get_prediction("AAPL", "daily")
print(prediction)
```

## Reward System

The model is optimized using a custom reward function that incentivizes:

1. **Accurate predictions**: Higher rewards for correctly predicting price movements
2. **Technical analysis**: Use of appropriate technical indicators and terminology
3. **Clear formatting**: Using the dual prediction format (hourly + options)
4. **Risk management**: Specifying stop-loss and take-profit levels
5. **Position sizing**: Appropriate position sizing based on confidence

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
- Reduce the `tokens_per_update` value (default: 64)
- Ensure you're using 4-bit quantization
- Try increasing the frequency of `torch.cuda.empty_cache()` calls

### Token ID Errors

If you see "Out-of-vocab token detected!":
- Check if tokenizer vocabulary size matches model vocabulary size
- Ensure proper padding and special token handling

## Advanced Usage

### Custom Reward Functions

You can modify the `custom_reward` function to implement different optimization targets:

```python
def custom_reward(sample, completion):
    # Your custom reward logic here
    reward = 0.0
    
    # Example: Reward for specific concepts
    if "support level" in completion.lower():
        reward += 0.2
    
    return reward
```

### Bankroll Manager

The built-in bankroll manager simulates trading outcomes. Configure parameters:

```python
manager = BankrollManager(
    initial_capital=200.0,  # Starting capital
    position_size_pct=0.20,  # Default position size
    max_position_pct=0.50    # Maximum position size
)
```

## Acknowledgments

Based on the Qwen3-14B model by the Qwen team at Alibaba Cloud.
