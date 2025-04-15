# === START OF MODIFIED CODE CELL 1: Mount Drive and Install Dependencies ===
# === Mount Google Drive ===
print("="*50 + "\n")
# === END OF MODIFIED CODE CELL 1 ===

# === FIXED IMPORTS CELL (CELL 2) ===
import os
import sys
import json
import torch
import random
import logging
import numpy as np
import pandas as pd
import re
from pathlib import Path
from tqdm.notebook import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

# Function to check bfloat16 support - ADDED MISSING FUNCTION
def is_bfloat16_supported():
    """Check if bfloat16 precision is supported on the current device."""
    if torch.cuda.is_available():
        if hasattr(torch.cuda, 'is_bf16_supported'):
            return torch.cuda.is_bf16_supported()
        else:
            try:
                torch.tensor([1.0], dtype=torch.bfloat16).cuda()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return True
            except Exception:
                return False
    return False

# --- Unsloth and Hugging Face imports ---
# IMPORT UNSLOTH FIRST as per the warning
try:
    import unsloth
    from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from datasets import load_dataset, Dataset
    from peft import LoraConfig, PeftModel, TaskType
    # VLLM import check
    import vllm
    print("Successfully imported Unsloth, TRL, Transformers, Datasets, PEFT, vLLM.")
except ImportError as e:
    print(f"Import Error: {e}. Make sure you've restarted the runtime after installation.")
    raise

# Unsloth FastLanguageModel - Import after base Unsloth
from unsloth import FastLanguageModel

# PyTorch utilities
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import gc # Garbage collection

# === Logging Setup ===
# Remove default handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("vllm").setLevel(logging.INFO)

# === Random Seed ===
def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set random seed to {seed}")

RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)

print("Imports and basic setup complete.")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16 Supported: {is_bfloat16_supported()}")

# Add this to your notebook if not already present
def load_and_prepare_dataset(dataset_path, max_samples=None):
    """Load and prepare the GRPO_PnL_Trainer.jsonl dataset"""
    logger.info(f"Loading dataset from {dataset_path}")

    try:
        # Load the dataset using Hugging Face datasets
        full_dataset = load_dataset("json", data_files=dataset_path, split="train")

        if max_samples and max_samples > 0 and max_samples < len(full_dataset):
            logger.info(f"Limiting dataset to {max_samples} samples (from {len(full_dataset)})")
            dataset = full_dataset.select(range(max_samples))
        else:
            dataset = full_dataset
            logger.info(f"Using all {len(dataset)} samples from dataset")

        # Check that the dataset has the expected fields
        required_fields = [
            'ticker', 'datetime_str', 'Open', 'High', 'Low', 'Close', 'Volume',
            'MA4', 'MA8', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Lower', 'current_price', 'future_prices',
            'actual_direction', 'actual_percentage', 'volatility'
        ]

        example = dataset[0] if len(dataset) > 0 else {}
        missing_fields = [field for field in required_fields if field not in example]

        if missing_fields:
            logger.warning(f"Dataset is missing expected fields: {missing_fields}")
            logger.warning("This may cause issues with training. Check your data.")

        return dataset

    except Exception as e:
        logger.error(f"Error loading or preparing dataset: {e}")
        raise

# === GRPO CONFIGURATION OPTIMIZED FOR L4 GPU (22.5GB VRAM) ===
import os
import logging
from trl import GRPOConfig

logger = logging.getLogger(__name__)

# --- Configuration ---
output_dir = "/content/drive/MyDrive/outputs_grpo_pnl_L4"
os.makedirs(output_dir, exist_ok=True)

# --- Define GRPO Training Arguments ---
training_args = GRPOConfig(
    output_dir = output_dir,
    # --- DISABLE VLLM ---
    use_vllm = False,
    # -------------------
    learning_rate = 2e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",   # 8-bit optimizer for memory savings
    logging_steps = 10,
    save_strategy="steps",
    save_steps = 100,
    save_total_limit = 2,

    # --- REDUCED BATCH SIZE FOR L4 GPU ---
    per_device_train_batch_size = 2,   # Reduced from 4 to 2
    num_generations = 2,               # Match batch size
    gradient_accumulation_steps = 4,   # Increase to compensate for smaller batch

    # --- REDUCED SEQUENCE LENGTHS ---
    max_prompt_length = 4092,          # Reduced from 1536
    max_completion_length = 1024,       # Reduced from 512
    max_steps = 500,

    # --- Memory Optimization ---
    max_grad_norm = 0.5,
    # --- USE FP16 for L4 ---
    bf16 = False,                      # L4 works better with fp16
    fp16 = True,                       # ENABLE float16
    gradient_checkpointing = True,     # Essential for memory savings

    # --- Additional Memory Optimizations ---
    dataloader_drop_last = True,
    dataloader_num_workers = 1,        # Reduce worker count
    greater_is_better = False,
    remove_unused_columns = False,     # Critical for metadata passing

    # --- Other ---
    logging_first_step = True,
    eval_strategy = "no",
    seed = 42,
    report_to = "none",
)

logger.info(f"GRPO Config optimized for L4 GPU with 22.5GB VRAM")
logger.info(f"Precision settings: fp16={training_args.fp16}, bf16={training_args.bf16}")
logger.info(f"Batch params: per_device={training_args.per_device_train_batch_size}, num_gen={training_args.num_generations}, accum={training_args.gradient_accumulation_steps}")
logger.info(f"Sequence length: prompt={training_args.max_prompt_length}, completion={training_args.max_completion_length}, total={training_args.max_prompt_length + training_args.max_completion_length}")
effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
logger.info(f"Effective batch size: {effective_batch_size} prompts per update")

# === FIXED GRPO CONFIGURATION FOR A100/vLLM ===
import os
import logging
from trl import GRPOConfig

logger = logging.getLogger(__name__)

# --- Configuration ---
output_dir = "/content/drive/MyDrive/outputs_grpo_pnl_vllm_fixed"
os.makedirs(output_dir, exist_ok=True)

# --- Define GRPO Training Arguments ---
training_args = GRPOConfig(
    output_dir = output_dir,
    # --- ENABLE VLLM ---
    use_vllm = True,  # CRITICAL for high-performance training
    # -------------------
    learning_rate = 2e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",

    # --- Lower this to 'adamw_torch' if 8bit optimizer causes issues ---
    optim = "adamw_torch",  # More stable than 8bit version

    logging_steps = 10,
    save_strategy="steps",
    save_steps = 100,
    save_total_limit = 2,

    # --- Batching & Generation ---
    # These values work well with vLLM
    per_device_train_batch_size = 1, # Start with 1 for stability
    num_generations = 4,             # Reduced from 8 for stability
    gradient_accumulation_steps = 4, # Overall effective batch: 16 completions

    # --- Lower these values if running into memory issues ---
    max_prompt_length = 1024,  # Reduced for more stability
    max_completion_length = 256, # Reduced for more stability

    # --- Training Duration ---
    max_steps = 500,            # Will override epochs

    # --- Precision & Performance ---
    max_grad_norm = 0.5,
    bf16 = True,  # For A100 GPUs
    fp16 = False, # Don't enable both
    gradient_checkpointing = True,

    # --- Critical for our reward function ---
    remove_unused_columns = False,
    logging_first_step = True,
    eval_strategy = "no",
    seed = 42,
    report_to = "none",
)

logger.info(f"GRPOConfig defined: use_vllm={training_args.use_vllm}, bf16={training_args.bf16}, fp16={training_args.fp16}")
effective_batch_size = training_args.per_device_train_batch_size * training_args.num_generations * training_args.gradient_accumulation_steps
logger.info(f"Effective batch size: {effective_batch_size} completions per gradient update")

# === GRPO CONFIGURATION WITH MAXIMIZED VRAM USAGE ===
import os
import logging
from trl import GRPOConfig

logger = logging.getLogger(__name__)

# --- Configuration ---
output_dir = "/content/drive/MyDrive/outputs_grpo_pnl_fixed"
os.makedirs(output_dir, exist_ok=True)

# --- Define GRPO Training Arguments ---
training_args = GRPOConfig(
    output_dir = output_dir,
    # --- DISABLE VLLM ---
    use_vllm = True,
    # -------------------
    learning_rate = 2e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch",
    logging_steps = 10,
    save_strategy="steps",
    save_steps = 100,
    save_total_limit = 2,

    # --- INCREASE BATCH SIZE TO USE MORE VRAM ---
    per_device_train_batch_size = 4,   # Double the batch size to 4
    num_generations = 4,               # Increase generations to match batch size
    gradient_accumulation_steps = 2,   # Lower this since we increased batch size

    # --- USE FULL CONTEXT LENGTH ---
    max_prompt_length = 1536,          # Keep prompt length (3/4)
    max_completion_length = 512,       # Keep completion length (1/4)
    max_steps = 500,

    # --- Memory Optimization ---
    max_grad_norm = 0.5,
    bf16 = True,                       # For A100
    fp16 = False,                      # Don't enable both
    gradient_checkpointing = True,     # Keep for memory savings

    # --- Disable unnecessary memory usage ---
    dataloader_drop_last = True,       # Small memory saving
    dataloader_num_workers = 4,        # Parallelize data loading
    greater_is_better = False,         # For gradient accumulation
    remove_unused_columns = False,     # Critical for metadata passing

    # --- Other ---
    logging_first_step = True,
    eval_strategy = "no",
    seed = 42,
    report_to = "none",
)

logger.info(f"GRPOConfig defined: use_vllm={training_args.use_vllm}, bf16={training_args.bf16}, fp16={training_args.fp16}")
logger.info(f"Batch params: per_device={training_args.per_device_train_batch_size}, num_gen={training_args.num_generations}, accum={training_args.gradient_accumulation_steps}")
logger.info(f"Sequence length: prompt={training_args.max_prompt_length}, completion={training_args.max_completion_length}, total={training_args.max_prompt_length + training_args.max_completion_length}")
effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
logger.info(f"Effective batch size: {effective_batch_size} prompts per update")

# === GRPO CONFIGURATION OPTIMIZED FOR L4 GPU (22.5GB VRAM) ===
import os
import logging
from trl import GRPOConfig

logger = logging.getLogger(__name__)

# --- Configuration ---
output_dir = "/content/drive/MyDrive/outputs_grpo_pnl_L4"
os.makedirs(output_dir, exist_ok=True)

# --- Define GRPO Training Arguments ---
training_args = GRPOConfig(
    output_dir = output_dir,
    # --- DISABLE VLLM ---
    use_vllm = False,
    # -------------------
    learning_rate = 2e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",   # 8-bit optimizer for memory savings
    logging_steps = 2,
    save_strategy="steps",
    save_steps = 50,
    save_total_limit = 2,

    # --- REDUCED BATCH SIZE FOR L4 GPU ---
    per_device_train_batch_size = 2,   # Reduced from 4 to 2
    num_generations = 2,               # Match batch size
    gradient_accumulation_steps = 4,   # Increase to compensate for smaller batch

    # --- REDUCED SEQUENCE LENGTHS ---
    max_prompt_length = 1024,          # Reduced from 1536
    max_completion_length = 1024,       # Reduced from 512
    max_steps = 500,

    # --- Memory Optimization ---
    max_grad_norm = 0.5,
    # --- CRITICAL FIX: Use fp16 instead of bf16 ---
    bf16 = False,                      # MUST be False since model is in fp16
    fp16 = True,                       # MUST be True to match model precision
    gradient_checkpointing = True,     # Essential for memory savings

    # --- Additional Memory Optimizations ---
    dataloader_drop_last = True,
    dataloader_num_workers = 1,        # Reduce worker count
    greater_is_better = False,
    remove_unused_columns = False,     # Critical for metadata passing

    # --- Other ---
    logging_first_step = True,
    eval_strategy = "no",
    seed = 42,
    report_to = "none",
)

logger.info(f"GRPO Config optimized for L4 GPU with 22.5GB VRAM")
logger.info(f"Precision settings: fp16={training_args.fp16}, bf16={training_args.bf16}")
logger.info(f"Batch params: per_device={training_args.per_device_train_batch_size}, num_gen={training_args.num_generations}, accum={training_args.gradient_accumulation_steps}")
logger.info(f"Sequence length: prompt={training_args.max_prompt_length}, completion={training_args.max_completion_length}, total={training_args.max_prompt_length + training_args.max_completion_length}")
effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
logger.info(f"Effective batch size: {effective_batch_size} prompts per update")

def load_and_process_data():
    logger.info("Loading and processing dataset...")

    # Load your custom dataset
    dataset_path = "/content/drive/MyDrive/Big_Data/GRPO_PnL_Trainer.jsonl"
    logger.info(f"Loading dataset from: {dataset_path}")

    from datasets import load_dataset
    raw_dataset = load_dataset('json', data_files=dataset_path)['train']

    # Optionally limit the dataset size during development
    max_samples = 2000
    if len(raw_dataset) > max_samples:
        logger.info(f"Limiting dataset to {max_samples} samples (from {len(raw_dataset)})")
        raw_dataset = raw_dataset.select(range(max_samples))

    logger.info(f"Successfully prepared dataset with {len(raw_dataset)} samples.")
    logger.info(f"Dataset features: {raw_dataset.features}")

    # Format dataset for financial analysis instruction tuning
    def format_financial_prompt(example):
        # Create a structured prompt from financial data
        prompt = f"""Analyze the following stock data for {example['ticker']} in the {example['sector']} sector:
Date: {example['datetime_str']}
Price: Open=${example['Open']:.2f}, High=${example['High']:.2f}, Low=${example['Low']:.2f}, Close=${example['Close']:.2f}
Volume: {int(example['Volume'])}
Technical Indicators:
- MA4: ${example['MA4']:.2f}
- MA8: ${example['MA8']:.2f}
- MA20: ${example['MA20']:.2f}
- RSI: {example['RSI']:.2f}
- MACD: {example['MACD']:.2f}, Signal: {example['MACD_Signal']:.2f}, Histogram: {example['MACD_Hist']:.2f}
- Bollinger Bands: Upper=${example['BB_Upper']:.2f}, Lower=${example['BB_Lower']:.2f}
- Price Change: ${example['Price_Change']:.2f} ({example['Pct_Change']:.2f}%)
- Current Price: ${example['current_price']:.2f}
- Volatility: {example['volatility']:.4f}

Based on this information, predict the price direction and percentage change."""

        # Create completion (model's response) using actual direction and percentage
        completion = f"""After analyzing the data for {example['ticker']}, I predict the price will move {example['actual_direction']} by approximately {abs(example['actual_percentage']):.2f}%.

My analysis:
{example['thinking_trace']}"""

        # Format the full text in the chat template format
        formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"

        return {"text": formatted_text}

    # Apply formatting
    formatted_dataset = raw_dataset.map(format_financial_prompt)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=1536,
        )

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in raw_dataset.column_names] + ["text"]
    )

    logger.info(f"Dataset processed: {len(tokenized_dataset)} examples")
    return tokenized_dataset

# === Prepare Dataset ===

# --- Configuration ---
# !!! ADJUST THIS PATH to your actual dataset location !!!
dataset_path = '/content/drive/MyDrive/Big_Data/GRPO_PnL_Trainer.jsonl'
max_samples_train = 2000 # Limit number of samples for faster testing, set to None to use all valid samples

logger.info(f"Loading dataset from: {dataset_path}")
# Use the function defined in Cell 3
train_dataset = load_and_prepare_dataset(dataset_path, max_samples=max_samples_train)

if train_dataset is None or len(train_dataset) == 0:
    raise RuntimeError("Dataset loading failed or resulted in an empty dataset. Please check the path and data.")

logger.info(f"Successfully prepared dataset with {len(train_dataset)} samples.")
logger.info(f"Dataset features: {train_dataset.features}")

# Optional: Inspect a sample
print("\nSample Prompt:")
print(train_dataset[0]['prompt'])
print("\nSample Metadata:")
print({k: train_dataset[0][k] for k in train_dataset.column_names if k != 'prompt'})

# === HELPER FUNCTIONS FOR TRADING ===
import re
import numpy as np

# Trade manager class
class TradeManager:
    def __init__(self, stop_loss_pct=0.02, take_profit_pct=0.03, max_holding_periods=5):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_periods = max_holding_periods

    def simulate_trade(self, entry_price, future_prices, direction):
        """Simulate a trade with risk management rules."""
        if direction not in ['UP', 'DOWN']:
            return 0.0, 0, 'INVALID'

        # Convert to numpy array if it's a list
        if isinstance(future_prices, list):
            future_prices = np.array(future_prices)

        # Get the first min(max_holding_periods, len(future_prices)) prices
        n_periods = min(self.max_holding_periods, len(future_prices))
        prices = future_prices[:n_periods]

        # Calculate price changes as percentages
        price_changes = (prices - entry_price) / entry_price

        # Adjust sign based on direction (multiply by -1 for DOWN)
        if direction == 'DOWN':
            price_changes = -price_changes

        # Check for stop loss or take profit hits
        stop_loss_hit = np.any(price_changes <= -self.stop_loss_pct)
        take_profit_hit = np.any(price_changes >= self.take_profit_pct)

        # Determine exit condition
        if take_profit_hit:
            # Find first index where take profit is hit
            exit_idx = np.argmax(price_changes >= self.take_profit_pct)
            return self.take_profit_pct, exit_idx + 1, 'TAKE_PROFIT'
        elif stop_loss_hit:
            # Find first index where stop loss is hit
            exit_idx = np.argmax(price_changes <= -self.stop_loss_pct)
            return -self.stop_loss_pct, exit_idx + 1, 'STOP_LOSS'
        else:
            # Exit at end of simulation period with the last price change
            return float(price_changes[-1]), n_periods, 'TIME_EXIT'

# Parse model outputs
def parse_trade_prediction(completion_text):
    """Parse model predictions from completion text."""
    prediction = {
        'direction': None,
        'percentage': None,
        'confidence': 0.5,  # Default confidence
        'entry_conditions': [],
        'exit_conditions': []
    }

    # Extract direction and percentage from <answer> tag
    answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip().upper()
        # Check for UP/DOWN keywords
        if "UP" in answer_text:
            prediction['direction'] = "UP"
        elif "DOWN" in answer_text:
            prediction['direction'] = "DOWN"

        # Try to extract percentage if present
        percentage_match = re.search(r'(\d+\.?\d*)%', answer_text)
        if percentage_match:
            prediction['percentage'] = float(percentage_match.group(1)) / 100.0

    # Extract confidence
    confidence_match = re.search(r'<confidence>(.*?)</confidence>', completion_text, re.DOTALL)
    if confidence_match:
        confidence_text = confidence_match.group(1).strip()
        # Try different formats (0.XX, XX%, or XX/100)
        if re.match(r'^0\.\d+$', confidence_text):
            prediction['confidence'] = float(confidence_text)
        elif re.match(r'^\d+%$', confidence_text):
            prediction['confidence'] = float(confidence_text.strip('%')) / 100.0
        elif re.match(r'^\d+/100$', confidence_text):
            prediction['confidence'] = float(confidence_text.split('/')[0]) / 100.0
        elif re.match(r'^\d+$', confidence_text) and int(confidence_text) <= 100:
            prediction['confidence'] = int(confidence_text) / 100.0

    # Extract entry conditions
    entry_match = re.search(r'<entry_conditions>(.*?)</entry_conditions>', completion_text, re.DOTALL)
    if entry_match:
        entry_text = entry_match.group(1).strip()
        prediction['entry_conditions'] = [line.strip() for line in entry_text.split('\n') if line.strip()]

    # Extract exit conditions
    exit_match = re.search(r'<exit_conditions>(.*?)</exit_conditions>', completion_text, re.DOTALL)
    if exit_match:
        exit_text = exit_match.group(1).strip()
        prediction['exit_conditions'] = [line.strip() for line in exit_text.split('\n') if line.strip()]

    return prediction

# Calculate trade reward
def calculate_trade_reward(prediction, metadata, trade_manager):
    """Calculate reward based on trade prediction and real price movements."""
    # Initialize metrics
    metrics = {
        'pnl': 0.0,
        'direction_score': 0.0,
        'exit_periods': 0,
        'exit_reason': 'NONE',
        'risk_reward_ratio': 0.0,
        'prediction_matched_actual': False
    }

    # Extract required values from metadata
    current_price = metadata.get('current_price')
    future_prices = metadata.get('future_prices')
    actual_direction = metadata.get('actual_direction')

    if not all([current_price, future_prices, actual_direction]):
        return -0.1, metrics  # Penalty for missing data

    # Convert prediction to values needed for simulation
    direction = prediction.get('direction')

    # Validate we have a valid direction prediction
    if direction not in ['UP', 'DOWN']:
        return -0.2, metrics  # Larger penalty for invalid prediction

    # Direction score: +0.3 for correct, -0.2 for incorrect
    metrics['direction_score'] = 0.3 if direction == actual_direction else -0.2
    metrics['prediction_matched_actual'] = (direction == actual_direction)

    # Simulate the trade using the trade manager
    pnl, periods_held, exit_reason = trade_manager.simulate_trade(
        entry_price=current_price,
        future_prices=future_prices,
        direction=direction
    )

    metrics['pnl'] = pnl
    metrics['exit_periods'] = periods_held
    metrics['exit_reason'] = exit_reason

    # Calculate reward components
    direction_reward = metrics['direction_score']

    # PnL reward: Scale based on stop loss and take profit
    # This ensures reward is proportional to performance
    tp_sl_range = trade_manager.take_profit_pct + trade_manager.stop_loss_pct
    pnl_reward = (pnl + trade_manager.stop_loss_pct) / tp_sl_range  # Normalize to [0, 1]
    pnl_reward = (pnl_reward * 0.5) - 0.25  # Scale to [-0.25, 0.25]

    # Risk management reward
    risk_reward = 0.0
    if exit_reason == 'TAKE_PROFIT':
        risk_reward = 0.2  # Bonus for hitting take profit
    elif exit_reason == 'STOP_LOSS':
        risk_reward = -0.1  # Small penalty for stop loss

    metrics['risk_reward_ratio'] = risk_reward

    # Calculate final reward
    final_reward = direction_reward + pnl_reward + risk_reward

    # Clip to valid range
    final_reward = max(-1.0, min(1.0, final_reward))

    return final_reward, metrics

# === REWARD FUNCTION AND TRAINING ===
import torch
import logging
import re
import os
from trl import GRPOTrainer
import numpy as np

logger = logging.getLogger(__name__)

# --- Main reward function ---
def pnl_reward_func(completions, **kwargs):
    """
    Enhanced reward function with multiple components and debugging
    """
    num_completions = len(completions)
    logger.info(f"Reward func received {num_completions} completions")

    # DEBUG: Print sample completions every 10 steps
    step = kwargs.get("step", [0])[0] if isinstance(kwargs.get("step", []), list) else 0
    if step % 10 == 0 or step < 5:
        sample_idx = 0 if len(completions) > 0 else -1
        if sample_idx >= 0:
            logger.info(f"SAMPLE COMPLETION (step {step}):\n{'-'*50}\n{completions[sample_idx][:500]}...\n{'-'*50}")

    # Initialize list of rewards with exactly the same length as completions
    rewards = []

    # Process each completion
    for i, completion_text in enumerate(completions):
        try:
            # Calculate which batch item this belongs to
            batch_idx = i // training_args.num_generations
            gen_idx = i % training_args.num_generations

            # Extract metadata for this completion
            meta = {}
            for key in ['current_price', 'future_prices', 'volatility', 'actual_direction', 'actual_percentage', 'ticker', 'datetime_str', 'RSI']:
                if key not in kwargs or not isinstance(kwargs[key], list) or len(kwargs[key]) <= batch_idx:
                    logger.error(f"Missing or invalid metadata for {key} at index {batch_idx}")
                    rewards.append(-0.1)  # Penalty for missing metadata
                    continue
                meta[key] = kwargs[key][batch_idx]

            # Add technical indicators in expected format
            meta['technical_indicators'] = {'RSI': meta.get('RSI')}

            # Parse prediction from model output
            prediction = parse_trade_prediction(completion_text)
            prediction['full_response'] = completion_text  # Add full text for format checking

            # DEBUG: Check what's being parsed (or not parsed)
            if i == 0 and (step % 10 == 0 or step < 5):
                logger.info(f"PARSED PREDICTION: direction={prediction.get('direction')}, percentage={prediction.get('percentage')}, confidence={prediction.get('confidence')}")
                # Search for answer tag content
                answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.DOTALL | re.IGNORECASE)
                if answer_match:
                    logger.info(f"ANSWER TAG CONTENT: '{answer_match.group(1).strip()}'")
                else:
                    logger.info("NO ANSWER TAG FOUND")

            # --- Calculate reward components ---
            xml_reward = xml_tag_count_reward(completion_text)
            format_reward_val = format_reward(prediction)
            direction_reward_val = direction_reward(prediction, meta.get('actual_direction'))
            strategy_reward = strategy_coherence_reward(prediction)

            # If format is good enough, calculate PnL reward
            pnl_reward = 0.0
            trade_metrics = {}
            if prediction['direction'] is not None:
                pnl_reward_val, trade_metrics = calculate_trade_reward(
                    prediction=prediction,
                    metadata=meta,
                    trade_manager=trade_manager_instance
                )
                # Only take the 'pnl' and 'risk_management' components
                # from the calculate_trade_reward result
                pnl_reward = pnl_reward_val * 0.5  # Weight this a bit less at first

            # --- Combine all rewards ---
            combined_reward = (
                xml_reward +            # 0.25 max - Having the right tags
                format_reward_val +     # 0.25 max - Structured data in correct format
                direction_reward_val +  # 0.3 max - Getting direction right
                strategy_reward +       # 0.2 max - Having a coherent strategy
                pnl_reward              # Using the original PnL calculation with weight
            )

            # Ensure reward is within reasonable bounds
            final_reward = max(-1.0, min(1.0, combined_reward))

            # Log the reward calculation
            logger.info(f"Completion {i} reward: {final_reward:.4f} [xml:{xml_reward:.2f}, fmt:{format_reward_val:.2f}, dir:{direction_reward_val:.2f}, strat:{strategy_reward:.2f}, pnl:{pnl_reward:.2f}] - Dir: {prediction.get('direction')}, Actual: {meta.get('actual_direction')}")

            rewards.append(float(final_reward))

        except Exception as e:
            logger.error(f"Error calculating reward for completion {i}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            rewards.append(-0.2)  # Penalty for errors

    # Log what we're returning
    logger.info(f"Returning rewards: {rewards}")
    return rewards

# --- Additional reward helper functions ---
def xml_tag_count_reward(completion_text):
    """Reward for having the correct XML tags in the completion"""
    expected_tags = [
        ('<think>', '</think>'),
        ('<answer>', '</answer>'),
        ('<confidence>', '</confidence>'),
        ('<entry_conditions>', '</entry_conditions>'),
        ('<exit_conditions>', '</exit_conditions>')
    ]

    reward = 0.0
    for open_tag, close_tag in expected_tags:
        if open_tag in completion_text and close_tag in completion_text:
            reward += 0.05

    return min(0.25, reward)  # Cap at 0.25

def format_reward(prediction):
    """Reward for having the correct format in parsed prediction"""
    format_reward = 0.0

    # Check format quality
    format_components = {
        'direction_parsed': prediction['direction'] is not None,
        'percentage_parsed': prediction['percentage'] is not None,
        'confidence_set': prediction['confidence'] != 0.5,  # 0.5 is the default
        'has_entry_conditions': bool(prediction.get('entry_conditions')),
        'has_exit_conditions': bool(prediction.get('exit_conditions'))
    }

    for key, present in format_components.items():
        if present:
            format_reward += 0.05

    return min(0.25, format_reward)  # Cap at 0.25

def direction_reward(prediction, actual_direction):
    """Reward for having the correct price direction prediction"""
    if prediction['direction'] == actual_direction:
        return 0.3  # Strong reward for correct direction
    elif prediction['direction'] is not None:
        return -0.1  # Small penalty for wrong direction
    return -0.2  # Larger penalty for missing direction

def strategy_coherence_reward(prediction):
    """Reward for having a coherent trading strategy"""
    coherence_score = 0.0

    # Check thinking presence
    has_thinking = '<think>' in prediction['full_response'] and '</think>' in prediction['full_response']
    if has_thinking:
        coherence_score += 0.05

        # Check for analysis structure in thinking
        thinking_content = re.search(r'<think>(.*?)</think>', prediction['full_response'], re.DOTALL)
        if thinking_content:
            text = thinking_content.group(1)
            if 'Key Factors:' in text and 'Analysis:' in text:
                coherence_score += 0.05

            # Check for numbered points
            if re.search(r'\d+\.\s+\w+', text):
                coherence_score += 0.05

    # Check for consistency between direction and confidence
    if prediction['direction'] is not None and prediction['confidence'] > 0.6:
        coherence_score += 0.05

    # Check for non-empty entry/exit conditions that make sense
    if prediction['entry_conditions'] and prediction['exit_conditions']:
        coherence_score += 0.05

    return min(0.2, coherence_score)  # Cap at 0.2

# === LOAD MODEL OPTIMIZED FOR A100 GPU ===
import os
import torch
from unsloth import FastLanguageModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Configuration ---
model_name = "Qwen/Qwen2.5-14B-Instruct"
lora_rank = 64                   # Increased for A100 GPU which has more memory
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Assuming training_args is defined elsewhere, create a placeholder if needed
class TrainingArgs:
    max_prompt_length = 1024
    max_completion_length = 512

training_args = TrainingArgs()

# --- Load Model and Tokenizer ---
logger.info(f"Loading model {model_name} optimized for A100 GPU...")

# Initialize model and tokenizer using more basic configuration
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=training_args.max_prompt_length + training_args.max_completion_length,
    dtype=torch.float16,  # Use float16 for better compatibility
    load_in_4bit=True,    # Stick with 4-bit which is better supported
    device_map="auto",
    # Removing Flash Attention 2 for now
)

# Add LoRA to the model with increased rank for A100
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=target_modules,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
)

# Configure tokenizer
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

logger.info(f"Model and tokenizer loaded successfully with optimizations for A100 GPU.")
logger.info(f"Model has {model.num_parameters()} parameters (excluding adapters)")

# Calculate adapter parameters manually instead
try:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {trainable_params}")
except Exception as e:
    logger.warning(f"Could not calculate trainable parameters: {e}")

# === TRAINING CODE OPTIMIZED FOR A100 GPU ===
import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import logging

# Configure logging if not already done
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training")

# --- Training Configuration ---
MODEL_OUTPUT_DIR = "/content/drive/MyDrive/A100_output"
BATCH_SIZE = 16             # A100 can handle larger batch sizes
GRADIENT_ACCUMULATION = 4   # Further increases effective batch size
MAX_GRAD_NORM = 0.3         # Helps prevent exploding gradients
LEARNING_RATE = 2e-4        # Starting learning rate for adapter training
WARMUP_RATIO = 0.03         # Percentage of steps for LR warmup
WEIGHT_DECAY = 0.01         # L2 regularization
NUM_EPOCHS = 3              # Number of training epochs
FP16 = False                # We're using bfloat16 instead
BF16 = True                 # Use bfloat16 training
DEEPSPEED = "stage3"        # Optimize memory usage and performance with DeepSpeed

# --- Dataset Loading and Processing ---
def load_and_process_data():
    logger.info("Loading and processing dataset...")

    # Load your dataset here - example with a public dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Format dataset for instruction tuning
    def format_instruction(example):
        return {
            "text": f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        }

    formatted_dataset = dataset.map(format_instruction)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=1536,  # Adjust based on your requirements
        )

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    logger.info(f"Dataset processed: {len(tokenized_dataset)} examples")
    return tokenized_dataset

# --- Setup Training ---
def setup_training(dataset):
    logger.info("Setting up training...")

    # Configure training arguments optimized for A100 GPU
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        fp16=FP16,
        bf16=BF16,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        deepspeed=DEEPSPEED,
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    return trainer

# === TRAINING CODE OPTIMIZED FOR A100 GPU ===
import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import logging

# Configure logging if not already done
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training")

# --- Training Configuration ---
MODEL_OUTPUT_DIR = "output"
BATCH_SIZE = 16             # A100 can handle larger batch sizes
GRADIENT_ACCUMULATION = 4   # Further increases effective batch size
MAX_GRAD_NORM = 0.3         # Helps prevent exploding gradients
LEARNING_RATE = 2e-4        # Starting learning rate for adapter training
WARMUP_RATIO = 0.03         # Percentage of steps for LR warmup
WEIGHT_DECAY = 0.01         # L2 regularization
NUM_EPOCHS = 3              # Number of training epochs
FP16 = False                # We're using bfloat16 instead
BF16 = True                 # Use bfloat16 training
# Remove the string "stage3" and use a proper dictionary instead
DEEPSPEED_CONFIG = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e6
    },
    "bf16": {
        "enabled": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    }
}

# --- Dataset Loading and Processing ---
def load_and_process_data():
    logger.info("Loading and processing dataset...")

    # Load your dataset here - example with a public dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Format dataset for instruction tuning
    def format_instruction(example):
        return {
            "text": f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        }

    formatted_dataset = dataset.map(format_instruction)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=1536,  # Adjust based on your requirements
        )

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    logger.info(f"Dataset processed: {len(tokenized_dataset)} examples")
    return tokenized_dataset

# --- Setup Training ---
def setup_training(dataset):
    logger.info("Setting up training...")

    # Configure training arguments optimized for A100 GPU
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        fp16=FP16,
        bf16=BF16,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        deepspeed=DEEPSPEED_CONFIG,  # Use the dictionary config
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    return trainer

# === TRAINING CODE OPTIMIZED FOR A100 GPU ===
import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import logging

# Configure logging if not already done
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training")

# --- Training Configuration ---
MODEL_OUTPUT_DIR = "output"
BATCH_SIZE = 16             # A100 can handle larger batch sizes
GRADIENT_ACCUMULATION = 4   # Further increases effective batch size
MAX_GRAD_NORM = 0.3         # Helps prevent exploding gradients
LEARNING_RATE = 2e-4        # Starting learning rate for adapter training
WARMUP_RATIO = 0.03         # Percentage of steps for LR warmup
WEIGHT_DECAY = 0.01         # L2 regularization
NUM_EPOCHS = 3              # Number of training epochs
FP16 = False                # We're using bfloat16 instead
BF16 = True                 # Use bfloat16 training
# REMOVE DEEPSPEED ENTIRELY

# --- Dataset Loading and Processing ---
def load_and_process_data():
    logger.info("Loading and processing dataset...")

    # Load your dataset here - example with a public dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Format dataset for instruction tuning
    def format_instruction(example):
        return {
            "text": f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        }

    formatted_dataset = dataset.map(format_instruction)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=1536,  # Adjust based on your requirements
        )

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    logger.info(f"Dataset processed: {len(tokenized_dataset)} examples")
    return tokenized_dataset

# --- Setup Training ---
def setup_training(dataset):
    logger.info("Setting up training...")

    # Configure training arguments optimized for A100 GPU WITHOUT DEEPSPEED
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        fp16=FP16,
        bf16=BF16,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        # REMOVED deepspeed parameter
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    return trainer

# === INFERENCE WITH FLASH ATTENTION 2 ===
import torch
from unsloth import FastLanguageModel
import logging
import os

# Configure logging if not already done
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

def format_prompt(text):
    # Format raw text into the chat template format
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

def load_inference_model(base_model="Qwen/Qwen2.5-14B-Instruct", adapter_path="output/lora_adapter"):
    logger.info(f"Loading base model: {base_model}")

    # Load base model with Flash Attention 2
    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model,
        dtype=torch.bfloat16,  # bfloat16 for A100
        device_map="auto",
        use_flash_attention_2=True,  # Enable Flash Attention 2
    )

    # Load trained adapter if available
    if os.path.exists(adapter_path):
        logger.info(f"Loading adapter from: {adapter_path}")
        model = FastLanguageModel.from_pretrained(
            model,
            adapter_path,
            device_map="auto",
        )

        # Prepare model for generation
        model = FastLanguageModel.get_peft_model(
            model,
            transition_config={"integration_type": "peft"}
        )
    else:
        logger.warning(f"Adapter path {adapter_path} not found. Using base model only.")

    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50):
    formatted_prompt = format_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    logger.info("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    parts = generated_text.split("<|im_start|>assistant\n")
    if len(parts) > 1:
        response = parts[1].split("<|im_end|>")[0] if "<|im_end|>" in parts[1] else parts[1]
        return response.strip()
    return generated_text.strip()

# === DEMO INFERENCE WITH THE MODEL ===

# Load the model for inference
inference_model, inference_tokenizer = load_inference_model()

# Example prompt
demo_prompt = "Write a short poem about artificial intelligence and creativity."

# Generate text
response = generate_text(
    inference_model,
    inference_tokenizer,
    demo_prompt,
    max_new_tokens=200,
    temperature=0.8
)

print(f"Prompt: {demo_prompt}\n")
print(f"Response:\n{response}")

# === LOAD MODEL OPTIMIZED FOR L4 GPU ===
import os
import torch
from unsloth import FastLanguageModel

# --- Model Configuration ---
model_name = "Qwen/Qwen2.5-14B-Instruct"  # Updated to Qwen2.5 as shown in your error
lora_rank = 32                   # Reduced from 64 for L4 GPU
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- Load Model and Tokenizer ---
logger.info(f"Loading model {model_name} optimized for L4 GPU...")

# Initialize model and tokenizer using Unsloth with memory optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=training_args.max_prompt_length + training_args.max_completion_length,
    dtype=torch.float16,  # Use fp16 for L4
    load_in_4bit=True,    # Use 4-bit quantization
    device_map="auto",
)

# Add LoRA to the model with reduced rank
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=target_modules,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

# Configure tokenizer
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

logger.info(f"Model and tokenizer loaded successfully with optimizations for L4 GPU.")
logger.info(f"Model has {model.num_parameters()} parameters (excluding adapters)")

# Calculate adapter parameters manually instead
try:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {trainable_params}")
except Exception as e:
    logger.warning(f"Could not calculate trainable parameters: {e}")

# === DATA FORMATTER AND PROMPT CREATION ===
import re
from transformers.tokenization_utils_base import BatchEncoding

def format_price_data(dataset_item):
    """Format a single data item into a prompt for the model"""
    ticker = dataset_item.get('ticker', 'UNKNOWN')
    datetime_str = dataset_item.get('datetime_str', 'UNKNOWN')

    # Format price data
    price_data = {
        'Open': dataset_item.get('Open', 0),
        'High': dataset_item.get('High', 0),
        'Low': dataset_item.get('Low', 0),
        'Close': dataset_item.get('Close', 0),
        'Volume': dataset_item.get('Volume', 0),
    }

    # Format technical indicators
    tech_indicators = {
        'MA4': dataset_item.get('MA4', 0),
        'MA8': dataset_item.get('MA8', 0),
        'MA20': dataset_item.get('MA20', 0),
        'RSI': dataset_item.get('RSI', 0),
        'MACD': dataset_item.get('MACD', 0),
        'MACD_Signal': dataset_item.get('MACD_Signal', 0),
        'MACD_Hist': dataset_item.get('MACD_Hist', 0),
        'BB_Upper': dataset_item.get('BB_Upper', 0),
        'BB_Lower': dataset_item.get('BB_Lower', 0),
    }

    # Format current price and volatility
    current_price = dataset_item.get('current_price', 0)
    volatility = dataset_item.get('volatility', 0)

    # Create prompt
    prompt = f"""You are a sophisticated trading assistant. Analyze the data below and determine whether the stock price will go UP or DOWN in the short term.

TICKER: {ticker}
DATE: {datetime_str}

PRICE DATA:
- Open: ${price_data['Open']:.2f}
- High: ${price_data['High']:.2f}
- Low: ${price_data['Low']:.2f}
- Close: ${price_data['Close']:.2f}
- Volume: {price_data['Volume']:.0f}

TECHNICAL INDICATORS:
- 4-Day Moving Average: ${tech_indicators['MA4']:.2f}
- 8-Day Moving Average: ${tech_indicators['MA8']:.2f}
- 20-Day Moving Average: ${tech_indicators['MA20']:.2f}
- RSI (14): {tech_indicators['RSI']:.2f}
- MACD: {tech_indicators['MACD']:.4f}
- MACD Signal: {tech_indicators['MACD_Signal']:.4f}
- MACD Histogram: {tech_indicators['MACD_Hist']:.4f}
- Bollinger Upper: ${tech_indicators['BB_Upper']:.2f}
- Bollinger Lower: ${tech_indicators['BB_Lower']:.2f}

CURRENT PRICE: ${current_price:.2f}
VOLATILITY (30-day): {volatility:.2f}%

Analyze this data and provide:
1. Your thought process with key factors and analysis
2. Your prediction (UP or DOWN) with confidence level (0-100%)
3. Entry conditions to confirm your trade
4. Exit conditions including stop loss

Format your response using these XML tags:
<think>Your detailed analysis here...</think>
<answer>UP or DOWN</answer>
<confidence>0-100</confidence>
<entry_conditions>List your entry conditions</entry_conditions>
<exit_conditions>List your exit conditions</exit_conditions>
"""
    return prompt

def prepare_training_dataset(dataset):
    """Prepare dataset for GRPO training"""
    formatted_dataset = []

    for item in dataset:
        # Create the prompt
        prompt = format_price_data(item)

        # Add to formatted dataset with all metadata preserved
        formatted_item = {
            'prompt': prompt,  # The formatted prompt
        }

        # Copy all original metadata fields
        for key in item:
            if key != 'prompt':  # Don't overwrite the prompt
                formatted_item[key] = item[key]

        formatted_dataset.append(formatted_item)

    return formatted_dataset

# Create the formatted dataset
logger.info("Preparing training data...")
formatted_train_data = prepare_training_dataset(train_dataset)
train_dataset = Dataset.from_list(formatted_train_data)

logger.info(f"Training dataset prepared with {len(train_dataset)} examples")
logger.info(f"Sample prompt:\n{train_dataset[0]['prompt'][:500]}...")

# === HELPER FUNCTIONS FOR REWARD CALCULATION ===
import re
import numpy as np

class TradeManager:
    """Manages trade simulation and risk management"""
    def __init__(self, stop_loss_pct=0.02, take_profit_pct=0.03, max_holding_periods=5):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_periods = max_holding_periods

    def simulate_trade(self, entry_price, future_prices, direction):
        """Simulate a trade with stop loss and take profit"""
        if direction not in ['UP', 'DOWN']:
            return 0, 0, "Invalid direction"

        # Set initial values
        position_multiplier = 1 if direction == 'UP' else -1
        stop_loss_price = entry_price * (1 - position_multiplier * self.stop_loss_pct)
        take_profit_price = entry_price * (1 + position_multiplier * self.take_profit_pct)

        # Check each future price point
        exit_reason = "Max holding period reached"
        exit_price = future_prices[-1] if len(future_prices) > 0 else entry_price
        exit_idx = len(future_prices) - 1

        for i, price in enumerate(future_prices):
            # Check stop loss
            if (direction == 'UP' and price <= stop_loss_price) or \
               (direction == 'DOWN' and price >= stop_loss_price):
                exit_price = stop_loss_price
                exit_reason = "Stop loss triggered"
                exit_idx = i
                break

            # Check take profit
            if (direction == 'UP' and price >= take_profit_price) or \
               (direction == 'DOWN' and price <= take_profit_price):
                exit_price = take_profit_price
                exit_reason = "Take profit triggered"
                exit_idx = i
                break

            # Check max holding period
            if i >= self.max_holding_periods - 1:
                exit_price = price
                exit_reason = "Max holding period reached"
                exit_idx = i
                break

        # Calculate PnL
        pnl_pct = position_multiplier * (exit_price - entry_price) / entry_price
        return pnl_pct, exit_idx + 1, exit_reason

def parse_trade_prediction(response_text):
    """Parse model's output into structured prediction data"""
    # Initialize with default values
    prediction = {
        'direction': None,
        'percentage': None,
        'confidence': 0.5,
        'entry_conditions': [],
        'exit_conditions': []
    }

    # Extract answer (UP or DOWN)
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip().upper()
        if answer in ['UP', 'DOWN']:
            prediction['direction'] = answer

    # Extract confidence (0-100)
    confidence_match = re.search(r'<confidence>(.*?)</confidence>', response_text, re.DOTALL | re.IGNORECASE)
    if confidence_match:
        try:
            confidence_value = float(confidence_match.group(1).strip())
            # Normalize to 0-1 range
            if 0 <= confidence_value <= 100:
                prediction['confidence'] = confidence_value / 100
            elif 0 <= confidence_value <= 1:
                prediction['confidence'] = confidence_value
        except ValueError:
            # Keep default if we can't parse the value
            pass

    # Extract entry conditions
    entry_match = re.search(r'<entry_conditions>(.*?)</entry_conditions>', response_text, re.DOTALL | re.IGNORECASE)
    if entry_match:
        entry_text = entry_match.group(1).strip()
        # Split on newlines or bullet points
        entries = [item.strip() for item in re.split(r'[\n\r]+|•|\*|\-|\d+\.', entry_text) if item.strip()]
        prediction['entry_conditions'] = entries

    # Extract exit conditions
    exit_match = re.search(r'<exit_conditions>(.*?)</exit_conditions>', response_text, re.DOTALL | re.IGNORECASE)
    if exit_match:
        exit_text = exit_match.group(1).strip()
        # Split on newlines or bullet points
        exits = [item.strip() for item in re.split(r'[\n\r]+|•|\*|\-|\d+\.', exit_text) if item.strip()]
        prediction['exit_conditions'] = exits

    # Look for percentage in the prediction (optional)
    percentage_pattern = r'(\d+(?:\.\d+)?)%'
    percentage_matches = re.findall(percentage_pattern, response_text)
    if percentage_matches:
        try:
            prediction['percentage'] = float(percentage_matches[0]) / 100
        except (ValueError, IndexError):
            pass

    return prediction

def calculate_trade_reward(prediction, metadata, trade_manager):
    """Calculate reward based on trade performance"""
    # Extract values needed for simulation
    direction = prediction.get('direction')
    confidence = prediction.get('confidence', 0.5)
    entry_price = metadata.get('current_price', 0)
    future_prices = metadata.get('future_prices', [])
    actual_direction = metadata.get('actual_direction')

    # Check if we have valid data for simulation
    if not direction or not future_prices or entry_price <= 0:
        return 0.0, {'error': 'Invalid simulation data'}

    # Convert list if needed
    if isinstance(future_prices, str):
        try:
            future_prices = [float(p) for p in future_prices.strip('[]').split(',')]
        except:
            future_prices = []

    # Simulate the trade
    pnl_pct, holding_periods, exit_reason = trade_manager.simulate_trade(
        entry_price=entry_price,
        future_prices=future_prices,
        direction=direction
    )

    # Base reward components
    direction_match = 1.0 if direction == actual_direction else -0.5

    # PnL component (scaled between -1 and 1)
    pnl_component = np.clip(pnl_pct * 10, -1.0, 1.0)

    # Risk management component
    risk_mgmt = 0.0
    if exit_reason == "Take profit triggered":
        risk_mgmt = 0.2  # Reward for hitting take profit
    elif exit_reason == "Stop loss triggered":
        risk_mgmt = -0.1  # Small penalty for hitting stop loss
    elif holding_periods < len(future_prices):
        risk_mgmt = 0.1  # Some reward for proper exit

    # Confidence alignment component
    confidence_alignment = 0.0
    if pnl_pct > 0 and confidence > 0.5:
        confidence_alignment = 0.1  # Reward for being confident when right
    elif pnl_pct < 0 and confidence > 0.7:
        confidence_alignment = -0.1  # Penalty for being too confident when wrong
    elif pnl_pct < 0 and confidence < 0.6:
        confidence_alignment = 0.05  # Small reward for appropriate uncertainty

    # Combine the components
    total_reward = (
        0.3 * direction_match +  # Direction match is most important
        0.4 * pnl_component +    # PnL is also very important
        0.2 * risk_mgmt +        # Risk management matters
        0.1 * confidence_alignment  # Confidence alignment is a bonus
    )

    # Metrics for logging
    metrics = {
        'pnl_pct': pnl_pct,
        'exit_reason': exit_reason,
        'holding_periods': holding_periods,
        'direction_match': direction_match,
        'pnl_component': pnl_component,
        'risk_mgmt': risk_mgmt,
        'confidence_alignment': confidence_alignment,
        'total_reward': total_reward
    }

    return total_reward, metrics

# === MISSING TRADE MANAGER INITIALIZATION ===
import logging
import re
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class TradeManager:
    """Manages trade simulation and risk management"""
    def __init__(self, stop_loss_pct=0.02, take_profit_pct=0.03, max_holding_periods=5):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_periods = max_holding_periods

    def simulate_trade(self, entry_price, future_prices, direction):
        """Simulate a trade with stop loss and take profit"""
        if direction not in ['UP', 'DOWN']:
            return 0, 0, "Invalid direction"

        # Set initial values
        position_multiplier = 1 if direction == 'UP' else -1
        stop_loss_price = entry_price * (1 - position_multiplier * self.stop_loss_pct)
        take_profit_price = entry_price * (1 + position_multiplier * self.take_profit_pct)

        # Check each future price point
        exit_reason = "Max holding period reached"
        exit_price = future_prices[-1] if len(future_prices) > 0 else entry_price
        exit_idx = len(future_prices) - 1

        for i, price in enumerate(future_prices):
            # Check stop loss
            if (direction == 'UP' and price <= stop_loss_price) or \
               (direction == 'DOWN' and price >= stop_loss_price):
                exit_price = stop_loss_price
                exit_reason = "Stop loss triggered"
                exit_idx = i
                break

            # Check take profit
            if (direction == 'UP' and price >= take_profit_price) or \
               (direction == 'DOWN' and price <= take_profit_price):
                exit_price = take_profit_price
                exit_reason = "Take profit triggered"
                exit_idx = i
                break

            # Check max holding period
            if i >= self.max_holding_periods - 1:
                exit_price = price
                exit_reason = "Max holding period reached"
                exit_idx = i
                break

        # Calculate PnL
        pnl_pct = position_multiplier * (exit_price - entry_price) / entry_price
        return pnl_pct, exit_idx + 1, exit_reason

def parse_trade_prediction(response_text):
    """Parse model's output into structured prediction data"""
    # Initialize with default values
    prediction = {
        'direction': None,
        'percentage': None,
        'confidence': 0.5,
        'entry_conditions': [],
        'exit_conditions': []
    }

    # Extract answer (UP or DOWN)
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip().upper()
        if answer in ['UP', 'DOWN']:
            prediction['direction'] = answer

    # Extract confidence (0-100)
    confidence_match = re.search(r'<confidence>(.*?)</confidence>', response_text, re.DOTALL | re.IGNORECASE)
    if confidence_match:
        try:
            confidence_value = float(confidence_match.group(1).strip())
            # Normalize to 0-1 range
            if 0 <= confidence_value <= 100:
                prediction['confidence'] = confidence_value / 100
            elif 0 <= confidence_value <= 1:
                prediction['confidence'] = confidence_value
        except ValueError:
            # Keep default if we can't parse the value
            pass

    # Extract entry conditions
    entry_match = re.search(r'<entry_conditions>(.*?)</entry_conditions>', response_text, re.DOTALL | re.IGNORECASE)
    if entry_match:
        entry_text = entry_match.group(1).strip()
        # Split on newlines or bullet points
        entries = [item.strip() for item in re.split(r'[\n\r]+|•|\*|\-|\d+\.', entry_text) if item.strip()]
        prediction['entry_conditions'] = entries

    # Extract exit conditions
    exit_match = re.search(r'<exit_conditions>(.*?)</exit_conditions>', response_text, re.DOTALL | re.IGNORECASE)
    if exit_match:
        exit_text = exit_match.group(1).strip()
        # Split on newlines or bullet points
        exits = [item.strip() for item in re.split(r'[\n\r]+|•|\*|\-|\d+\.', exit_text) if item.strip()]
        prediction['exit_conditions'] = exits

    # Look for percentage in the prediction (optional)
    percentage_pattern = r'(\d+(?:\.\d+)?)%'
    percentage_matches = re.findall(percentage_pattern, response_text)
    if percentage_matches:
        try:
            prediction['percentage'] = float(percentage_matches[0]) / 100
        except (ValueError, IndexError):
            pass

    return prediction

def calculate_trade_reward(prediction, metadata, trade_manager):
    """Calculate reward based on trade performance"""
    # Extract values needed for simulation
    direction = prediction.get('direction')
    confidence = prediction.get('confidence', 0.5)
    entry_price = metadata.get('current_price', 0)
    future_prices = metadata.get('future_prices', [])
    actual_direction = metadata.get('actual_direction')

    # Check if we have valid data for simulation
    if not direction or not future_prices or entry_price <= 0:
        return 0.0, {'error': 'Invalid simulation data'}

    # Convert list if needed
    if isinstance(future_prices, str):
        try:
            future_prices = [float(p) for p in future_prices.strip('[]').split(',')]
        except:
            future_prices = []

    # Simulate the trade
    pnl_pct, holding_periods, exit_reason = trade_manager.simulate_trade(
        entry_price=entry_price,
        future_prices=future_prices,
        direction=direction
    )

    # Base reward components
    direction_match = 1.0 if direction == actual_direction else -0.5

    # PnL component (scaled between -1 and 1)
    pnl_component = np.clip(pnl_pct * 10, -1.0, 1.0)

    # Risk management component
    risk_mgmt = 0.0
    if exit_reason == "Take profit triggered":
        risk_mgmt = 0.2  # Reward for hitting take profit
    elif exit_reason == "Stop loss triggered":
        risk_mgmt = -0.1  # Small penalty for hitting stop loss
    elif holding_periods < len(future_prices):
        risk_mgmt = 0.1  # Some reward for proper exit

    # Confidence alignment component
    confidence_alignment = 0.0
    if pnl_pct > 0 and confidence > 0.5:
        confidence_alignment = 0.1  # Reward for being confident when right
    elif pnl_pct < 0 and confidence > 0.7:
        confidence_alignment = -0.1  # Penalty for being too confident when wrong
    elif pnl_pct < 0 and confidence < 0.6:
        confidence_alignment = 0.05  # Small reward for appropriate uncertainty

    # Combine the components
    total_reward = (
        0.3 * direction_match +  # Direction match is most important
        0.4 * pnl_component +    # PnL is also very important
        0.2 * risk_mgmt +        # Risk management matters
        0.1 * confidence_alignment  # Confidence alignment is a bonus
    )

    # Metrics for logging
    metrics = {
        'pnl_pct': pnl_pct,
        'exit_reason': exit_reason,
        'holding_periods': holding_periods,
        'direction_match': direction_match,
        'pnl_component': pnl_component,
        'risk_mgmt': risk_mgmt,
        'confidence_alignment': confidence_alignment,
        'total_reward': total_reward
    }

    return total_reward, metrics

# Create the TradeManager instance here to fix the missing reference
trade_manager_instance = TradeManager(
    stop_loss_pct=0.02,
    take_profit_pct=0.03,
    max_holding_periods=5
)

logger.info("TradeManager initialized with stop_loss=2%, take_profit=3%, max_holding_periods=5")

# === EXECUTE TRAINING WITH FINANCIAL DATA ===
import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training")

# ---- Load Dataset ----
dataset_path = "/content/drive/MyDrive/Big_Data/GRPO_PnL_Trainer.jsonl"
logger.info(f"Loading dataset from: {dataset_path}")

# Load data
raw_dataset = load_dataset('json', data_files=dataset_path)['train']

# Limit dataset size for faster training during development
max_samples = 2000
if len(raw_dataset) > max_samples:
    logger.info(f"Limiting dataset to {max_samples} samples (from {len(raw_dataset)})")
    raw_dataset = raw_dataset.select(range(max_samples))

logger.info(f"Dataset loaded with {len(raw_dataset)} samples")

# ---- Format Dataset ----
def format_financial_prompt(example):
    # Create structured prompt from financial data
    prompt = f"""Analyze the following stock data for {example['ticker']} in the {example['sector']} sector:
Date: {example['datetime_str']}
Price: Open=${example['Open']:.2f}, High=${example['High']:.2f}, Low=${example['Low']:.2f}, Close=${example['Close']:.2f}
Volume: {int(example['Volume'])}
Technical Indicators:
- MA4: ${example['MA4']:.2f}
- MA8: ${example['MA8']:.2f}
- MA20: ${example['MA20']:.2f}
- RSI: {example['RSI']:.2f}
- MACD: {example['MACD']:.2f}, Signal: {example['MACD_Signal']:.2f}, Histogram: {example['MACD_Hist']:.2f}
- Bollinger Bands: Upper=${example['BB_Upper']:.2f}, Lower=${example['BB_Lower']:.2f}
- Price Change: ${example['Price_Change']:.2f} ({example['Pct_Change']:.2f}%)
- Current Price: ${example['current_price']:.2f}
- Volatility: {example['volatility']:.4f}

Based on this information, predict the price direction and percentage change."""

    # Create completion using actual direction and percentage
    completion = f"""After analyzing the data for {example['ticker']}, I predict the price will move {example['actual_direction']} by approximately {abs(example['actual_percentage']):.2f}%.

My analysis:
{example['thinking_trace']}"""

    # Format in chat template
    formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"

    return {"text": formatted_text}

# Apply formatting
logger.info("Formatting dataset for financial analysis...")
formatted_dataset = raw_dataset.map(format_financial_prompt)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=1536,  # Adjust based on your data
    )

logger.info("Tokenizing dataset...")
tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[col for col in raw_dataset.column_names] + ["text"]
)

logger.info(f"Dataset processed with {len(tokenized_dataset)} examples")

# ---- Configure Training ----
# Training parameters
output_dir = "/content/drive/MyDrive/financial_model_output"
os.makedirs(output_dir, exist_ok=True)

# Configure training arguments with only supported parameters
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,           # Reduced batch size
    gradient_accumulation_steps=8,           # Increased gradient accumulation
    learning_rate=2e-5,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    weight_decay=0.01,
    num_train_epochs=3,
    fp16=True,                               # Use fp16 instead of bf16
    bf16=False,                              # Disable bf16
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
    attn_implementation="eager"              # Use standard attention implementation
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ---- Initialize and Run Trainer ----
logger.info("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Execute training
logger.info("Starting training...")
trainer.train()

# Save trained model and adapter
logger.info("Training complete. Saving model...")
trainer.save_model(os.path.join(output_dir, "final_model"))

# Save LoRA adapter separately
logger.info("Saving LoRA adapter...")
model.save_pretrained(os.path.join(output_dir, "final_adapter"))

logger.info("Model training and saving complete!")

# === LOAD MODEL FROM SPECIFIED CHECKPOINT ===
import os
import torch
from unsloth import FastLanguageModel
from peft import PeftModel, PeftConfig
import logging

logger = logging.getLogger(__name__)

# --- Checkpoint Configuration ---
checkpoint_path = "/content/drive/My Drive/Colab_Stock_Prediction/Qwen2.5_14BSFT_more_significant_movement_traces_8bit/final_adapter"

logger.info(f"Loading adapter configuration from: {checkpoint_path}")

# --- Load Adapter Configuration to get base model name ---
try:
    peft_config = PeftConfig.from_pretrained(checkpoint_path)
    base_model_name = peft_config.base_model_name_or_path
    logger.info(f"Base model identified from config: {base_model_name}")
except Exception as e:
    logger.error(f"Failed to load PeftConfig from {checkpoint_path}: {e}")
    # Fallback or raise error if needed
    base_model_name = "unsloth/Qwen2.5-14B-Instruct" # Provide a default if config load fails
    logger.warning(f"Using fallback base model name: {base_model_name}")

# --- Load Base Model ---
logger.info(f"Loading base model: {base_model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=training_args.max_prompt_length + training_args.max_completion_length,
    dtype=torch.float16,  # Use fp16 for L4/A100
    load_in_4bit=True,    # Use 4-bit quantization
    device_map="auto",    # Let accelerate handle device mapping
    # Removed max_memory constraint to let accelerate manage more freely
)

# --- Load the PEFT Adapter ---
logger.info(f"Loading adapter weights from {checkpoint_path} onto the base model")
try:
    # Use PeftModel.from_pretrained to load the adapter correctly
    # Set is_trainable=True to ensure we can continue training
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    logger.info(f"Successfully loaded adapter from {checkpoint_path}")

    # Merge adapter for faster inference if needed later, but keep separate for training
    # model = model.merge_and_unload() # Uncomment for inference only

except Exception as e:
    logger.error(f"Failed to load PEFT adapter from {checkpoint_path}: {e}")
    logger.error("Ensure the checkpoint path is correct and contains adapter_model.bin or adapter_model.safetensors")
    raise  # Re-raise the exception as we cannot proceed

# --- Configure Tokenizer ---
tokenizer.padding_side = "left"
# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Set tokenizer pad_token to eos_token: {tokenizer.pad_token}")

# Ensure chat template is set (important for Qwen models)
if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    logger.info("Applied default Qwen chat template to tokenizer.")

# --- Verify Trainable Parameters ---
# Use the built-in method for clarity
model.print_trainable_parameters()

# Or manually count for logging
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Manually counted trainable parameters: {trainable_params}")

if trainable_params == 0:
    logger.error("CRITICAL: No trainable parameters found after loading the adapter!")
    raise ValueError("Model loaded from checkpoint has no trainable parameters.")
else:
    logger.info("Model loaded with trainable parameters. Ready for further training.")

# === SAVE CONFIGURATION ===
import json
import os
from datetime import datetime

# --- Use the captured training_args object (assuming it's named global_training_args) ---
if 'global_training_args' in globals() and global_training_args is not None:
    logger.info("Saving configuration using global_training_args...")

    # Attempt to get LoRA config from the model object if it exists
    lora_rank_val = "N/A"
    target_modules_val = "N/A"
    if 'model' in globals() and hasattr(model, 'peft_config') and 'default' in model.peft_config:
        lora_rank_val = getattr(model.peft_config['default'], 'r', "N/A")
        target_modules_val = getattr(model.peft_config['default'], 'target_modules', "N/A")

    # Attempt to get dataset info if available
    dataset_path_val = "N/A"
    num_samples_val = "N/A"
    if 'train_dataset' in globals() and hasattr(train_dataset, '_data_files') and train_dataset._data_files:
        # Assuming the path is stored in the dataset object's _data_files attribute
        try:
            dataset_path_val = train_dataset._data_files[0]['origin_path']
        except (KeyError, IndexError, TypeError):
            logger.warning("Could not determine dataset path from train_dataset object.")
    if 'train_dataset' in globals():
        try:
            num_samples_val = len(train_dataset)
        except TypeError:
             logger.warning("Could not determine number of samples from train_dataset object.")

    # Configuration settings used during training
    config_to_save = {
        "model_name": global_training_args.model_name_or_path if hasattr(global_training_args, 'model_name_or_path') else "N/A",
        "lora_rank": lora_rank_val,
        "target_modules": target_modules_val,
        "training_args": {
            "output_dir": global_training_args.output_dir,
            "learning_rate": global_training_args.learning_rate,
            "per_device_train_batch_size": global_training_args.per_device_train_batch_size,
            "num_generations": global_training_args.num_generations,
            "gradient_accumulation_steps": global_training_args.gradient_accumulation_steps,
            "max_prompt_length": global_training_args.max_prompt_length,
            "max_completion_length": global_training_args.max_completion_length,
            "max_steps": global_training_args.max_steps,
            "bf16": global_training_args.bf16,
            "fp16": global_training_args.fp16,
            "use_vllm": getattr(global_training_args, 'use_vllm', False), # Check if use_vllm exists
        },
        "dataset": {
            "path": dataset_path_val,
            "num_samples": num_samples_val,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save the configuration to a JSON file
    config_file = os.path.join(global_training_args.output_dir, "training_config.json")
    try:
        os.makedirs(global_training_args.output_dir, exist_ok=True) # Ensure directory exists
        with open(config_file, "w") as f:
            json.dump(config_to_save, f, indent=2)
        logger.info(f"Configuration saved to {config_file}")
    except Exception as e:
         logger.error(f"Error saving configuration: {e}")

else:
    logger.warning("Global training arguments (`global_training_args`) not found. Cannot save configuration.")

logger.info("Configuration saving section finished.")