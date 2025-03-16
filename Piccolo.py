#!/usr/bin/env python3
# Namek Project - Stage 2 GRPO Training Script with Balanced Dataset
# Owner: ./install_AI

import os
import sys
import math
import json
import torch
import random
import logging
import argparse
import numpy as np
import re
from typing import List, Dict, Any, Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import GRPOConfig
from Over9Thousand import GRPOTrainer, MarketConditions, ConfidenceCalibrator
from torch.utils.data import DataLoader
import outlines
from enum import Enum
from pydantic import BaseModel
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Piccolo")

class AccuracyMonitorCallback(TrainerCallback):
    """Callback for monitoring prediction accuracy during training"""
    def __init__(self):
        self.correct_predictions = 0
        self.total_predictions = 0
        self.history = []
        
    def on_step_end(self, args: GRPOConfig, state: TrainerState, control: TrainerControl, **kwargs):
        if "metrics" in kwargs and "up_down_accuracy" in kwargs["metrics"]:
            accuracy = kwargs["metrics"]["up_down_accuracy"]
            self.history.append(accuracy)
            self.correct_predictions += accuracy[0]  # Assuming accuracy is [correct, total]
            self.total_predictions += accuracy[1]
            
            # Print current accuracy
            current_accuracy = self.correct_predictions / max(1, self.total_predictions)
            print(f"\nCurrent UP/DOWN prediction accuracy: {current_accuracy:.4f} ({self.correct_predictions}/{self.total_predictions})")

class StockDirection(str, Enum):
    up = "up"
    down = "down"

class StockPrediction(BaseModel):
    analysis: str  # Will contain the <think>...</think> section
    prediction: StockDirection

def intermediate_up_down_reward_func(prompts, completions, actual_percentages=None, **kwargs):
    """
    Intermediate reward function for stage 2 that focuses on prediction accuracy
    with more lenient rewards to encourage learning.
    """
    rewards = []
    up_down_accuracy = []
    correct_count = 0
    total_count = 0
    
    for i, completion in enumerate(completions):
        # Extract the completion text
        if isinstance(completion, dict) and 'content' in completion:
            extracted_text = completion['content']
        elif isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
            extracted_text = completion[0].get('content', '')
        else:
            extracted_text = str(completion)
        
        # Get actual percentage for this example if available
        actual_pct = None
        if actual_percentages and i < len(actual_percentages):
            actual_pct = actual_percentages[i]
        
        # Print the completion for debugging
        print(f"\nCompletion {i}:")
        print(f"Extracted text: {extracted_text[:100]}...")
        if actual_pct:
            print(f"Actual percentage change: {actual_pct}%")
        
        # Initial reward
        reward = 0.0
        
        # Check for decisive up or down
        up_pattern = re.compile(r'<answer>.*?direction:\s*up.*?change:\s*(\d+\.?\d*)%.*?</answer>', re.DOTALL | re.IGNORECASE)
        down_pattern = re.compile(r'<answer>.*?direction:\s*down.*?change:\s*(\d+\.?\d*)%.*?</answer>', re.DOTALL | re.IGNORECASE)
        
        # Check for thinking tag
        thinking_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        
        # Extract direction and percentage
        up_match = up_pattern.search(extracted_text)
        down_match = down_pattern.search(extracted_text)
        
        # Determine decision and percentage
        is_up = up_match is not None
        is_down = down_match is not None
        percentage = None
        
        if is_up:
            percentage = float(up_match.group(1)) if up_match else None
        elif is_down:
            percentage = float(down_match.group(1)) if down_match else None
        
        # Track up/down accuracy and apply rewards/penalties
        is_correct_prediction = False
        
        if is_up and actual_pct is not None:
            total_count += 1
            # UP was correct if price went up
            if actual_pct > 0:
                up_down_accuracy.append("Correct")
                correct_count += 1
                # Higher reward for correct prediction in stage 2
                reward = 3.5  # Slightly higher than stage 1 but lower than stage 3
                print(f"Correct UP prediction (price went up {actual_pct}%): +3.5")
                is_correct_prediction = True
            else:
                up_down_accuracy.append("Wrong")
                # Milder penalty for wrong prediction in stage 2
                reward = -1.5  # Less severe than stage 3
                print(f"Wrong UP prediction (price went down {actual_pct}%): -1.5")
                is_correct_prediction = False
        elif is_down and actual_pct is not None:
            total_count += 1
            # DOWN was correct if price went down
            if actual_pct < 0:
                up_down_accuracy.append("Correct")
                correct_count += 1
                # Higher reward for correct prediction in stage 2
                reward = 3.5  # Slightly higher than stage 1 but lower than stage 3
                print(f"Correct DOWN prediction (price went down {actual_pct}%): +3.5")
                is_correct_prediction = True
            else:
                up_down_accuracy.append("Wrong")
                # Milder penalty for wrong prediction in stage 2
                reward = -1.5  # Less severe than stage 3
                print(f"Wrong DOWN prediction (price went up {actual_pct}%): -1.5")
                is_correct_prediction = False
        else:
            up_down_accuracy.append("Invalid")
            # Penalty for invalid/missing prediction in stage 2
            reward = -2.5  # Less severe than stage 3
            print("Invalid or missing up/down prediction: -2.5")
            is_correct_prediction = False
        
        # Reward for including proper tags - moderately important in stage 2
        if thinking_pattern.search(extracted_text):
            reward += 0.4  # More than stage 1 but less than stage 3
            print("Thinking tag found: +0.4")
        else:
            reward -= 0.4
            print("Missing thinking tag: -0.4")
        
        # Reward for percentage prediction accuracy - moderate importance in stage 2
        if is_correct_prediction and percentage is not None and actual_pct is not None:
            pct_diff = abs(percentage - abs(actual_pct))
            if pct_diff < 0.5:  # Very accurate
                reward += 1.0
                print(f"Very accurate percentage prediction (within 0.5%): +1.0")
            elif pct_diff < 1.0:  # Reasonably accurate
                reward += 0.5
                print(f"Reasonably accurate percentage prediction (within 1.0%): +0.5")
            elif pct_diff > 5.0:  # Very inaccurate
                reward -= 0.5
                print(f"Very inaccurate percentage prediction (off by {pct_diff:.2f}%): -0.5")
        
        # Format compliance rewards - moderate importance in stage 2
        format_score = 0.0
        
        # Check for key factors section
        key_factors_pattern = re.compile(r'Key\s+Factors:', re.IGNORECASE)
        if key_factors_pattern.search(extracted_text):
            format_score += 0.3
            print("Format reward: Has Key Factors section: +0.3")
        
        # Check for analysis section
        analysis_pattern = re.compile(r'Analysis:', re.IGNORECASE)
        if analysis_pattern.search(extracted_text):
            format_score += 0.3
            print("Format reward: Has Analysis section: +0.3")
        
        # Check for detailed thinking
        thinking_match = thinking_pattern.search(extracted_text)
        if thinking_match:
            thinking_text = thinking_match.group(1)
            if len(thinking_text.split()) < 50:  # Short thinking section
                format_score -= 0.3
                print("Format penalty: Thinking section too short: -0.3")
            elif len(thinking_text.split()) > 150:  # Detailed thinking section
                format_score += 0.3
                print("Format reward: Detailed thinking section: +0.3")
        
        # Add format score to reward
        reward += format_score
        
        # Ensure reward is within reasonable bounds
        reward = max(-5.0, min(5.0, reward))
        
        rewards.append(reward)
    
    # Return rewards and accuracy metrics
    if total_count > 0:
        accuracy_percentage = (correct_count / total_count) * 100
        print(f"\nBatch accuracy: {accuracy_percentage:.2f}% ({correct_count}/{total_count})")
    
    return rewards, [correct_count, total_count]

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 GRPO training for UP/DOWN predictions with balanced dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name or path")
    parser.add_argument("--dataset_path", type=str, default="2084Collective/deepstock-sp500-companies-with-info-and-user-prompt", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="outputs_namek_stage2", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of training steps")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--use_pretrained_checkpoint", type=str, default=None, help="Path to pretrained checkpoint to continue training from")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def format_dataset_for_model(dataset):
    """Format the dataset for training, including metadata for actual percentage changes."""
    system_message = """You are an advanced AI stonk market analyst specialized in pattern recognition and quantitative analysis.
Your task is to analyze financial data and predict whether a stonk will go UP or DOWN, along with an estimated percentage change.

You MUST use EXACTLY this format:

<think>
Key Factors:
1. [First key observation]
2. [Second key observation]
3. [Third key observation]

Analysis:
[Your detailed analysis of the technical indicators, financial metrics, and market sentiment.
Consider both bullish and bearish signals before making your final decision.]
</think>

<answer>direction: up change: X.X%</answer> OR <answer>direction: down change: X.X%</answer>

Example of correct format:
<think>
Key Factors:
1. RSI at 65.2 indicates strong momentum but not overbought
2. Revenue growth of 15.7% exceeds industry average
3. Recent positive news about product launches

Analysis:
The technical indicators show positive momentum with RSI at 65.2, suggesting strong buying pressure without being overbought. The MACD line crossing above the signal line confirms this upward trend.

Financial metrics are solid with 15.7% revenue growth and improving profit margins. The P/E ratio of 22.5 is reasonable for the sector.

Recent news about successful product launches and positive analyst coverage provide additional confidence for upward movement.
</think>

<answer>direction: up change: 2.3%</answer>

IMPORTANT:
1. You MUST include the <think> and <answer> tags exactly as shown
2. Your answer MUST include "direction: up/down" and "change: X.X%" - no other variations
3. Include at least 3 specific Key Factors
4. Provide detailed analysis with both technical and fundamental factors
5. Make a clear, decisive prediction with a reasonable percentage change estimate"""

    # Process all examples to calculate percentage changes
    all_examples = []
    
    for item in dataset:
        # Process company information and price data
        company_info = item.get('company_info', {})
        price_data = {}
        financials = {}
        news = {}
        
        if isinstance(company_info, dict):
            price_data = company_info.get('price', {})
            financials_container = company_info.get('financials', {})
            if isinstance(financials_container, dict):
                financials_str = financials_container.get('financials', '{}')
                if isinstance(financials_str, str):
                    try:
                        financials = json.loads(financials_str)
                    except json.JSONDecodeError:
                        financials = {}
            news = company_info.get('news', {})
        
        # Calculate actual price change percentage
        price_change_pct = 0
        if isinstance(price_data, dict):
            current_price = float(price_data.get('open', 0))
            previous_price = float(price_data.get('close_previous', 0))
            if previous_price:
                price_change_pct = ((current_price - previous_price) / previous_price * 100)
        
        # Create detailed prompt
        prompt = create_detailed_financial_prompt(item.get('ticker', 'STOCK'), company_info, price_data, financials, news)
        
        # Add to examples with direction
        all_examples.append({
            "prompt": prompt,
            "actual_percentage": price_change_pct,
            "direction": "up" if price_change_pct >= 0 else "down"
        })
    
    # Split examples by direction
    up_examples = [ex for ex in all_examples if ex["direction"] == "up"]
    down_examples = [ex for ex in all_examples if ex["direction"] == "down"]
    
    print(f"Total examples: {len(all_examples)}")
    print(f"UP examples: {len(up_examples)}")
    print(f"DOWN examples: {len(down_examples)}")
    
    # Balance the dataset by taking equal numbers of UP and DOWN examples
    min_count = min(len(up_examples), len(down_examples))
    
    # Shuffle examples to ensure diversity
    random.shuffle(up_examples)
    random.shuffle(down_examples)
    
    # Take equal numbers from each direction
    balanced_examples = up_examples[:min_count] + down_examples[:min_count]
    
    # Shuffle again to mix UP and DOWN examples
    random.shuffle(balanced_examples)
    
    # Format for training
    formatted_prompts = []
    for example in balanced_examples:
        formatted_prompts.append({
            "prompt": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": example["prompt"]}
            ],
            "actual_percentage": example["actual_percentage"]
        })
    
    print(f"Created balanced dataset with {len(formatted_prompts)} examples (50% UP, 50% DOWN)")
    return formatted_prompts

def create_detailed_financial_prompt(ticker, company_info, price_data, financials, news):
    """Create a detailed financial prompt for analysis."""
    # Extract basic information
    company_name = company_info.get('name', ticker)
    sector = company_info.get('sector', 'Unknown')
    industry = company_info.get('industry', 'Unknown')
    
    # Extract price data
    current_price = price_data.get('open', 'N/A')
    previous_close = price_data.get('close_previous', 'N/A')
    
    # Extract financial metrics (use default values if not available)
    revenue = financials.get('revenue', 'N/A')
    net_income = financials.get('net_income', 'N/A')
    eps = financials.get('eps', 'N/A')
    pe_ratio = financials.get('pe_ratio', 'N/A')
    
    # Generate random but reasonable technical indicators
    rsi = round(random.uniform(30, 70), 2)
    macd = round(random.uniform(-2, 2), 2)
    moving_avg_50 = float(current_price) * random.uniform(0.9, 1.1) if current_price != 'N/A' else 'N/A'
    moving_avg_200 = float(current_price) * random.uniform(0.8, 1.2) if current_price != 'N/A' else 'N/A'
    
    # Format moving averages to two decimal places if they are numbers
    if moving_avg_50 != 'N/A':
        moving_avg_50 = round(moving_avg_50, 2)
    if moving_avg_200 != 'N/A':
        moving_avg_200 = round(moving_avg_200, 2)
    
    # Extract news headlines (up to 5)
    headlines = []
    if isinstance(news, dict) and 'headlines' in news:
        news_items = news['headlines']
        if isinstance(news_items, list):
            headlines = [item.get('headline', '') for item in news_items[:5] if isinstance(item, dict)]
    
    # Create news section
    news_section = "Recent News:\n"
    if headlines:
        for headline in headlines:
            news_section += f"- {headline}\n"
    else:
        news_section += "- No recent news available\n"
    
    # Create the prompt
    prompt = f"""Analysis Request for {ticker} ({company_name})

Company Information:
Sector: {sector}
Industry: {industry}

Price Data:
Current Price: ${current_price}
Previous Close: ${previous_close}

Technical Indicators:
RSI (14-day): {rsi}
MACD: {macd}
50-day Moving Average: ${moving_avg_50}
200-day Moving Average: ${moving_avg_200}

Financial Metrics:
Revenue: ${revenue}
Net Income: ${net_income}
EPS: ${eps}
P/E Ratio: {pe_ratio}

{news_section}

Based on this information, predict whether {ticker} stonk will go UP or DOWN, and estimate the percentage change. Include your analysis reasoning."""
    return prompt

def process_completion(completion, actual_pct=None, debug=True):
    """Process a completion and extract the prediction."""
    # Add debug parameter and verbose logging
    if debug:
        print("\n" + "="*80)
        print("FULL MODEL RESPONSE:")
        print(completion)
        print("="*80)
        
    extracted_text = completion
    
    # Extract direction and percentage
    direction_pattern = re.compile(r'<answer>.*?direction:\s*(up|down).*?change:\s*(\d+\.?\d*)%.*?</answer>', re.DOTALL | re.IGNORECASE)
    match = direction_pattern.search(extracted_text)
    
    direction = None
    percentage = None
    format_score = 0.0
    
    if match:
        direction = match.group(1).lower()
        percentage = float(match.group(2))
        format_score = 1.0  # Good format
    else:
        # Try alternative patterns
        up_pattern = re.compile(r'<answer>\s*up\s*</answer>', re.IGNORECASE)
        down_pattern = re.compile(r'<answer>\s*down\s*</answer>', re.IGNORECASE)
        
        if up_pattern.search(extracted_text):
            direction = "up"
            format_score = 0.5  # Partial format
        elif down_pattern.search(extracted_text):
            direction = "down"
            format_score = 0.5  # Partial format
    
    if debug:
        print(f"\nEXTRACTED PREDICTION:")
        print(f"Direction: {direction}")
        print(f"Percentage: {percentage}%")
        if actual_pct:
            print(f"Actual Change: {actual_pct}%")
        print(f"Format Score: {format_score}")
        print("="*80 + "\n")
    
    return direction, percentage, format_score

def main():
    # Parse arguments
    args = parse_args()

    # Set up training arguments
    training_args = GRPOConfig(
        learning_rate=1e-6,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_prompt_length=512,
        max_completion_length=512,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=5,
        save_total_limit=3,
        output_dir=args.output_dir,
        report_to="none",
        warmup_steps=2,
        logging_steps=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.01,
        remove_unused_columns=False,
    )

    # Load dataset
    if args.dataset_path.startswith("2084Collective/"):
        dataset = load_dataset(args.dataset_path, split="train")
    else:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Format dataset for the model with balanced UP/DOWN examples
    formatted_prompts = format_dataset_for_model(dataset)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_name}")
    
    # Check if we should load from a pretrained checkpoint
    if args.use_pretrained_checkpoint:
        print(f"Loading from pretrained checkpoint: {args.use_pretrained_checkpoint}")
        model_path = args.use_pretrained_checkpoint
    else:
        # Default to stage 1 checkpoint
        model_path = "outputs_namek_stage1"
        print(f"Loading from default stage 1 checkpoint: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Initialize accuracy monitor callback
    accuracy_monitor = AccuracyMonitorCallback()
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        reward_fn=intermediate_up_down_reward_func,
        prompts=formatted_prompts,
        callbacks=[accuracy_monitor]
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Print final accuracy
    if accuracy_monitor.total_predictions > 0:
        final_accuracy = accuracy_monitor.correct_predictions / accuracy_monitor.total_predictions
        print(f"\nFinal UP/DOWN prediction accuracy: {final_accuracy:.4f} ({accuracy_monitor.correct_predictions}/{accuracy_monitor.total_predictions})")

if __name__ == "__main__":
    main() 