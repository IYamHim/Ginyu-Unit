#!/usr/bin/env python3
# Namek Project - Stage 3 GRPO Training Script
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

def advanced_up_down_reward_func(prompts, completions, actual_percentages=None, **kwargs):
    """
    Advanced reward function for stage 3 that focuses more on prediction accuracy
    and quality of analysis rather than just format compliance.
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
        up_pattern = re.compile(r'<answer>\s*up\s*</answer>', re.IGNORECASE)
        down_pattern = re.compile(r'<answer>\s*down\s*</answer>', re.IGNORECASE)
        
        # Check for thinking tag
        thinking_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        
        # Determine decision
        is_up = up_pattern.search(extracted_text) is not None
        is_down = down_pattern.search(extracted_text) is not None
        
        # Track up/down accuracy and apply rewards/penalties
        if is_up and actual_pct is not None:
            total_count += 1
            # UP was correct if price went up
            if actual_pct > 0:
                up_down_accuracy.append("Correct")
                correct_count += 1
                # Slightly lower reward for correct UP prediction to favor DOWN predictions
                reward = 3.8  # Reduced from 4.0 to make the model slightly favor DOWN predictions
                print(f"Correct UP prediction (price went up {actual_pct}%): +3.8")
                is_correct_prediction = True
            else:
                up_down_accuracy.append("Wrong")
                # Stronger penalty for wrong prediction in stage 3
                reward = -2.0  # Increased from -1.0 in stage 1
                print(f"Wrong UP prediction (price went down {actual_pct}%): -2.0")
                is_correct_prediction = False
        elif is_down and actual_pct is not None:
            total_count += 1
            # DOWN was correct if price went down
            if actual_pct < 0:
                up_down_accuracy.append("Correct")
                correct_count += 1
                # Higher reward for correct prediction in stage 3
                reward = 4.0  # Kept at 4.0 to favor DOWN predictions
                print(f"Correct DOWN prediction (price went down {actual_pct}%): +4.0")
                is_correct_prediction = True
            else:
                up_down_accuracy.append("Wrong")
                # Stronger penalty for wrong prediction in stage 3
                reward = -2.0  # Increased from -1.0 in stage 1
                print(f"Wrong DOWN prediction (price went up {actual_pct}%): -2.0")
                is_correct_prediction = False
        else:
            up_down_accuracy.append("Invalid")
            # Even stronger penalty for invalid/missing prediction in stage 3
            reward = -3.0  # Increased from -2.0 in stage 1
            print("Invalid or missing up/down prediction: -3.0")
            is_correct_prediction = False
        
        # Reward for including proper tags - less important in stage 3
        if thinking_pattern.search(extracted_text):
            reward += 0.3  # Reduced from 0.5 in stage 1
            print("Thinking tag found: +0.3")
        else:
            reward -= 0.3  # Reduced from 0.5 in stage 1
            print("Missing thinking tag: -0.3")
        
        # Check for quality of analysis in thinking section
        thinking_match = thinking_pattern.search(extracted_text)
        if thinking_match:
            thinking_text = thinking_match.group(1).lower()
            
            # Reward for mentioning specific financial metrics
            financial_metrics = [
                "revenue", "earnings", "profit margin", "eps", "p/e ratio", 
                "debt", "cash flow", "dividend", "growth rate", "market share"
            ]
            metrics_mentioned = 0
            for metric in financial_metrics:
                if metric in thinking_text:
                    metrics_mentioned += 1
            
            if metrics_mentioned >= 3:
                reward += 1.5  # Increased from 1.0 in stage 1
                print(f"Mentioned {metrics_mentioned} financial metrics: +1.5")
            elif metrics_mentioned >= 1:
                reward += 0.5
                print(f"Mentioned {metrics_mentioned} financial metrics: +0.5")
            
            # Reward for mentioning technical indicators
            technical_indicators = [
                "moving average", "rsi", "macd", "support", "resistance", 
                "trend", "momentum", "volume", "volatility", "breakout"
            ]
            indicators_mentioned = 0
            for indicator in technical_indicators:
                if indicator in thinking_text:
                    indicators_mentioned += 1
            
            if indicators_mentioned >= 2:
                reward += 1.0
                print(f"Mentioned {indicators_mentioned} technical indicators: +1.0")
            elif indicators_mentioned >= 1:
                reward += 0.5
                print(f"Mentioned {indicators_mentioned} technical indicators: +0.5")
            
            # Reward for balanced analysis (considering both positive and negative factors)
            positive_terms = ["positive", "increase", "growth", "upward", "bullish", "strong", "opportunity"]
            negative_terms = ["negative", "decrease", "decline", "downward", "bearish", "weak", "risk"]
            
            has_positive = any(term in thinking_text for term in positive_terms)
            has_negative = any(term in thinking_text for term in negative_terms)
            
            if has_positive and has_negative:
                reward += 1.0
                print("Balanced analysis (both positive and negative factors): +1.0")
        
        # Ensure correct predictions always have a positive reward
        if is_correct_prediction and reward <= 0:
            reward = 1.0
            print(f"Applied minimum reward floor: reward set to +1.0 (correct prediction)")
            
        print(f"Final reward: {reward}")
        rewards.append(reward)
    
    # Make sure we have a reward for each completion
    if len(rewards) < len(completions):
        print(f"WARNING: Only {len(rewards)} rewards generated for {len(completions)} completions, adding default negative rewards")
        for _ in range(len(completions) - len(rewards)):
            rewards.append(-3.0)
    
    # Convert rewards to tensor properly
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    print("UP/DOWN accuracy:", up_down_accuracy)
    print("UP/DOWN rewards tensor:", rewards_tensor)
    
    # Return accuracy metrics for the callback
    if "metrics" in kwargs:
        kwargs["metrics"]["up_down_accuracy"] = [correct_count, total_count]
    
    return rewards_tensor.clone().detach()

def strict_format_reward_func(prompts, completions, **kwargs):
    """Enhanced strict format reward function that enforces template compliance"""
    rewards = []
    for completion in completions:
        try:
            # Extract the text content from the completion dictionary
            if isinstance(completion, dict) and 'content' in completion:
                completion_text = completion['content']
            elif isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                completion_text = completion[0].get('content', '')
            else:
                completion_text = str(completion)

            # Convert to lowercase for case-insensitive matching
            completion_text = completion_text.lower()
            
            reward = 0.0  # Start with neutral reward
            
            # Check for required tags
            has_think_tag = bool(re.search(r'<think>.*?</think>', completion_text, re.DOTALL))
            has_answer_tag = bool(re.search(r'<answer>(up|down)</answer>', completion_text))
            
            # Strong reward for having both required tags in correct format
            if has_think_tag and has_answer_tag:
                reward += 2.0
                print("Format reward: Both required tags present: +2.0")
            else:
                if not has_think_tag:
                    reward -= 2.0
                    print("Format penalty: Missing think tag: -2.0")
                if not has_answer_tag:
                    reward -= 2.0
                    print("Format penalty: Missing or invalid answer tag: -2.0")
            
            # Check for proper thinking section structure
            if has_think_tag:
                think_content = re.search(r'<think>(.*?)</think>', completion_text, re.DOTALL).group(1)
                
                # Check for "Key Factors" section
                has_key_factors = bool(re.search(r'key factors:', think_content, re.IGNORECASE))
                if has_key_factors:
                    reward += 1.0
                    print("Format reward: Has Key Factors section: +1.0")
                    
                    # Check for numbered factors
                    numbered_factors = len(re.findall(r'\d+\.', think_content))
                    if numbered_factors >= 3:
                        reward += 1.0
                        print(f"Format reward: Has {numbered_factors} numbered factors: +1.0")
                
                # Check for "Analysis" section
                has_analysis = bool(re.search(r'analysis:', think_content, re.IGNORECASE))
                if has_analysis:
                    reward += 1.0
                    print("Format reward: Has Analysis section: +1.0")
                
                # Penalize extremely short thinking sections
                think_length = len(think_content.strip())
                if think_length < 100:
                    reward -= 1.5
                    print("Format penalty: Thinking section too short: -1.5")
                elif think_length > 300:
                    reward += 0.5
                    print("Format reward: Detailed thinking section: +0.5")
            
            # Penalize multiple think or answer tags
            think_count = len(re.findall(r'<think>', completion_text))
            answer_count = len(re.findall(r'<answer>', completion_text))
            
            if think_count > 1:
                reward -= 2.0
                print(f"Format penalty: Multiple think tags ({think_count}): -2.0")
            if answer_count > 1:
                reward -= 2.0
                print(f"Format penalty: Multiple answer tags ({answer_count}): -2.0")
            
            # Penalize any non-standard tags or formats
            if re.search(r'<(?!think|answer|/think|/answer)[^>]+>', completion_text):
                reward -= 1.0
                print("Format penalty: Invalid tags present: -1.0")
            
            print(f"Final format reward: {reward}")
            rewards.append(reward)
            
        except Exception as e:
            print(f"Error in format checking: {e}")
            rewards.append(-2.0)
    
    # Convert rewards to tensor
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    print("Format rewards tensor:", rewards_tensor)
    return rewards_tensor.clone().detach()

def detailed_confidence_reward_func(prompts, completions, actual_percentages=None, **kwargs):
    """
    Rewards model for appropriate confidence levels based on evidence quality and analysis depth.
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        reward = 0.0
        extracted_text = str(completion)
        
        # Extract thinking and answer sections
        thinking_match = re.search(r'<think>(.*?)</think>', extracted_text, re.DOTALL)
        answer_match = re.search(r'<answer>(up|down)</answer>', extracted_text, re.IGNORECASE)
        
        if thinking_match and answer_match:
            thinking_text = thinking_match.group(1).lower()
            prediction = answer_match.group(1).lower()
            
            # Check for required sections
            has_key_factors = "key factors:" in thinking_text
            has_analysis = "analysis:" in thinking_text
            
            if has_key_factors and has_analysis:
                reward += 1.0
                print("Format reward: Has both Key Factors and Analysis sections: +1.0")
                
                # Count numbered key factors
                key_factors = len(re.findall(r'\d+\.', thinking_text))
                if key_factors >= 3:
                    reward += 0.5
                    print(f"Format reward: Has {key_factors} numbered factors: +0.5")
                
                # Check for specific metrics and indicators
                metrics = [
                    "rsi", "macd", "moving average", "revenue", "eps", "profit margin",
                    "p/e ratio", "volume", "price", "growth", "trend"
                ]
                
                metrics_found = sum(1 for m in metrics if m in thinking_text)
                if metrics_found >= 4:
                    reward += 1.0
                    print(f"Analysis reward: Referenced {metrics_found} specific metrics: +1.0")
                
                # Check for balanced analysis
                positive_terms = ["increase", "growth", "positive", "strong", "bullish"]
                negative_terms = ["decrease", "decline", "negative", "weak", "bearish"]
                
                has_positive = any(term in thinking_text for term in positive_terms)
                has_negative = any(term in thinking_text for term in negative_terms)
                
                if has_positive and has_negative:
                    reward += 0.5
                    print("Analysis reward: Balanced consideration of positive and negative factors: +0.5")
                
                # Check analysis depth
                word_count = len(thinking_text.split())
                if word_count > 150:
                    reward += 0.5
                    print("Analysis reward: Detailed analysis (>150 words): +0.5")
                
                # Check for quantitative references
                quant_refs = len(re.findall(r'\d+\.?\d*%?', thinking_text))
                if quant_refs >= 3:
                    reward += 0.5
                    print(f"Analysis reward: {quant_refs} quantitative references: +0.5")
                
                # Penalize hedging language without clear decision
                hedging_terms = ["might", "maybe", "possibly", "unclear", "uncertain"]
                hedging_count = sum(1 for term in hedging_terms if term in thinking_text)
                if hedging_count > 2:
                    reward -= 0.5
                    print(f"Confidence penalty: Excessive hedging language ({hedging_count} instances): -0.5")
            
            else:
                reward -= 1.0
                print("Format penalty: Missing required sections: -1.0")
        
        print(f"Final confidence reward: {reward}")
        rewards.append(reward)
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    return rewards_tensor.clone().detach()

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 GRPO training for UP/DOWN predictions")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name or path")
    parser.add_argument("--checkpoint_path", type=str, default="outputs_up_down_simple_batch4_20steps", help="Path to stage 1 checkpoint")
    parser.add_argument("--dataset_path", type=str, default="2084Collective/deepstock-sp500-companies-with-info-and-user-prompt", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="outputs_up_down_stage3", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of training steps")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to use")
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

    formatted_prompts = []
    
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
        
        # Add to formatted prompts
        formatted_prompts.append({
            "prompt": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "actual_percentage": price_change_pct
        })
    
    print(f"Formatted {len(formatted_prompts)} prompts for training")
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

Based on this information, predict whether {ticker} stonk will go UP or DOWN, and estimate the percentage change. Include your analysis reasoning.
"""
    return prompt

def create_structured_prediction(model, prompt):
    """Generate structured prediction using Outlines"""
    try:
        # First generate the analysis section
        analysis_prompt = prompt + "\n\nProvide your analysis in <think></think> tags:"
        analysis_generator = outlines.generate.regex(
            model,
            r"<think>.*?</think>",
            sampler=outlines.samplers.greedy()
        )
        analysis = analysis_generator(analysis_prompt, max_tokens=512)
        
        # Then make the UP/DOWN decision
        decision_prompt = prompt + "\n" + analysis + "\n\nBased on this analysis, predict UP or DOWN:"
        decision_generator = outlines.generate.choice(model, StockDirection)
        prediction = decision_generator(decision_prompt)
        
        # Combine into final format
        return f"{analysis}\n\n<answer>{prediction}</answer>"
    except Exception as e:
        print(f"Error in structured prediction: {e}")
        return None

def train_grpo(model, tokenizer, formatted_prompts, training_args, callbacks=None):
    # Convert formatted prompts to the expected dataset format
    train_dataset = []
    for prompt in formatted_prompts:
        # Extract company info from the prompt content
        content = prompt["prompt"][1]["content"]  # Get the user message content
        
        # Extract price data using regex
        current_price = float(re.search(r"Current Price: \$([0-9.]+)", content).group(1))
        previous_close = float(re.search(r"Previous Close: \$([0-9.]+)", content).group(1))
        
        # Calculate actual percentage change
        actual_percentage = prompt["actual_percentage"]
        actual_direction = "UP" if actual_percentage > 0 else "DOWN"
        
        # Extract RSI and MACD
        rsi = float(re.search(r"RSI \(14-day\): ([0-9.]+)", content).group(1))
        macd = float(re.search(r"MACD: ([0-9.-]+)", content).group(1))
        
        # Extract headlines
        headlines = re.findall(r"- (.+?)(?:\n|$)", content)
        
        # Create dataset entry
        entry = {
            "ticker": "STOCK",  # Default ticker since we don't have it in the prompt
            "company_name": "Company",  # Default company name
            "current_price": current_price,
            "previous_price": previous_close,
            "sector": "General",  # Default sector
            "industry": "General",  # Default industry
            "technical_indicators": {
                "rsi": rsi,
                "macd": macd,
                "vix": 20.0,  # Default VIX
                "volume": 1000000,  # Default volume
                "market_trend": "neutral"  # Default market trend
            },
            "headlines": headlines,
            "input_ids": tokenizer(
                [msg["content"] for msg in prompt["prompt"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).input_ids[0],
            "attention_mask": tokenizer(
                [msg["content"] for msg in prompt["prompt"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).attention_mask[0],
            "metadata": {
                "actual_direction": actual_direction,
                "actual_percentage": abs(actual_percentage)  # Use absolute value for percentage change
            }
        }
        train_dataset.append(entry)

    # Set up GRPO trainer with the correct parameters
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=512,
        kl_coef=0.1,
        beta=0.1,
    )

    # The custom GRPOTrainer from Over9Thousand.py doesn't support add_callback
    # If callbacks are needed, they should be implemented directly in the custom GRPOTrainer

    return trainer

def process_completion(completion, actual_pct=None, debug=True):
    """Process a completion and extract the prediction."""
    # Add debug parameter and verbose logging
    if debug:
        print("\n" + "="*80)
        print("FULL MODEL RESPONSE:")
        print(completion)
        print("="*80)
        
    extracted_text = completion
    # ... existing code ...
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--use_pretrained_checkpoint", type=str, default=None, help="Path to pretrained checkpoint to continue training from")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

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
    
    # Format dataset for the model
    formatted_prompts = format_dataset_for_model(dataset)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_name}")
    
    # Check if we should load from a pretrained checkpoint
    if args.use_pretrained_checkpoint:
        print(f"Loading from pretrained checkpoint: {args.use_pretrained_checkpoint}")
        model_path = args.use_pretrained_checkpoint
    else:
        # Default to "Highly Trained GRPO" checkpoint as requested
        model_path = "outputs_up_down_stage3/checkpoint-30"
        print(f"Loading from default checkpoint: {model_path}")
        
    # Load model and tokenizer from the specified path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Train GRPO
    print("Starting GRPO training...")
    trainer = train_grpo(model, tokenizer, formatted_prompts, training_args)
    trainer.train()
    
    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 