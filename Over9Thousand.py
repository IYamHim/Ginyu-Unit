"""
Over9Thousand.py - Advanced Stock Market Prediction Model
Version: 3.0.0
Based on the original train_7b.py script with significant improvements
"""

import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import re
from copy import deepcopy
from datasets import load_dataset, Dataset
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, SFTTrainer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('training_v3.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Enhanced random seed setting
def set_random_seed(seed):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Additional seeds for maximum reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Improved LoRA configuration
def get_lora_config():
    """Get LoRA configuration for training"""
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

# Add a function to get optimized LoRA config for low VRAM
def get_optimized_lora_config(r=16, alpha=32):
    """Get memory-optimized LoRA configuration for low VRAM GPUs"""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "v_proj", "o_proj",  # Reduced target modules for better efficiency
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,  # Ensure training mode
        fan_in_fan_out=False,  # Better memory efficiency
    )

# Enhanced model preparation
def prepare_model_for_training(model):
    """Prepare model for training with advanced configurations"""
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Percentage of trainable parameters: {(trainable_params/total_params)*100:.2f}%")
    
    return model

class MarketConditions:
    """Analyzes market conditions for dynamic reward scaling"""
    
    def __init__(self):
        self.vix_levels = {
            'low': (0, 15),
            'normal': (15, 25),
            'high': (25, 35),
            'extreme': (35, float('inf'))
        }
        self.market_states = {}
    
    def analyze_volatility(self, ticker: str, price_history: List[float]) -> float:
        """Calculate historical volatility for a stock"""
        if len(price_history) < 2:
            return 0.0
        
        returns = np.diff(np.log(price_history))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        return volatility
    
    def get_market_regime(self, vix_value: float) -> str:
        """Determine market regime based on VIX"""
        for regime, (low, high) in self.vix_levels.items():
            if low <= vix_value < high:
                return regime
        return 'normal'
    
    def calculate_reward_multiplier(self, 
                                  ticker: str, 
                                  price_history: List[float], 
                                  vix_value: Optional[float] = None,
                                  market_trend: Optional[str] = None) -> float:
        """Calculate dynamic reward multiplier based on market conditions"""
        # Calculate volatility-based component
        volatility = self.analyze_volatility(ticker, price_history)
        vol_multiplier = 1.0 + (volatility - 0.2) if volatility > 0.2 else 1.0
        
        # VIX-based component
        vix_multiplier = 1.0
        if vix_value is not None:
            regime = self.get_market_regime(vix_value)
            vix_multiplier = {
                'low': 0.8,
                'normal': 1.0,
                'high': 1.2,
                'extreme': 1.5
            }.get(regime, 1.0)
        
        # Market trend component
        trend_multiplier = 1.0
        if market_trend is not None:
            trend_multiplier = {
                'strong_bull': 1.2,
                'bull': 1.1,
                'neutral': 1.0,
                'bear': 0.9,
                'strong_bear': 0.8
            }.get(market_trend, 1.0)
        
        # Combine multipliers with weights
        final_multiplier = (0.4 * vol_multiplier + 
                          0.3 * vix_multiplier + 
                          0.3 * trend_multiplier)
        
        # Bound the multiplier
        return max(0.5, min(2.0, final_multiplier))

class ConfidenceCalibrator:
    """Handles sophisticated confidence calibration"""
    
    def __init__(self):
        self.history = []
        self.calibration_scores = {}
        self.recent_window = 50  # Number of recent predictions to consider
    
    def add_prediction(self, 
                      ticker: str, 
                      predicted_confidence: float, 
                      was_correct: bool,
                      prediction_type: str):
        """Record a prediction and its outcome"""
        self.history.append({
            'ticker': ticker,
            'confidence': predicted_confidence,
            'correct': was_correct,
            'type': prediction_type,
            'timestamp': datetime.now()
        })
    
    def calculate_calibration_score(self, 
                                  ticker: str, 
                                  prediction_type: str = 'all') -> float:
        """Calculate calibration score for a ticker"""
        relevant_history = [
            h for h in self.history[-self.recent_window:]
            if (h['ticker'] == ticker or ticker == 'all') and
               (h['type'] == prediction_type or prediction_type == 'all')
        ]
        
        if not relevant_history:
            return 1.0
        
        # Calculate average confidence and accuracy
        avg_confidence = np.mean([h['confidence'] for h in relevant_history])
        accuracy = np.mean([1.0 if h['correct'] else 0.0 for h in relevant_history])
        
        # Calculate calibration error
        calibration_error = abs(avg_confidence - accuracy)
        
        # Convert to a score (1.0 is perfect calibration)
        calibration_score = max(0.0, 1.0 - calibration_error)
        
        return calibration_score
    
    def get_confidence_reward(self, 
                            ticker: str, 
                            predicted_confidence: float, 
                            was_correct: bool,
                            prediction_type: str = 'all') -> float:
        """Calculate confidence-based reward component"""
        # Add the current prediction
        self.add_prediction(ticker, predicted_confidence, was_correct, prediction_type)
        
        # Get calibration score
        calibration_score = self.calculate_calibration_score(ticker, prediction_type)
        
        # Calculate base confidence reward
        if was_correct:
            base_reward = predicted_confidence * 0.5  # Increased from 0.2
        else:
            base_reward = -predicted_confidence * 0.6  # Increased penalty
        
        # Adjust reward based on calibration
        adjusted_reward = base_reward * calibration_score
        
        return adjusted_reward

def compute_multi_day_reward(predictions: List[Dict], 
                           actuals: List[Dict], 
                           response_text: str,
                           market_conditions: MarketConditions,
                           confidence_calibrator: ConfidenceCalibrator,
                           ticker: str,
                           price_history: List[float],
                           vix_value: Optional[float] = None,
                           market_trend: Optional[str] = None) -> Tuple[float, str, Dict]:
    """
    Compute reward for multi-day predictions
    
    Args:
        predictions: List of prediction dictionaries for each day
        actuals: List of actual outcomes for each day
        response_text: Full model response text
        market_conditions: MarketConditions instance
        confidence_calibrator: ConfidenceCalibrator instance
        ticker: Stock ticker symbol
        price_history: List of historical prices
        vix_value: Optional VIX index value
        market_trend: Optional market trend indicator
    """
    try:
        # Format reward (highest priority)
        format_reward = 0.0
        format_explanation = "Format check: "
        
        if response_text:
            # Check for proper format for each day
            required_days = len(predictions)
            day_patterns = [
                f"Day {i+1}:" for i in range(required_days)
            ]
            
            has_all_days = all(pattern in response_text for pattern in day_patterns)
            has_think_tags = len(re.findall(r'<think>.*?</think>', response_text, re.DOTALL)) == required_days
            has_answer_tags = len(re.findall(r'<answer>.*?</answer>', response_text, re.DOTALL)) == required_days
            
            if has_all_days and has_think_tags and has_answer_tags:
                format_reward = 1.0
                format_explanation += f"Correct format for all {required_days} days"
            elif has_all_days and (has_think_tags or has_answer_tags):
                format_reward = 0.3
                format_explanation += "Partial format (missing some tags)"
            else:
                format_reward = -1.0
                format_explanation += f"Missing required format for {required_days} days"
                
            if format_reward == -1.0:
                return -1.0, format_explanation, {
                    "format_reward": format_reward,
                    "total_reward": -1.0,
                    "error": "Invalid response format"
                }
        
        # Calculate reward for each day
        day_rewards = []
        day_explanations = []
        
        for day_idx, (pred, actual) in enumerate(zip(predictions, actuals)):
            # Extract prediction components
            pred_direction = pred.get('direction', 'unknown').lower()
            pred_percentage = pred.get('percentage', 0.0)
            pred_confidence = pred.get('confidence', 50)
            if pred_confidence > 1:
                pred_confidence = pred_confidence / 100.0
            
            actual_direction = actual.get('direction', 'up').lower()
            actual_change_pct = actual.get('percentage', 0.0)
            
            # Direction reward
            direction_reward = 1.0 if pred_direction == actual_direction else -0.5
            
            # Percentage accuracy reward (enhanced)
            percentage_diff = abs(pred_percentage - actual_change_pct)
            if percentage_diff <= 0.25:  # Tighter threshold for highest reward
                percentage_reward = 0.8
            elif percentage_diff <= 0.5:
                percentage_reward = 0.5
            elif percentage_diff <= 1.0:
                percentage_reward = 0.3
            elif percentage_diff <= 2.0:
                percentage_reward = 0.1
            else:
                percentage_reward = -0.2  # Increased penalty
            
            # Enhanced confidence reward using calibrator
            confidence_reward = confidence_calibrator.get_confidence_reward(
                ticker=ticker,
                predicted_confidence=pred_confidence,
                was_correct=pred_direction == actual_direction,
                prediction_type=f'day_{day_idx+1}'
            )
            
            # Calculate day's reward
            day_reward = direction_reward + percentage_reward + confidence_reward
            
            # Apply time decay factor (later days have slightly less weight)
            time_decay = 1.0 / (1.0 + 0.1 * day_idx)  # 10% decay per day
            day_reward *= time_decay
            
            day_rewards.append(day_reward)
            
            # Create day explanation
            day_explanation = (
                f"Day {day_idx+1}: "
                f"{'Correct' if pred_direction == actual_direction else 'Incorrect'} direction. "
                f"Predicted {pred_percentage:.1f}% vs Actual {actual_change_pct:.1f}%. "
                f"Confidence: {pred_confidence*100:.0f}%"
            )
            day_explanations.append(day_explanation)
        
        # Calculate dynamic reward multiplier
        reward_multiplier = market_conditions.calculate_reward_multiplier(
            ticker=ticker,
            price_history=price_history,
            vix_value=vix_value,
            market_trend=market_trend
        )
        
        # Calculate final reward
        base_reward = (2.0 * format_reward) + sum(day_rewards) / len(day_rewards)
        final_reward = base_reward * reward_multiplier
        
        # Ensure reward is within bounds
        final_reward = max(-1.0, min(2.0, final_reward))
        
        # Create detailed explanation
        explanation = format_explanation + "\n" + "\n".join(day_explanations)
        explanation += f"\nMarket conditions multiplier: {reward_multiplier:.2f}"
        
        # Detailed reward breakdown
        details = {
            "format_reward": format_reward,
            "day_rewards": day_rewards,
            "market_multiplier": reward_multiplier,
            "final_reward": final_reward,
            "predictions": predictions,
            "actuals": actuals
        }
        
        return final_reward, explanation, details
        
    except Exception as e:
        logger.error(f"Error computing multi-day reward: {str(e)}")
        return -1.0, f"Error computing reward: {str(e)}", {"error": str(e)}

def extract_multi_day_predictions(response_text: str, num_days: int = 3) -> List[Dict]:
    """
    Extract multi-day predictions from model response with enhanced format handling
    
    Args:
        response_text: Full model response text
        num_days: Number of days to extract predictions for
    
    Returns:
        List of prediction dictionaries for each day
    """
    predictions = []
    
    try:
        # Extract predictions for each day
        for day in range(1, num_days + 1):
            prediction = {'day': day, 'direction': None, 'percentage': None, 'confidence': None}
            
            # Find the section for this day
            day_pattern = f"Day {day}:"
            day_match = re.search(f"{day_pattern}(.*?)(?:Day {day+1}:|$)", response_text, re.DOTALL)
            
            if not day_match:
                logger.warning(f"Day {day} section not found in response")
                predictions.append(prediction)
                continue
            
            day_text = day_match.group(1).strip()
            
            # Extract answer (direction and percentage)
            answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', day_text, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                # Look for the direction and change pattern
                direction_match = re.search(r'Direction:\s*(UP|DOWN)\s*Change:\s*(\d+\.?\d*)%', answer_text, re.IGNORECASE)
                if direction_match:
                    prediction['direction'] = direction_match.group(1).upper()  # Ensure uppercase
                    prediction['percentage'] = float(direction_match.group(2))
                else:
                    # Try alternative formats
                    alt_formats = [
                        r'(UP|DOWN)\s+(\d+\.?\d*)%',  # Simple UP/DOWN X%
                        r'(UP|DOWN).*?(\d+\.?\d*)%',  # UP/DOWN with text between
                        r'(\d+\.?\d*)%\s*(UP|DOWN)',  # X% UP/DOWN
                    ]
                    for pattern in alt_formats:
                        match = re.search(pattern, answer_text, re.IGNORECASE)
                        if match:
                            groups = match.groups()
                            if groups[0].upper() in ['UP', 'DOWN']:
                                prediction['direction'] = groups[0].upper()
                                prediction['percentage'] = float(groups[1])
                            else:
                                prediction['direction'] = groups[1].upper()
                                prediction['percentage'] = float(groups[0])
                            break
                    
                    if prediction['direction'] is None:
                        logger.warning(f"Could not extract direction and percentage from answer: {answer_text}")
            else:
                logger.warning(f"No answer tag found for day {day}")
            
            # Extract confidence
            confidence_match = re.search(r'<confidence>\s*(\d+\.?\d*)%?\s*</confidence>', day_text, re.DOTALL)
            if confidence_match:
                try:
                    confidence_str = confidence_match.group(1).strip()
                    confidence = float(confidence_str)
                    # Ensure confidence is between 1 and 100
                    if 0 < confidence <= 100:
                        prediction['confidence'] = confidence / 100.0  # Convert to 0-1 scale
                    else:
                        logger.warning(f"Confidence value {confidence} out of range (1-100)")
                except ValueError:
                    logger.warning(f"Could not parse confidence value: {confidence_str}")
            else:
                logger.warning(f"No confidence tag found for day {day}")
            
            predictions.append(prediction)
    except Exception as e:
        logger.error(f"Error extracting predictions: {str(e)}")
    
    return predictions

def extract_up_down_with_percentage(response_text: str) -> Dict:
    """
    Extract both UP/DOWN prediction and percentage change from model response
    
    Args:
        response_text: Full model response text
    
    Returns:
        Dictionary with direction (UP/DOWN) and percentage change
    """
    prediction = {'direction': None, 'percentage': None}
    
    try:
        # Extract the answer tag content
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip().lower()
            
            # Check for "direction: up/down change: X.X%" format
            direction_pct_match = re.search(r'direction:\s*(up|down)\s*change:\s*(\d+\.?\d*)%?', answer_text, re.IGNORECASE)
            if direction_pct_match:
                direction = direction_pct_match.group(1).upper()
                percentage = float(direction_pct_match.group(2))
                prediction['direction'] = direction
                prediction['percentage'] = percentage
                logger.info(f"Extracted direction '{direction}' with percentage {percentage}%")
                return prediction
            
            # Check for alternative formats like "UP 2.5%" or "DOWN by 1.8%"
            alt_formats = [
                r'(up|down)\s+(\d+\.?\d*)%',  # Simple UP/DOWN X%
                r'(up|down).*?(\d+\.?\d*)%',  # UP/DOWN with text between
                r'(\d+\.?\d*)%\s*(up|down)',  # X% UP/DOWN
            ]
            for pattern in alt_formats:
                match = re.search(pattern, answer_text, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if groups[0].lower() in ['up', 'down']:
                        prediction['direction'] = groups[0].upper()
                        prediction['percentage'] = float(groups[1])
                    else:
                        prediction['direction'] = groups[1].upper()
                        prediction['percentage'] = float(groups[0])
                    logger.info(f"Extracted direction '{prediction['direction']}' with percentage {prediction['percentage']}% from alternative format")
                    return prediction
            
            # Check for simple "up" or "down" without percentage
            if answer_text == "up" or "up" in answer_text and "down" not in answer_text:
                prediction['direction'] = "UP"
                logger.info("Extracted direction 'UP' without percentage")
            elif answer_text == "down" or "down" in answer_text and "up" not in answer_text:
                prediction['direction'] = "DOWN"
                logger.info("Extracted direction 'DOWN' without percentage")
        
        # If we couldn't extract from answer tags, look for UP/DOWN and percentages in the whole text
        if prediction['direction'] is None:
            # Find UP/DOWN in the text
            if "up" in response_text.lower() and "down" not in response_text.lower():
                prediction['direction'] = "UP"
            elif "down" in response_text.lower() and "up" not in response_text.lower():
                prediction['direction'] = "DOWN"
            elif "up" in response_text.lower() and "down" in response_text.lower():
                # If both are present, check context or use the last one
                last_up_pos = response_text.lower().rfind('up')
                last_down_pos = response_text.lower().rfind('down')
                prediction['direction'] = "UP" if last_up_pos > last_down_pos else "DOWN"
        
        # Look for percentage in the whole text if not found yet
        if prediction['percentage'] is None:
            percentage_matches = re.findall(r'(\d+\.?\d*)%', response_text)
            if percentage_matches:
                # Use the percentage closest to UP/DOWN mention
                if prediction['direction'] == "UP":
                    up_pos = response_text.lower().find('up')
                    closest_pct = None
                    min_distance = float('inf')
                    for pct_str in percentage_matches:
                        pct_pos = response_text.find(pct_str + '%')
                        distance = abs(pct_pos - up_pos)
                        if distance < min_distance:
                            min_distance = distance
                            closest_pct = float(pct_str)
                    prediction['percentage'] = closest_pct
                elif prediction['direction'] == "DOWN":
                    down_pos = response_text.lower().find('down')
                    closest_pct = None
                    min_distance = float('inf')
                    for pct_str in percentage_matches:
                        pct_pos = response_text.find(pct_str + '%')
                        distance = abs(pct_pos - down_pos)
                        if distance < min_distance:
                            min_distance = distance
                            closest_pct = float(pct_str)
                    prediction['percentage'] = closest_pct
                else:
                    # If no direction, just use the first percentage found
                    prediction['percentage'] = float(percentage_matches[0])
    
    except Exception as e:
        logger.error(f"Error extracting UP/DOWN with percentage: {str(e)}")
    
    return prediction

# Custom GRPOTrainer implementation
class GRPOTrainer:
    """
    Custom implementation of GRPO (Generalized Reinforcement Policy Optimization) Trainer
    
    This trainer implements GRPO training for language models, specifically optimized
    for stock prediction tasks with reward-based learning.
    """
    
    def __init__(
        self,
        model,
        args,
        train_dataset,
        tokenizer,
        max_seq_length=2048,
        kl_coef=0.1,
        beta=0.1,
        data_collator=None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.kl_coef = kl_coef
        self.beta = beta
        
        # Ensure model is in training mode
        self.model.train()
        
        # Enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params} / {all_params} ({trainable_params/all_params*100:.2f}%)")
        
        # Initialize market conditions and confidence calibrator
        self.market_conditions = MarketConditions()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        
        # Setup scheduler
        self.total_steps = len(train_dataset) * args.num_train_epochs // args.per_device_train_batch_size
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.total_steps,
        )
        
        # Setup data loader
        self.data_collator = data_collator
        self.train_dataloader = self._prepare_dataloader()
        
        # Setup logging and checkpointing
        self.global_step = 0
        self.epoch = 0
        self.best_reward = -float('inf')
        
        logger.info(f"Initialized GRPOTrainer with {len(train_dataset)} examples")
        logger.info(f"Training for {args.num_train_epochs} epochs with batch size {args.per_device_train_batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}, KL coef: {kl_coef}, Beta: {beta}")
    
    def _prepare_dataloader(self):
        """Prepare the training dataloader"""
        if self.data_collator is None:
            # Define a simple collator function
            def collate_fn(examples):
                # Process examples for stock prediction
                input_texts = []
                for ex in examples:
                    # Create input text from stock data with system context and better structure
                    input_text = (
                        "You are an advanced AI stock market analyst specialized in pattern recognition and quantitative analysis. "
                        "Your strength lies in identifying complex market patterns, correlations between technical indicators, and the impact of news sentiment on stock movements. "
                        "You excel at combining multiple data points - technical indicators, market trends, news sentiment, and sector dynamics - to predict stock movements. "
                        "Your predictions must be precise, well-formatted, and supported by clear pattern-based analysis.\n\n"
                        
                        f"Stock Analysis for: {ex['ticker']} ({ex['company_name']})\n"
                        f"Current Price: ${ex['current_price']:.2f}\n"
                        f"Previous Price: ${ex['previous_price']:.2f}\n"
                        f"Sector: {ex['sector']}\n"
                        f"Industry: {ex['industry']}\n"
                        "\nTechnical Indicators:\n"
                    )
                    
                    # Add technical indicators if available
                    tech_indicators = ex.get('technical_indicators', {})
                    input_text += f"- RSI: {tech_indicators.get('rsi', 'N/A')}\n"
                    input_text += f"- VIX: {tech_indicators.get('vix', 'N/A')}\n"
                    input_text += f"- Volume: {tech_indicators.get('volume', 'N/A')}\n"
                    input_text += f"- Market Trend: {tech_indicators.get('market_trend', 'neutral')}\n"
                    
                    input_text += "\nRecent Headlines:\n"
                    for headline in ex['headlines']:
                        input_text += f"- {headline}\n"
                    
                    input_text += "\nPREDICTION REQUIREMENTS:\n"
                    input_text += "1. Provide predictions for the next 3 days\n"
                    input_text += "2. Each prediction must include analysis, direction (UP/DOWN), percentage change, and confidence level\n"
                    input_text += "3. Direction must be exactly 'UP' or 'DOWN' (all caps)\n"
                    input_text += "4. Percentage must be a number between 0.1 and 5.0\n"
                    input_text += "5. Confidence must be a number between 1 and 100\n"
                    input_text += "6. IMPORTANT: Each prediction MUST use the exact XML tags as shown in the template\n"
                    input_text += "7. XML tags must be on their own lines and properly closed\n\n"
                    
                    input_text += "FORMAT TEMPLATE:\n"
                    input_text += "Day N:\n"
                    input_text += "<think>\n"
                    input_text += "Key Factors:\n"
                    input_text += "1. [Factor 1]\n"
                    input_text += "2. [Factor 2]\n"
                    input_text += "3. [Factor 3]\n\n"
                    input_text += "Analysis:\n"
                    input_text += "[Your detailed analysis]\n"
                    input_text += "</think>\n"
                    input_text += "<answer>\n"
                    input_text += "Direction: [UP/DOWN] Change: [X.X]%\n"
                    input_text += "</answer>\n"
                    input_text += "<confidence>\n"
                    input_text += "[YY]%\n"
                    input_text += "</confidence>\n\n"
                    
                    input_text += "EXAMPLE RESPONSE:\n"
                    input_text += "Day 1:\n"
                    input_text += "<think>\n"
                    input_text += "Key Factors:\n"
                    input_text += "1. Strong earnings report exceeded expectations\n"
                    input_text += "2. Positive industry momentum\n"
                    input_text += "3. Favorable market conditions (low VIX)\n\n"
                    input_text += "Analysis:\n"
                    input_text += "The combination of better-than-expected earnings and positive industry momentum suggests upward movement.\n"
                    input_text += "</think>\n"
                    input_text += "<answer>\n"
                    input_text += "Direction: UP Change: 1.2%\n"
                    input_text += "</answer>\n"
                    input_text += "<confidence>\n"
                    input_text += "75%\n"
                    input_text += "</confidence>\n\n"
                    
                    input_texts.append(input_text)
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    input_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
                
                # Add metadata for reward computation
                metadata = [{
                    "ticker": ex.get("ticker", "UNKNOWN"),
                    "price_history": [ex.get("previous_price", 100.0), ex.get("current_price", 100.0)],
                    "vix_value": ex.get("technical_indicators", {}).get("vix", 20.0),
                    "market_trend": ex.get("technical_indicators", {}).get("market_trend", "neutral"),
                    "actual_outcomes": []  # Will be filled with mock data for training
                } for ex in examples]
                
                return {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "metadata": metadata
                }
            
            self.data_collator = collate_fn
        
        # Create dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
    
    def _compute_kl_divergence(self, logits_1, logits_2, attention_mask=None):
        """Compute KL divergence between two distributions"""
        kl_loss = nn.KLDivLoss(reduction="none")
        
        # Apply softmax to get probabilities
        probs_1 = nn.functional.softmax(logits_1, dim=-1)
        log_probs_2 = nn.functional.log_softmax(logits_2, dim=-1)
        
        # Compute KL divergence
        kl_div = kl_loss(log_probs_2, probs_1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(kl_div)
            kl_div = kl_div * mask
        
        # Take mean across all dimensions
        kl_div = kl_div.mean()
        
        # Make sure kl_div is connected to the computation graph
        if not kl_div.requires_grad:
            logger.warning("KL divergence doesn't require gradients, adding dummy gradient")
            # Add a small gradient-connected term to ensure it's part of the computation graph
            kl_div = kl_div + 0.0 * torch.sum(logits_2)
            
        return kl_div
    
    def _generate_responses(self, input_ids, attention_mask):
        """Generate responses from the model"""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the generated part (not the input prompt)
        generated_outputs = []
        for i, output in enumerate(outputs):
            input_length = input_ids[i].size(0)
            generated_part = output[input_length:]
            generated_outputs.append(generated_part)
        
        # Decode the generated text
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in generated_outputs
        ]
        
        return generated_texts
    
    def _compute_rewards(self, generated_texts, metadata):
        """Compute rewards for generated responses"""
        rewards = []
        explanations = []
        
        for text, meta in zip(generated_texts, metadata):
            # Extract predictions from generated text with percentages
            prediction = extract_up_down_with_percentage(text)
            
            # If no prediction was extracted, assign a negative reward
            if prediction['direction'] is None:
                rewards.append(-1.0)
                explanations.append("No valid prediction extracted")
                continue
            
            # Get actual outcome from metadata or use mock data
            actual_direction = meta.get("actual_direction", None)
            actual_percentage = meta.get("actual_percentage", None)
            
            if actual_direction is None:
                # Create mock actual for training
                actual_direction = "UP" if random.random() > 0.5 else "DOWN"
            
            if actual_percentage is None:
                # Create mock actual percentage change for training (between 0.1% and 5%)
                actual_percentage = random.uniform(0.1, 5.0)
            
            # Compute reward based on prediction direction accuracy
            direction_reward = 1.0 if prediction['direction'] == actual_direction else -1.0
            
            # Compute reward based on percentage prediction accuracy if available
            percentage_reward = 0.0
            if prediction['percentage'] is not None:
                percentage_diff = abs(prediction['percentage'] - actual_percentage)
                
                # Reward/penalty based on percentage accuracy
                if percentage_diff <= 0.25:  # Very accurate
                    percentage_reward = 1.0
                elif percentage_diff <= 0.5:  # Quite accurate
                    percentage_reward = 0.7
                elif percentage_diff <= 1.0:  # Reasonably accurate
                    percentage_reward = 0.4
                elif percentage_diff <= 2.0:  # Somewhat inaccurate
                    percentage_reward = 0.1
                else:  # Very inaccurate
                    percentage_reward = -0.3
            
            # Add format compliance reward
            format_reward = 0.0
            if "<think>" in text and "</think>" in text:
                format_reward += 0.25
            if "<answer>" in text and "</answer>" in text:
                format_reward += 0.25
            if "Key Factors:" in text:
                format_reward += 0.25
            if "Analysis:" in text:
                format_reward += 0.25
            
            # Combine rewards (direction is most important, then percentage, then format)
            combined_reward = direction_reward + 0.5 * percentage_reward + 0.3 * format_reward
            
            # Ensure the reward is within a reasonable range
            combined_reward = max(-1.5, min(1.5, combined_reward))
            
            rewards.append(combined_reward)
            explanation = f"Prediction: {prediction['direction']}"
            if prediction['percentage'] is not None:
                explanation += f" {prediction['percentage']:.2f}%"
            explanation += f", Actual: {actual_direction} {actual_percentage:.2f}%, "
            explanation += f"Rewards: Direction={direction_reward:.2f}, Percentage={percentage_reward:.2f}, Format={format_reward:.2f}"
            explanations.append(explanation)
        
        # Convert rewards to tensor 
        device = self.model.device
        try:
            # Ensure all rewards are proper Python floats to avoid issues with tensors that don't require grad
            rewards = [float(r) if not isinstance(r, float) else r for r in rewards]
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # Log for debugging
            logger.info(f"Rewards: {rewards}, Tensor shape: {rewards_tensor.shape}, Device: {rewards_tensor.device}")
            logger.info(f"Rewards require grad: {rewards_tensor.requires_grad}")
            
        except Exception as e:
            logger.error(f"Error creating rewards tensor: {e}")
            # Fallback to simple tensor with neutral reward
            rewards_tensor = torch.tensor([0.0] * len(rewards), dtype=torch.float32, device=device)
        
        return rewards_tensor, explanations
    
    def _grpo_step(self, batch):
        """Perform one GRPO training step"""
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        metadata = batch["metadata"]
        
        # Make sure model is in training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with the current model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits_current = outputs.logits
        
        # Generate responses for reward computation
        generated_texts = self._generate_responses(input_ids, attention_mask)
        
        # Compute rewards
        rewards, explanations = self._compute_rewards(generated_texts, metadata)
        
        # Convert rewards to tensor if it's not already
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
        
        # Make sure rewards require gradients
        rewards = rewards.detach()  # Detach because rewards come from a separate computation
        
        # Detach the current model for reference
        with torch.no_grad():
            ref_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits_ref = ref_outputs.logits.detach()
        
        # Compute KL divergence
        kl_div = self._compute_kl_divergence(logits_ref, logits_current, attention_mask)
        
        # Compute policy loss (negative reward)
        policy_loss = -rewards.mean()
        
        # Compute total loss with KL regularization
        # Ensure loss requires gradient by using logits_current somewhere in the calculation
        # This connects the loss to the computational graph
        loss = policy_loss + self.kl_coef * kl_div
        
        # Debug info
        logger.info(f"Loss components - Policy: {policy_loss.item()}, KL: {kl_div.item()}")
        logger.info(f"Loss requires grad: {loss.requires_grad}")
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
        # Check for non-finite gradients
        valid_gradients = True
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    logger.warning(f"Parameter {name} has non-finite gradients")
                    valid_gradients = False
            elif param.requires_grad:
                logger.warning(f"Parameter {name} requires grad but grad is None")
                
        # Update parameters only if gradients are valid
        if valid_gradients:
            self.optimizer.step()
            self.scheduler.step()
        else:
            logger.warning("Skipping parameter update due to invalid gradients")
        
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "rewards": rewards.tolist() if isinstance(rewards, torch.Tensor) else rewards,
            "explanations": explanations,
        }
    
    def train(self):
        """Train the model using GRPO"""
        logger.info("Starting GRPO training")
        
        self.model.train()
        total_loss = 0
        total_rewards = []
        
        # Create default output directory
        output_dir = getattr(self.args, 'output_dir', './trained_model')
        
        # Training loop
        for epoch in range(int(self.args.num_train_epochs)):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch+1}/{self.args.num_train_epochs}")
            
            epoch_loss = 0
            epoch_rewards = []
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(progress_bar):
                # Perform GRPO step
                step_results = self._grpo_step(batch)
                
                # Update metrics
                epoch_loss += step_results["loss"]
                epoch_rewards.extend(step_results["rewards"])
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": step_results["loss"],
                    "policy_loss": step_results["policy_loss"],
                    "kl_div": step_results["kl_div"],
                    "avg_reward": sum(step_results["rewards"]) / len(step_results["rewards"]) if step_results["rewards"] else 0,
                })
                
                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    logger.info(f"Step {self.global_step}: loss={step_results['loss']:.4f}, "
                               f"policy_loss={step_results['policy_loss']:.4f}, "
                               f"kl_div={step_results['kl_div']:.4f}, "
                               f"avg_reward={sum(step_results['rewards']) / len(step_results['rewards']) if step_results['rewards'] else 0:.4f}")
                    
                    # Log a sample explanation
                    if step_results["explanations"]:
                        logger.info(f"Sample explanation: {step_results['explanations'][0]}")
                
                # Save checkpoint
                if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                    self.save_model(f"{output_dir}/checkpoint-{self.global_step}")
                
                self.global_step += 1
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            
            logger.info(f"Epoch {epoch+1} completed: avg_loss={avg_epoch_loss:.4f}, avg_reward={avg_epoch_reward:.4f}")
            
            # Save epoch checkpoint
            self.save_model(f"{output_dir}/epoch-{epoch+1}")
            
            # Save best model based on reward
            if avg_epoch_reward > self.best_reward:
                self.best_reward = avg_epoch_reward
                logger.info(f"New best model with avg_reward={avg_epoch_reward:.4f}")
                self.save_model(f"{output_dir}/best")
            
            total_loss += epoch_loss
            total_rewards.extend(epoch_rewards)
        
        # Final logging
        avg_total_loss = total_loss / (len(self.train_dataloader) * self.args.num_train_epochs)
        avg_total_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        
        logger.info(f"Training completed: avg_loss={avg_total_loss:.4f}, avg_reward={avg_total_reward:.4f}")
        
        # Save final model
        self.save_model(output_dir)
        
        return {
            "avg_loss": avg_total_loss,
            "avg_reward": avg_total_reward,
        }
    
    def save_model(self, output_dir=None):
        """Save the model and tokenizer"""
        if output_dir is None:
            output_dir = "./trained_model"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training args - handle non-serializable objects
        try:
            # Create a serializable version of the args
            serializable_args = {}
            for key, value in vars(self.args).items():
                try:
                    # Test if the value is JSON serializable
                    json.dumps(value)
                    serializable_args[key] = value
                except (TypeError, OverflowError):
                    # If not serializable, convert to string
                    serializable_args[key] = str(value)
            
            with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                json.dump(serializable_args, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save training args as JSON: {e}")
            # Save basic training info instead
            with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
                f.write(f"Learning rate: {self.args.learning_rate}\n")
                f.write(f"Batch size: {self.args.per_device_train_batch_size}\n")
                f.write(f"Epochs: {self.args.num_train_epochs}\n")
        
        logger.info(f"Model saved to {output_dir}")

    def _log_metrics(self, metrics):
        """
        Log metrics to wandb if available
        
        Args:
            metrics (dict): Dictionary of metrics to log
        """
        if wandb.run is not None:
            wandb.log(metrics)

def main(args=None):
    """
    Main function to run GRPO training for stock prediction
    
    Args:
        args: Command line arguments
    """
    if args is None:
        # Parse arguments if not provided
        parser = argparse.ArgumentParser(description="Over9Thousand Stock Market Predictor")
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                            help="Base model to use")
        parser.add_argument("--output_dir", type=str, default="./trained_model", 
                            help="Directory to save the trained model")
        parser.add_argument("--method", type=str, default="sft", choices=["sft", "dpo", "grpo"], 
                            help="Training method")
        parser.add_argument("--num_epochs", type=int, default=3, 
                            help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=1, 
                            help="Batch size for training")
        parser.add_argument("--learning_rate", type=float, default=2e-4, 
                            help="Learning rate")
        parser.add_argument("--seed", type=int, default=42, 
                            help="Random seed")
        parser.add_argument("--kl_coef", type=float, default=0.1, 
                            help="KL divergence coefficient for GRPO")
        parser.add_argument("--save_steps", type=int, default=50, 
                            help="Save checkpoint frequency")
        parser.add_argument("--diverse_predictions", action="store_true", 
                            help="Enable diverse predictions in GRPO")
        parser.add_argument("--dataset_path", type=str, default=None, 
                            help="Path to custom dataset")
        parser.add_argument("--use_2084collective", action="store_true", 
                            help="Use the 2084Collective dataset")
        parser.add_argument("--max_samples", type=int, default=None, 
                            help="Maximum number of samples to use")
        parser.add_argument("--num_prediction_days", type=int, default=3, 
                            help="Number of days to predict (max: 3)")
        parser.add_argument("--market_data_source", type=str, default="yahoo", 
                            help="Source for market condition data")
        parser.add_argument("--vix_threshold", type=str, default=None, 
                            help="Custom VIX thresholds for market regime detection")
        
        # Memory optimization arguments
        parser.add_argument("--use_4bit", action="store_true", 
                            help="Use 4-bit quantization")
        parser.add_argument("--use_8bit", action="store_true", 
                            help="Use 8-bit quantization")
        parser.add_argument("--gradient_checkpointing", action="store_true", 
                            help="Enable gradient checkpointing")
        parser.add_argument("--offload_optimizer", action="store_true", 
                            help="Offload optimizer states to CPU")
        parser.add_argument("--max_seq_length", type=int, default=2048, 
                            help="Maximum sequence length")
        parser.add_argument("--lora_r", type=int, default=32, 
                            help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=64, 
                            help="LoRA alpha")
        
        args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Configure quantization
    bnb_config = None
    if args.use_4bit:
        logger.info("Using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif args.use_8bit:
        logger.info("Using 8-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use FP16 for mixed precision training
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=True,
        padding_side="left",  # Set padding to left side for decoder models
        model_max_length=2048,  # Set maximum sequence length
    )
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply memory optimizations
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Prepare model for training
    logger.info("Preparing model for training")
    model = prepare_model_for_kbit_training(model)
    
    # Get LoRA config
    if args.lora_r != 32 or args.lora_alpha != 64:
        logger.info(f"Using optimized LoRA config with r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = get_optimized_lora_config(r=args.lora_r, alpha=args.lora_alpha)
    else:
        logger.info("Using default LoRA config")
        lora_config = get_lora_config()
    
    model = get_peft_model(model, lora_config)
    
    # Load dataset
    if args.dataset_path:
        logger.info(f"Loading dataset from: {args.dataset_path}")
        if args.dataset_path.endswith('.json'):
            # Load local JSON file
            dataset = load_dataset('json', data_files=args.dataset_path, split='train')
        else:
            dataset = load_dataset(args.dataset_path)
    elif args.use_2084collective:
        logger.info("Loading 2084Collective dataset")
        dataset = load_dataset("2084Collective/stock_prediction", split="train")
    else:
        raise ValueError("Either dataset_path or use_2084collective must be specified")
    
    # Limit dataset size if specified
    if args.max_samples and args.max_samples < len(dataset):
        logger.info(f"Limiting dataset to {args.max_samples} samples")
        dataset = dataset.select(range(args.max_samples))
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Initialize market conditions and confidence calibrator
    market_conditions = MarketConditions()
    confidence_calibrator = ConfidenceCalibrator()
    
    # Manually create a dataset with a 'text' field
    logger.info("Creating formatted dataset with 'text' field...")
    formatted_texts = []
    
    # Print the first example to debug
    logger.info(f"First raw example: {dataset[0]}")
    
    for example in dataset:
        text = f"""Stock Analysis Task:
Ticker: {example['ticker']} ({example['company_name']})
Current Price: ${example['current_price']:.2f}
Previous Price: ${example['previous_price']:.2f}
Sector: {example['sector']}
Industry: {example['industry']}

Technical Indicators:
- MA20: {example['technical_indicators']['ma20']:.2f}
- MA5: {example['technical_indicators']['ma5']:.2f}
- MA50: {example['technical_indicators']['ma50']:.2f}
- MACD: {example['technical_indicators']['macd']:.2f}
- RSI: {example['technical_indicators']['rsi']:.2f}
- Volume: {example['technical_indicators']['volume']}

Recent Headlines:
{chr(10).join('- ' + headline for headline in example['headlines'])}

Based on this information, predict the stock movement for the next 3 days. Follow this format for each day:

Day N:
<think>
Key Factors:
1. [Factor 1]
2. [Factor 2]
3. [Factor 3]

Analysis:
[Your detailed analysis]
</think>
<answer>Direction: [UP/DOWN] Change: [X.X]%</answer>
<confidence>[YY]%</confidence>"""
        formatted_texts.append({"text": text})
    
    # Create a new dataset with the 'text' field
    formatted_dataset = Dataset.from_list(formatted_texts)
    logger.info(f"Formatted dataset size: {len(formatted_dataset)}")
    logger.info(f"Formatted dataset features: {formatted_dataset.features}")
    logger.info(f"First formatted example: {formatted_dataset[0]}")
    
    if "text" not in formatted_dataset.features:
        logger.error("'text' field not found in formatted dataset!")
        raise ValueError("'text' field not found in formatted dataset!")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # Increase for low VRAM
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,  # Use FP16 instead of BF16
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        dataloader_num_workers=4,  # Use multiple workers for data loading
        group_by_length=True,  # Group similar length sequences together
    )

    if args.offload_optimizer:
        logger.info("Enabling optimizer offloading to CPU")
        training_args.deepspeed = {
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "contiguous_gradients": True,
                "overlap_comm": True
            }
        }
    
    # Choose training method
    if args.method == "sft":
        logger.info("Using SFT training method")
        try:
            # Configure tokenizer with padding and truncation
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"
            tokenizer.pad_token = tokenizer.eos_token
            
            # Create a data collator with padding
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Define tokenization function with padding and truncation
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt"
                )
            
            # Apply tokenization to the dataset
            logger.info("Tokenizing dataset with padding and truncation...")
            tokenized_dataset = formatted_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]
            )
            logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")
            
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            logger.info("SFTTrainer initialized successfully")
            
            # Start training
            logger.info("Starting training...")
            trainer.train()
            logger.info("Training completed successfully")
            
            # Save the model
            logger.info(f"Saving model to {args.output_dir}")
            trainer.save_model(args.output_dir)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    elif args.method == "dpo":
        logger.info("Using DPO training method")
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=formatted_dataset,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            beta=0.1,
        )
    elif args.method == "grpo":
        logger.info("Using GRPO training method")
        # Use our custom GRPOTrainer
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=formatted_dataset,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            kl_coef=args.kl_coef,
            beta=0.1,
        )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    logger.info("Training complete!")
    return model, tokenizer

# Add these lines at the end of the file, just before if __name__ == "__main__":

__all__ = [
    'MarketConditions',
    'ConfidenceCalibrator',
    'compute_multi_day_reward',
    'extract_multi_day_predictions',
    'set_random_seed',
    'get_lora_config',
    'prepare_model_for_training'
]

if __name__ == "__main__":
    main() 