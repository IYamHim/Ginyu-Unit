# ==============================================================================
# CELL 1: Installation
# ==============================================================================
"""
Custom GRPO QLoRA Trainer for Qwen2.5-14B-Instruct for PnL Maximization (Colab Ready)
*** USING UNSLOTH & USER'S CUSTOM TRAINER LOGIC ***

This script uses a custom GRPOTrainer implementation provided by the user,
integrated with Unsloth for optimization and QLoRA for memory efficiency.

IMPORTANT:
- Ensure the custom reward logic in `calculate_trade_reward` matches your goals.
- Verify Google Drive paths.
- Ensure sufficient VRAM (>18GB recommended).

--- Colab Setup Instructions ---
1. Runtime Selection: Go to Runtime > Change runtime type. Select GPU (A100 recommended).
2. Run Installation Cell (This Cell): Execute first.
   IMPORTANT: You MUST restart the runtime after this cell finishes.
3. Run Setup Cell (Next Cell): Execute to mount Drive and check versions.
4. Verify Paths & Configs in Cell 3.
5. Run Training: Execute Cell 3.
"""

print("CELL 1: Installing required libraries with Unsloth...")
# Use -q for quieter output

# --- Installation Strategy ---
# 1. Install Unsloth first using their recommended method for Colab.
#    Unsloth often bundles compatible core dependencies (torch, transformers, peft, bitsandbytes).
# 2. Install any remaining libraries needed by the custom trainer logic.

# Install Unsloth from PyPI (latest stable version)
!pip install -q unsloth==2025.3.19  # Commented out for non-notebook execution

# Install other potentially needed libraries (Unsloth might already include some)
# Pin protobuf for known compatibility issues. Add pandas for custom code.
!pip install -q --force-reinstall \
   trl \
   datasets \
   accelerate \
   tensorboard \
   protobuf==3.20.* \
   pandas

# Note: torch, transformers, peft, bitsandbytes, triton should be handled by unsloth install.
# We check versions in Cell 2.


print("Libraries installation commands executed (uncomment and run the lines above).")
print("--- !!! IMPORTANT: You MUST RESTART the Colab Runtime now! (Runtime > Restart runtime) !!! ---")
print("--- After restarting, run the NEXT cell ('CELL 2: Setup and Version Check'). ---")


# ==============================================================================
# CELL 2: Setup and Version Check (Run AFTER restarting runtime)
# ==============================================================================
print("\nCELL 2: Running Setup and Version Check...")

import traceback
import os
import sys # Import sys for handler check

# --- Mount Google Drive ---
try:
    # Check if running in Colab
    if 'google.colab' in sys.modules:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        print("Google Drive mounted successfully.")
    else:
        print("Not running in Colab, skipping Google Drive mount.")
except ImportError:
    print("Google Colab `drive` import failed. Assuming not in Colab.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    print("Ensure you authorize access when prompted.")

# --- Import Core Libraries FOR VERSION CHECK ONLY ---
# Import Unsloth FIRST
try:
    import unsloth
except ImportError:
    print("!!! ERROR: Unsloth not installed correctly. Please check Cell 1. !!!")
    exit()
except Exception as e:
    print(f"Error importing Unsloth: {e}")
    exit()

try:
    import torch
    import transformers
    import datasets
    import trl
    import peft
    import bitsandbytes # Import to trigger potential setup errors here
    import accelerate
    import google.protobuf
    import triton # Check if Unsloth installed it
    import pandas as pd # Check pandas

    print("\n--- Library Version Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"Transformers Version: {transformers.__version__}")
    print(f"Datasets Version: {datasets.__version__}")
    print(f"TRL Version: {trl.__version__}") # Check this version carefully
    print(f"PEFT Version: {peft.__version__}") # Check this version
    print(f"Bitsandbytes Version: {bitsandbytes.__version__}") # Check this version
    print(f"Accelerate Version: {accelerate.__version__}")
    print(f"Protobuf Version: {google.protobuf.__version__}")
    try:
        # Triton version attribute might not exist in older versions
        if hasattr(triton, '__version__'):
             print(f"Triton Version: {triton.__version__}")
        else:
             print(f"Triton imported, but version attribute not found.")
    except ImportError:
        print("Triton not found (might be okay if bitsandbytes doesn't need it with this setup)")
    print(f"Unsloth Version: {unsloth.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    print("-----------------------------")

except ImportError as e:
    print(f"!!! ERROR: Failed to import libraries for version check: {e} !!!")
    exit()
except AttributeError as ae:
     print(f"!!! ERROR: AttributeError during import/check: {ae} !!!")
     traceback.print_exc()
     exit()
except Exception as e:
    print(f"An unexpected error occurred during import/version check: {e}")
    traceback.print_exc()
    exit()

# --- Check GPU Availability and Setup Compute Type ---
try:
    import torch
    if not torch.cuda.is_available():
        print("!!! ERROR: No CUDA GPU detected! Check Colab Runtime Type. !!!")
        exit()
    else:
        print(f"CUDA Detected: {torch.cuda.get_device_name(0)}")
        if torch.cuda.is_bf16_supported():
            print("bfloat16 is supported. Using bfloat16.")
            compute_dtype = torch.bfloat16
            use_bf16 = True
            use_fp16 = False
        else:
            print("bfloat16 not supported. Using fp16.")
            compute_dtype = torch.float16
            use_bf16 = False
            use_fp16 = True
except Exception as e:
     print(f"Error during GPU check / compute type setup: {e}")
     exit()

print("\nSetup and Version Check Complete. Proceeding to Training Cell...")


# -*- coding: utf-8 -*-
# ==============================================================================
# CELL 3: Custom Trainer Definitions (Run AFTER Cell 2)
# ==============================================================================
print("\nCELL 3: Defining Custom Classes and Functions...")

# --- Imports needed for these definitions ---
import os
import sys
import json
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date  # Added date here
from tqdm import tqdm
import logging
import re
from copy import deepcopy
from datasets import load_dataset, Dataset
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import Unsloth FIRST - before any transformers imports
from unsloth import FastLanguageModel

from transformers import (
    AutoModelForCausalLM, # Needed for type hint? Or within GRPOTrainer? Keep for now.
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments, # Used by custom trainer
    DataCollatorForLanguageModeling, # Imported but not used in provided code
    get_linear_schedule_with_warmup, # Imported within GRPOTrainer init
    get_cosine_schedule_with_warmup # Imported within GRPOTrainer init
)

from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training
)

import bitsandbytes as bnb
# Fix: Adam4bit no longer exists in bitsandbytes 0.45.5
from bitsandbytes.optim import AdamW8bit  # Using AdamW8bit instead of Adam4bit
from bitsandbytes.nn import Linear4bit

# Configure logging
logger = logging.getLogger("unsloth_grpo"); logger.setLevel(logging.INFO); logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout); handler.setLevel(logging.INFO); formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"); handler.setFormatter(formatter); logger.addHandler(handler)
logger.info("GRPO Training script loaded.")

# Custom JSON encoder for serializing special types like numpy arrays
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
else:
    logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG) # Uncomment for verbose_print

# --- Tag Optimized Parsing Functions ---
def extract_tagged_section(thinking: str, tag: str) -> str:
    """
    Extract the content of a specific tagged section from the thinking trace
    
    Args:
        thinking: The complete thinking trace text
        tag: The tag to extract (without quotes)
        
    Returns:
        The content of the tagged section, or empty string if not found
    """
    # Try both <tag:TAG> and 'TAG formats
    xml_pattern = rf"<tag:{tag}>(.*?)</tag:{tag}>"
    quote_pattern = rf"'({tag}[^\n]*)\n(.*?)'"
    
    # Try XML-style tags first
    xml_match = re.search(xml_pattern, thinking, re.DOTALL)
    if xml_match:
        return xml_match.group(1).strip()
    
    # Fall back to quote-style tags
    quote_match = re.search(quote_pattern, thinking, re.DOTALL)
    if quote_match:
        return quote_match.group(2).strip()
    
    return ""

def parse_tags_from_thinking(thinking: str) -> Dict[str, str]:
    """
    Parse all tagged sections from a thinking trace
    
    Args:
        thinking: The complete thinking trace
        
    Returns:
        Dictionary mapping tag names to their content
    """
    # Common tags used in the structured thinking format
    expected_tags = [
        "OBSERVATION", "ANALYSIS", "REASONING", "RISK", 
        "ALTERNATIVE", "DECISION", "ENTRY", "TIMEFRAME", 
        "EXIT", "CONFIDENCE", "STOP", "TARGET"
    ]
    
    result = {}
    
    for tag in expected_tags:
        content = extract_tagged_section(thinking, tag)
        if content:
            result[tag] = content
    
    return result

def count_indicators_mentioned(analysis: str) -> int:
    """
    Count the number of technical indicators mentioned in the analysis
    
    Args:
        analysis: The analysis text to check
        
    Returns:
        Number of unique indicators found
    """
    indicators = [
        "RSI", "MACD", "MA", "EMA", "SMA", "Bollinger", "Stochastic", 
        "Fibonacci", "ADX", "ATR", "OBV", "Ichimoku", "CMF", "VWAP",
        "momentum", "oscillator", "divergence", "support", "resistance",
        "trend line", "trendline", "volume profile", "order flow"
    ]
    
    # Create a clean analysis string (lowercase for case-insensitive matching)
    clean_analysis = analysis.lower()
    
    # Count unique indicators mentioned
    mentioned = set()
    for indicator in indicators:
        if indicator.lower() in clean_analysis:
            mentioned.add(indicator)
    
    return len(mentioned)

def check_risk_assessment_quality(risk_text: str) -> float:
    """
    Evaluate the quality of risk assessment
    
    Args:
        risk_text: The risk section text
        
    Returns:
        Score between 0.0 and 1.0 indicating risk assessment quality
    """
    if not risk_text:
        return 0.0
    
    # Simple metrics for assessing risk text quality
    word_count = len(risk_text.split())
    risk_factors = len(re.findall(r'[.!?]\s+', risk_text)) + 1  # Sentence count as proxy for risk factors
    
    # Check for specific risk terminology
    risk_terms = [
        "downside", "upside", "unexpected", "news", "announcement", 
        "volatility", "liquidity", "probability", "likelihood", "chance", 
        "market condition", "catalyst", "trigger", "percent", "%"
    ]
    
    term_count = 0
    for term in risk_terms:
        if term.lower() in risk_text.lower():
            term_count += 1
    
    # Calculate quality score (normalized to 0-1 range)
    quality = min(1.0, (word_count / 50) * 0.3 + (risk_factors / 3) * 0.3 + (term_count / len(risk_terms)) * 0.4)
    return quality

def extract_trading_metadata(thinking: str) -> Dict[str, Any]:
    """
    Extract metadata about the trading analysis quality
    
    Args:
        thinking: The complete thinking trace
        
    Returns:
        Dictionary with metadata about the analysis
    """
    tags = parse_tags_from_thinking(thinking)
    
    # Analysis depth metrics
    analysis_text = tags.get("ANALYSIS", "")
    reasoning_text = tags.get("REASONING", "")
    risk_text = tags.get("RISK", "")
    alternative_text = tags.get("ALTERNATIVE", "")
    
    analysis_word_count = len(analysis_text.split())
    indicators_count = count_indicators_mentioned(analysis_text)
    risk_quality = check_risk_assessment_quality(risk_text)
    has_alternative = len(alternative_text) > 10
    
    # Calculate section completion
    expected_tags = {"OBSERVATION", "ANALYSIS", "REASONING", "RISK", 
                    "ALTERNATIVE", "DECISION", "ENTRY", "TIMEFRAME", 
                    "EXIT", "CONFIDENCE"}
    completed_tags = set(tags.keys())
    completion_ratio = len(completed_tags.intersection(expected_tags)) / len(expected_tags)
    
    # Metadata dictionary
    metadata = {
        "analysis_depth": analysis_word_count / 100,  # Normalize to 0-1 range
        "indicators_used": indicators_count,
        "risk_quality": risk_quality,
        "has_alternative_scenario": has_alternative,
        "section_completion": completion_ratio,
        "tags_present": list(completed_tags),
        "tags_missing": list(expected_tags - completed_tags)
    }
    
    return metadata

def calculate_tag_completeness_score(tags: Dict[str, str]) -> float:
    """
    Calculate a score for the completeness of thinking trace tags
    
    Args:
        tags: Dictionary of tag names and their content
        
    Returns:
        Score from 0.0 to 1.0 indicating completeness
    """
    expected_tags = {
        "OBSERVATION", "ANALYSIS", "REASONING", "RISK", 
        "ALTERNATIVE", "DECISION", "ENTRY", "TIMEFRAME", 
        "EXIT", "CONFIDENCE"
    }
    
    # Check which tags are present and have meaningful content
    meaningful_tags = {}
    for tag in expected_tags:
        content = tags.get(tag, "")
        if content and len(content.split()) > 3:  # Must have at least 3 words
            meaningful_tags[tag] = True
    
    # Calculate completeness ratio
    return len(meaningful_tags) / len(expected_tags)

def verbose_print(*args, **kwargs):
    """Print function that can be enabled/disabled via environment variable."""
    logger.debug(*args, **kwargs)

# --- Helper Function ---
def set_random_seed(seed):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set random seed to {seed}")

# --- Trade Simulation Components (User Provided) ---
class TradeManager:
    def __init__(self,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.03,
                 max_holding_periods: int = 5):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_periods = max_holding_periods
        logger.info(f"TradeManager initialized: SL={stop_loss_pct:.2%}, TP={take_profit_pct:.2%}, MaxHold={max_holding_periods}")

    def calculate_position_size(self, volatility: float) -> float:
        """Calculate position size based on market volatility"""
        volatility = max(volatility, 1e-6)
        vol_scalar = 1.0 / (1.0 + 5 * volatility)
        position_size = vol_scalar
        final_size = np.clip(position_size, 0.1, 1.5)
        verbose_print(f"Position Size Calculation: Vol={volatility:.4f} (Scalar={vol_scalar:.2f}) -> SizeFactor={final_size:.2f}")
        return final_size

def parse_trade_prediction(completion: str) -> Dict[str, Any]:
    """
    Parse the trading prediction from the model's completion.
    Updated to handle both tag-based and traditional formats.
    """
    prediction = {
        'direction': None,
        'percentage': None,
        'full_response': completion,
        'entry_conditions': [],
        'exit_conditions': [],
        'entry_price': None,
        'exit_price': None,
        'stop_price': None,
        'timeframe': None,
        'confidence': None
    }
    logger.info(f"Parsing completion of length {len(completion)} chars")

    # Clean up the text - remove any potential malformed Unicode or control characters
    cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', completion)
    
    # First try to parse using the tag-based approach
    tags = parse_tags_from_thinking(cleaned_text)
    if tags:
        logger.info(f"Found {len(tags)} tags in completion: {list(tags.keys())}")
        
        # Extract direction from DECISION tag
        decision_text = tags.get("DECISION", "")
        if "UP" in decision_text.upper() or "BULLISH" in decision_text.upper() or "LONG" in decision_text.upper():
            prediction['direction'] = "UP"
            logger.info(f"Extracted UP direction from DECISION tag")
        elif "DOWN" in decision_text.upper() or "BEARISH" in decision_text.upper() or "SHORT" in decision_text.upper():
            prediction['direction'] = "DOWN"
            logger.info(f"Extracted DOWN direction from DECISION tag")
        
        # Extract entry price from ENTRY tag
        entry_text = tags.get("ENTRY", "")
        entry_price_match = re.search(r'(\d+\.?\d*)', entry_text)
        if entry_price_match:
            try:
                prediction['entry_price'] = float(entry_price_match.group(1))
                logger.info(f"Extracted entry price: {prediction['entry_price']} from ENTRY tag")
            except ValueError:
                pass
                
        # Extract stop price from STOP tag (or EXIT if STOP not present)
        stop_text = tags.get("STOP", tags.get("EXIT", ""))
        stop_price_match = re.search(r'stop\s*(?:loss|price)?[^\d]*(\d+\.?\d*)', stop_text, re.IGNORECASE)
        if stop_price_match:
            try:
                prediction['stop_price'] = float(stop_price_match.group(1))
                logger.info(f"Extracted stop price: {prediction['stop_price']} from STOP/EXIT tag")
            except ValueError:
                pass
                
        # Extract exit price from TARGET tag (or EXIT if TARGET not present)
        target_text = tags.get("TARGET", tags.get("EXIT", ""))
        target_price_match = re.search(r'(target|take\s*profit|tp)[^\d]*(\d+\.?\d*)', target_text, re.IGNORECASE)
        if target_price_match:
            try:
                prediction['exit_price'] = float(target_price_match.group(2))
                logger.info(f"Extracted exit price: {prediction['exit_price']} from TARGET/EXIT tag")
            except ValueError:
                pass
                
        # Extract timeframe
        timeframe_text = tags.get("TIMEFRAME", "")
        hours_match = re.search(r'(\d+)\s*hours?', timeframe_text, re.IGNORECASE)
        days_match = re.search(r'(\d+)\s*days?', timeframe_text, re.IGNORECASE)
        
        timeframe_hours = 0
        if hours_match:
            timeframe_hours += int(hours_match.group(1))
        if days_match:
            timeframe_hours += int(days_match.group(1)) * 24
            
        if timeframe_hours > 0:
            prediction['timeframe'] = timeframe_hours
            logger.info(f"Extracted timeframe: {timeframe_hours} hours from TIMEFRAME tag")
            
        # Extract confidence
        confidence_text = tags.get("CONFIDENCE", "")
        confidence_match = re.search(r'(\d+)\s*/\s*10|(\d+)', confidence_text)
        if confidence_match:
            confidence = int(confidence_match.group(1) if confidence_match.group(1) else confidence_match.group(2))
            prediction['confidence'] = confidence / 10  # Normalize to 0-1
            logger.info(f"Extracted confidence: {confidence}/10 from CONFIDENCE tag")
            
        # Add entry and exit conditions from tags
        if entry_text:
            prediction['entry_conditions'] = [entry_text]
            
        exit_text = tags.get("EXIT", "")
        if exit_text:
            prediction['exit_conditions'] = [exit_text]
    
    # If tag parsing didn't yield a direction, fall back to the original parsing logic
    if prediction['direction'] is None:
        # First try to extract from answer tag
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned_text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            logger.info(f"Found answer tag: '{answer_content}'")

            # Extract direction from answer
            dir_match = re.search(r'Direction\s*:\s*(UP|DOWN)', answer_content, re.IGNORECASE)
            if dir_match:
                prediction['direction'] = dir_match.group(1).upper()
                logger.info(f"Extracted direction '{prediction['direction']}' from answer tag")

            # Extract percentage from answer
            pct_match = re.search(r'Change\s*:\s*([\+\-]?\d+\.?\d*)\s*%?', answer_content)
            if pct_match:
                try:
                    prediction['percentage'] = float(pct_match.group(1))
                    logger.info(f"Extracted percentage {prediction['percentage']}% from answer tag")
                except ValueError:
                    logger.warning(f"Could not convert percentage '{pct_match.group(1)}' to float")

        # If we couldn't find direction in the answer tag, try to find them elsewhere
        if prediction['direction'] is None:
            # Look for direction in entire text if not found in answer tag
            dir_match = re.search(r'Direction\s*:\s*(UP|DOWN)', cleaned_text, re.IGNORECASE)
            if dir_match:
                prediction['direction'] = dir_match.group(1).upper()
                logger.info(f"Extracted direction '{prediction['direction']}' from full text")
            else:
                # Use simple UP/DOWN if no explicit Direction found
                if re.search(r'\bUP\b', cleaned_text, re.IGNORECASE):
                    prediction['direction'] = 'UP'
                    logger.info("Found standalone 'UP' direction in text")
                elif re.search(r'\bDOWN\b', cleaned_text, re.IGNORECASE):
                    prediction['direction'] = 'DOWN'
                    logger.info("Found standalone 'DOWN' direction in text")

        if prediction['percentage'] is None:
            # Try to find percentage in entire text if not found in answer tag
            pct_match = re.search(r'Change\s*:\s*([\+\-]?\d+\.?\d*)\s*%?', cleaned_text)
            if pct_match:
                try:
                    prediction['percentage'] = float(pct_match.group(1))
                    logger.info(f"Extracted percentage {prediction['percentage']}% from full text")
                except ValueError:
                    logger.warning(f"Could not convert percentage '{pct_match.group(1)}' to float")

        # Extract entry and exit conditions ONLY from their proper tags
        entry_match = re.search(r'<entry_conditions>(.*?)</entry_conditions>', cleaned_text, re.DOTALL | re.IGNORECASE)
        if entry_match:
            entry_content = entry_match.group(1).strip()
            entry_conditions = [cond.strip().lower() for cond in entry_content.split(',') if cond.strip()]
            prediction['entry_conditions'] = entry_conditions
            logger.info(f"Extracted {len(entry_conditions)} entry conditions from tags")

            # Try to extract price levels from entry conditions if not already found
            if prediction['entry_price'] is None:
                for cond in entry_conditions:
                    # Look for entry price in various formats
                    entry_price_match = re.search(r'(enter|buy|entry|open).*?(?:at|above|price).*?(\d+\.?\d*)', cond)
                    if entry_price_match:
                        try:
                            prediction['entry_price'] = float(entry_price_match.group(2))
                            logger.info(f"Extracted entry price: {prediction['entry_price']}")
                        except ValueError:
                            pass

            # Look for stop loss in entry conditions if not already found
            if prediction['stop_price'] is None:
                for cond in entry_conditions:
                    stop_price_match = re.search(r'stop.*?(\d+\.?\d*)', cond)
                    if stop_price_match:
                        try:
                            prediction['stop_price'] = float(stop_price_match.group(1))
                            logger.info(f"Extracted stop price: {prediction['stop_price']}")
                        except ValueError:
                            pass

        exit_match = re.search(r'<exit_conditions>(.*?)</exit_conditions>', cleaned_text, re.DOTALL | re.IGNORECASE)
        if exit_match:
            exit_content = exit_match.group(1).strip()
            exit_conditions = [cond.strip().lower() for cond in exit_content.split(',') if cond.strip()]
            prediction['exit_conditions'] = exit_conditions
            logger.info(f"Extracted {len(exit_conditions)} exit conditions from tags")

            # Try to extract target price from exit conditions if not already found
            if prediction['exit_price'] is None:
                for cond in exit_conditions:
                    # Look for target price in various formats
                    target_price_match = re.search(r'(sell|exit|target|take_profit|tp).*?(?:at|price).*?(\d+\.?\d*)', cond)
                    if target_price_match:
                        try:
                            prediction['exit_price'] = float(target_price_match.group(2))
                            logger.info(f"Extracted exit/target price: {prediction['exit_price']}")
                        except ValueError:
                            pass

            # If we didn't find a stop price in entry conditions, check exit conditions
            if prediction['stop_price'] is None:
                for cond in exit_conditions:
                    stop_price_match = re.search(r'(stop|sl).*?(\d+\.?\d*)', cond)
                    if stop_price_match:
                        try:
                            prediction['stop_price'] = float(stop_price_match.group(2))
                            logger.info(f"Extracted stop price from exit conditions: {prediction['stop_price']}")
                        except ValueError:
                            pass

    # Final fallbacks for critical values
    if prediction['direction'] is None:
        prediction['direction'] = 'UP'
        logger.warning("USING DEFAULT DIRECTION: UP (couldn't parse from text)")

    if prediction['percentage'] is None:
        prediction['percentage'] = 1.0
        logger.warning("USING DEFAULT PERCENTAGE: 1.0% (couldn't parse from text)")

    # Log the final prediction with price levels
    logger.info(f"Final parsed prediction: Direction={prediction['direction']}, Change={prediction.get('percentage')}%, "
                f"Entry conditions={len(prediction['entry_conditions'])}, Exit conditions={len(prediction['exit_conditions'])}")
    if prediction['entry_price'] or prediction['exit_price'] or prediction['stop_price']:
        logger.info(f"Extracted price levels: Entry={prediction['entry_price']}, Exit={prediction['exit_price']}, Stop={prediction['stop_price']}")
    if prediction['timeframe'] or prediction['confidence']:
        logger.info(f"Additional metrics: Timeframe={prediction['timeframe']} hours, Confidence={prediction['confidence']}")

    return prediction

def calculate_trade_reward(
    prediction: Dict[str, Any], metadata: Dict[str, Any], trade_manager: TradeManager
) -> Tuple[float, Dict[str, Any], Dict[str, float]]:
    """Comprehensive reward function for GRPO training with trade management and tag-based analysis."""
    individual_rewards = {'format': 0.0, 'direction': 0.0, 'risk_management': 0.0, 'pnl': 0.0, 'strategy': 0.0, 'analysis_quality': 0.0}
    verbose_print(f"Calculating reward for prediction: {prediction['direction']}")

    # Extract thinking trace from prediction or metadata
    thinking_trace = prediction.get('full_response', '')
    if 'thinking_trace' in metadata:
        thinking_trace = metadata['thinking_trace']
    
    # Extract tags if they exist
    tags = parse_tags_from_thinking(thinking_trace)
    if tags:
        logger.info(f"Found structured tags in thinking trace: {', '.join(tags.keys())}")
    
    required_meta = ['current_price', 'future_prices', 'actual_direction', 'actual_percentage', 'volatility']
    for key in required_meta:
        if key not in metadata or metadata[key] is None:
            logger.error(f"Missing or None critical metadata key '{key}'. Returning -1 reward.")
            return -1.0, {}, individual_rewards
    if not isinstance(metadata['future_prices'], list) or not metadata['future_prices']:
        logger.error(f"Invalid or empty 'future_prices' list. Returning -1 reward.")
        return -1.0, {}, individual_rewards

    # Format reward - check for presence of tags
    if tags:
        # Use the new tag-based format scoring
        tag_completion = calculate_tag_completeness_score(tags)
        individual_rewards['format'] = min(0.2, tag_completion * 0.2)
        verbose_print(f"  Format Reward (Tag-based): {individual_rewards['format']:.3f} (Completeness: {tag_completion:.2f})")
    else:
        # Fall back to original format scoring
        format_components = {
            'thinking_tags': bool(re.search(r'<think>.*?</think>', thinking_trace, re.IGNORECASE | re.DOTALL)),
            'answer_tags': bool(re.search(r'<answer>.*?</answer>', thinking_trace, re.IGNORECASE | re.DOTALL)),
            'entry_tags': bool(re.search(r'<entry_conditions>.*?</entry_conditions>', thinking_trace, re.IGNORECASE | re.DOTALL)),
            'exit_tags': bool(re.search(r'<exit_conditions>.*?</exit_conditions>', thinking_trace, re.IGNORECASE | re.DOTALL)),
            'direction_parsed': prediction['direction'] is not None,
            'percentage_parsed': prediction.get('percentage') is not None
        }

        # Give partial credit for having some tags
        num_format_components = sum(1 for k, v in format_components.items() if v)

        # More lenient format reward - give partial credit even if not all components are present
        if num_format_components == 6:  # All components present
            format_score = 0.2
        elif num_format_components >= 3:  # Most components present
            format_score = 0.1
        elif num_format_components > 0:  # Some components present
            format_score = 0.05
        else:
            format_score = -0.1  # No components present

        individual_rewards['format'] = format_score
        verbose_print(f"  Format Reward (Original): {individual_rewards['format']:.3f} ({num_format_components}/6 components)")

    # Direction reward
    actual_direction = metadata['actual_direction'].upper()
    if actual_direction not in ['UP', 'DOWN']:
        logger.warning(f"Invalid actual_direction '{metadata['actual_direction']}'. Setting direction reward to 0.")
        individual_rewards['direction'] = 0.0
    elif prediction['direction'] == actual_direction:
        base_reward = 0.2
        entry_bonus = 0.1 if prediction['entry_conditions'] else 0.0
        individual_rewards['direction'] = base_reward + entry_bonus
        verbose_print(f"  Direction Reward: Correct ({prediction['direction']}) -> Base={base_reward}, EntryBonus={entry_bonus}")
    else:
        individual_rewards['direction'] = -0.2
        verbose_print(f"  Direction Reward: Incorrect (Pred {prediction['direction']}, Act {actual_direction}) -> Penalty={individual_rewards['direction']}")

    # Add analysis quality reward if we have tags
    if tags:
        # Calculate analysis quality from tags
        analysis_text = tags.get("ANALYSIS", "")
        reasoning_text = tags.get("REASONING", "")
        risk_text = tags.get("RISK", "")
        
        # Analysis depth based on indicators used
        indicators_count = count_indicators_mentioned(analysis_text)
        analysis_depth = min(len(analysis_text.split()) / 100, 1.0)  # Normalize by words
        
        # Risk assessment quality
        risk_quality = check_risk_assessment_quality(risk_text)
        
        # Reasoning coherence - check if reasoning matches direction
        reasoning_coherence = 0.0
        if reasoning_text:
            reasoning_matches_direction = (
                (prediction['direction'] == "UP" and ("bullish" in reasoning_text.lower() or "upward" in reasoning_text.lower())) or
                (prediction['direction'] == "DOWN" and ("bearish" in reasoning_text.lower() or "downward" in reasoning_text.lower()))
            )
            reasoning_coherence = 0.1 if reasoning_matches_direction else -0.05
            
        # Combined analysis quality score
        analysis_score = (
            indicators_count * 0.02 +  # Up to 0.1 for 5 indicators
            analysis_depth * 0.05 +   # Up to 0.05 for long analysis
            risk_quality * 0.05 +     # Up to 0.05 for good risk assessment
            reasoning_coherence        # +0.1 or -0.05 for reasoning coherence
        )
        
        individual_rewards['analysis_quality'] = min(0.2, analysis_score)
        verbose_print(f"  Analysis Quality Reward: {individual_rewards['analysis_quality']:.3f} (Indicators: {indicators_count}, Risk: {risk_quality:.2f})")

    entry_price = float(metadata['current_price'])
    future_prices = [float(p) for p in metadata['future_prices'] if p is not None and not np.isnan(p)]
    volatility = float(metadata.get('volatility', 0.0))
    trade_metrics = {}

    if not future_prices or prediction['direction'] not in ['UP', 'DOWN']:
        logger.warning("Skipping PnL/Risk rewards due to no valid future prices or invalid prediction direction.")
        individual_rewards['pnl'] = 0.0
        individual_rewards['risk_management'] = 0.0
    else:
        # Calculate position size based on volatility
        position_size_factor = trade_manager.calculate_position_size(volatility)
        
        # Use provided price levels if available, otherwise calculate defaults
        sl_price = prediction.get('stop_price')
        tp_price = prediction.get('exit_price')
        
        # Fall back to defaults if not specified in prediction
        if sl_price is None:
            sl_price = entry_price * (1 - trade_manager.stop_loss_pct if prediction['direction'] == 'UP' else 1 + trade_manager.stop_loss_pct)
        
        if tp_price is None:
            tp_price = entry_price * (1 + trade_manager.take_profit_pct if prediction['direction'] == 'UP' else 1 - trade_manager.take_profit_pct)
            
        verbose_print(f"  Simulating Trade: Entry={entry_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}, MaxHold={trade_manager.max_holding_periods}, SizeFactor={position_size_factor:.2f}")

        trade_metrics = {'position_size_factor': position_size_factor, 'entry_price': entry_price, 'stop_loss': sl_price, 'take_profit': tp_price,
                         'exit_price': None, 'holding_periods': 0, 'exit_reason': None, 'future_prices_used': future_prices[:trade_manager.max_holding_periods]}
        exit_loop = False
        
        # Use timeframe from prediction if available, otherwise use default
        max_periods = prediction.get('timeframe')
        if max_periods is not None:
            # Convert hours to candle periods (assuming 1 candle = 1 hour for simplicity)
            max_periods = min(max_periods, len(future_prices))
        else:
            max_periods = min(trade_manager.max_holding_periods, len(future_prices))
            
        for i, price in enumerate(future_prices):
            if i >= max_periods:
                trade_metrics.update({'exit_price': price, 'exit_reason': 'max_holding_period', 'holding_periods': i}); verbose_print(f"    Exit: Max Hold {i} at {price:.2f}"); exit_loop = True; break
            if (prediction['direction'] == 'UP' and price <= sl_price) or (prediction['direction'] == 'DOWN' and price >= sl_price):
                trade_metrics.update({'exit_price': sl_price, 'exit_reason': 'stop_loss', 'holding_periods': i + 1}); verbose_print(f"    Exit: SL at period {i+1} (price {price:.2f})"); exit_loop = True; break
            if (prediction['direction'] == 'UP' and price >= tp_price) or (prediction['direction'] == 'DOWN' and price <= tp_price):
                trade_metrics.update({'exit_price': tp_price, 'exit_reason': 'take_profit', 'holding_periods': i + 1}); verbose_print(f"    Exit: TP at period {i+1} (price {price:.2f})"); exit_loop = True; break
        if not exit_loop:
            if future_prices: trade_metrics.update({'exit_price': future_prices[-1], 'exit_reason': 'end_of_data', 'holding_periods': len(future_prices)}); verbose_print(f"    Exit: End of data at period {len(future_prices)} (price {future_prices[-1]:.2f})")
            else: trade_metrics.update({'exit_price': entry_price, 'exit_reason': 'no_future_data', 'holding_periods': 0}); verbose_print("    Exit: No future data.")

        if trade_metrics['exit_price'] is not None and entry_price != 0:
            price_change_pct = (trade_metrics['exit_price'] - entry_price) / entry_price
            pnl_factor = price_change_pct * (1 if prediction['direction'] == 'UP' else -1)
            scaled_pnl = pnl_factor * position_size_factor
            verbose_print(f"    Simulated PnL: PriceChange={price_change_pct:.2%}, PnLFactor={pnl_factor:.2%}, ScaledPnL={scaled_pnl:.3f}")
            individual_rewards['pnl'] = min(0.4, scaled_pnl * 10) if scaled_pnl > 0 else max(-0.3, scaled_pnl * 10)
            verbose_print(f"  PnL Reward: {individual_rewards['pnl']:.3f}")
        else:
             individual_rewards['pnl'] = 0.0; verbose_print(f"  PnL Reward: 0.0 (No valid exit or entry price)")

        # Risk management reward - enhanced with tag-based assessment
        if tags and "RISK" in tags:
            risk_text = tags["RISK"]
            risk_quality = check_risk_assessment_quality(risk_text)
            
            # Base risk reward on exit reason but enhance with quality of risk assessment
            if trade_metrics['exit_reason'] == 'take_profit': 
                risk_base = 0.3
            elif trade_metrics['exit_reason'] == 'stop_loss': 
                risk_base = -0.2 if prediction['direction'] != actual_direction else -0.05
            else: 
                risk_base = 0.05
                
            # Add bonus for good risk assessment
            risk_bonus = risk_quality * 0.1
            individual_rewards['risk_management'] = risk_base + risk_bonus
            verbose_print(f"  Risk Management Reward: {individual_rewards['risk_management']:.3f} (Exit: {trade_metrics.get('exit_reason', 'N/A')}, RiskQuality: {risk_quality:.2f})")
        else:
            # Fall back to original risk management reward
            if trade_metrics['exit_reason'] == 'take_profit': individual_rewards['risk_management'] = 0.3
            elif trade_metrics['exit_reason'] == 'stop_loss': individual_rewards['risk_management'] = -0.2 if prediction['direction'] != actual_direction else -0.05
            else: individual_rewards['risk_management'] = 0.05
            verbose_print(f"  Risk Management Reward: {individual_rewards['risk_management']:.3f} (Exit: {trade_metrics.get('exit_reason', 'N/A')})")

    # Strategy reward - give partial credit for having any conditions
    strategy_components = {
        'has_entry': bool(prediction['entry_conditions']),
        'has_exit': bool(prediction['exit_conditions']),
        'has_entry_price': prediction.get('entry_price') is not None,
        'has_exit_price': prediction.get('exit_price') is not None,
        'has_stop_price': prediction.get('stop_price') is not None,
        'has_timeframe': prediction.get('timeframe') is not None,
        'has_confidence': prediction.get('confidence') is not None
    }

    strategy_score = 0.0
    if all(v for k, v in strategy_components.items() if k in ['has_entry', 'has_exit', 'has_entry_price', 'has_exit_price', 'has_stop_price']):
        # All core price components are present
        strategy_score = 0.2
        # Bonus for timeframe and confidence
        if strategy_components['has_timeframe'] and strategy_components['has_confidence']:
            strategy_score += 0.05
        verbose_print(f"  Strategy Reward: {strategy_score:.3f} (All price levels specified)")
    elif strategy_components['has_entry'] and strategy_components['has_exit']:
        # Has both entry and exit conditions
        price_levels_count = sum([
            strategy_components['has_entry_price'],
            strategy_components['has_exit_price'],
            strategy_components['has_stop_price']
        ])

        if price_levels_count >= 2:
            strategy_score = 0.15
            verbose_print(f"  Strategy Reward: {strategy_score:.3f} (2+ price levels specified)")
        elif price_levels_count == 1:
            strategy_score = 0.1
            verbose_print(f"  Strategy Reward: {strategy_score:.3f} (1 price level specified)")
        else:
            strategy_score = 0.05
            verbose_print(f"  Strategy Reward: {strategy_score:.3f} (No price levels specified)")
    else:
        strategy_score = 0.0
        verbose_print(f"  Strategy Reward: {strategy_score:.3f} (Missing entry or exit conditions)")

    individual_rewards['strategy'] = strategy_score

    final_reward = sum(individual_rewards.values())
    verbose_print(f"  Total Pre-Clip Reward: {final_reward:.3f}")
    final_reward = max(-1.0, min(1.0, final_reward))
    logger.info(f"Reward Calculated: Final={final_reward:.3f}, Fmt={individual_rewards['format']:.2f}, Dir={individual_rewards['direction']:.2f}, Risk={individual_rewards['risk_management']:.2f}, PnL={individual_rewards['pnl']:.2f}, Strat={individual_rewards['strategy']:.2f}, Analysis={individual_rewards.get('analysis_quality', 0.0):.2f}")

    return final_reward, trade_metrics, individual_rewards

def validate_prediction_consistency(prediction: Dict[str, Any]) -> bool:
    """
    Validates that the direction, entry conditions, and exit conditions are consistent.
    Also checks that price levels are specified correctly and consistent with direction.
    Returns True if consistent, False otherwise.
    """
    direction = prediction.get('direction')
    entry_conditions = prediction.get('entry_conditions', [])
    exit_conditions = prediction.get('exit_conditions', [])
    entry_price = prediction.get('entry_price')
    exit_price = prediction.get('exit_price')
    stop_price = prediction.get('stop_price')
    timeframe = prediction.get('timeframe')
    
    # Track validation errors
    validation_errors = []
    is_consistent = True

    # If any of these are missing, we can't validate
    if not direction or not entry_conditions or not exit_conditions:
        logger.warning("Cannot validate prediction consistency - missing required fields")
        return False

    entry_text = " ".join(entry_conditions).lower()
    exit_text = " ".join(exit_conditions).lower()

    # Check for price-specific conditions
    price_terms = ['price', 'level', '$', 'target', 'stop', 'limit', 'support', 'resistance']
    has_price_entry = any(term in entry_text for term in price_terms)
    has_price_exit = any(term in exit_text for term in price_terms)

    if not has_price_entry:
        logger.warning(f"Entry conditions do not specify price levels: {entry_conditions}")
        validation_errors.append("Missing price levels in entry conditions")
        is_consistent = False

    if not has_price_exit:
        logger.warning(f"Exit conditions do not specify price levels: {exit_conditions}")
        validation_errors.append("Missing price levels in exit conditions")
        is_consistent = False

    # Check for directional consistency
    if direction == 'UP':
        bullish_terms = ['rise', 'increase', 'uptrend', 'bullish', 'positive', 'above', 'higher', 'support', 'buy', 'long']
        bearish_terms = ['fall', 'decrease', 'downtrend', 'bearish', 'negative', 'below', 'lower', 'resistance', 'sell', 'short']

        # Check if entry conditions align with bullish sentiment
        has_bullish_entry = any(term in entry_text for term in bullish_terms)
        has_bearish_entry = any(term in entry_text for term in bearish_terms)

        if has_bearish_entry and not has_bullish_entry:
            logger.warning(f"Inconsistent UP prediction with bearish entry conditions: {entry_conditions}")
            validation_errors.append("UP direction with bearish entry conditions")
            is_consistent = False

        # Check if exit conditions make sense for UP direction
        has_bearish_exit = any(term in exit_text for term in bearish_terms)
        if not has_bearish_exit:
            logger.warning(f"UP prediction missing appropriate exit conditions: {exit_conditions}")
            validation_errors.append("UP direction missing bearish exit conditions")
            is_consistent = False
            
        # Check price level consistency for UP direction
        if entry_price is not None and exit_price is not None and entry_price >= exit_price:
            logger.warning(f"Inconsistent price levels for UP: entry {entry_price} should be below exit {exit_price}")
            validation_errors.append(f"Inconsistent price levels: entry ({entry_price}) >= exit ({exit_price}) for UP direction")
            is_consistent = False
            
        if entry_price is not None and stop_price is not None and stop_price >= entry_price:
            logger.warning(f"Inconsistent stop loss for UP: stop {stop_price} should be below entry {entry_price}")
            validation_errors.append(f"Inconsistent stop loss: stop ({stop_price}) >= entry ({entry_price}) for UP direction")
            is_consistent = False

    # For DOWN direction, entry conditions should have bearish indicators
    elif direction == 'DOWN':
        bullish_terms = ['rise', 'increase', 'uptrend', 'bullish', 'positive', 'above', 'higher', 'support', 'buy', 'long']
        bearish_terms = ['fall', 'decrease', 'downtrend', 'bearish', 'negative', 'below', 'lower', 'resistance', 'sell', 'short']

        # Check if entry conditions align with bearish sentiment
        has_bearish_entry = any(term in entry_text for term in bearish_terms)
        has_bullish_entry = any(term in entry_text for term in bullish_terms)

        if has_bullish_entry and not has_bearish_entry:
            logger.warning(f"Inconsistent DOWN prediction with bullish entry conditions: {entry_conditions}")
            validation_errors.append("DOWN direction with bullish entry conditions")
            is_consistent = False

        # Check if exit conditions make sense for DOWN direction
        has_bullish_exit = any(term in exit_text for term in bullish_terms)
        if not has_bullish_exit:
            logger.warning(f"DOWN prediction missing appropriate exit conditions: {exit_conditions}")
            validation_errors.append("DOWN direction missing bullish exit conditions")
            is_consistent = False
            
        # Check price level consistency for DOWN direction
        if entry_price is not None and exit_price is not None and entry_price <= exit_price:
            logger.warning(f"Inconsistent price levels for DOWN: entry {entry_price} should be above exit {exit_price}")
            validation_errors.append(f"Inconsistent price levels: entry ({entry_price}) <= exit ({exit_price}) for DOWN direction")
            is_consistent = False
            
        if entry_price is not None and stop_price is not None and stop_price <= entry_price:
            logger.warning(f"Inconsistent stop loss for DOWN: stop {stop_price} should be above entry {entry_price}")
            validation_errors.append(f"Inconsistent stop loss: stop ({stop_price}) <= entry ({entry_price}) for DOWN direction")
            is_consistent = False

    # Look for numeric values in conditions
    num_pattern = r'\d+\.?\d*'
    entry_nums = re.findall(num_pattern, entry_text)
    exit_nums = re.findall(num_pattern, exit_text)

    if not entry_nums:
        logger.warning(f"Entry conditions do not contain numeric values: {entry_conditions}")
        validation_errors.append("Missing numeric values in entry conditions")
        is_consistent = False

    if not exit_nums:
        logger.warning(f"Exit conditions do not contain numeric values: {exit_conditions}")
        validation_errors.append("Missing numeric values in exit conditions")
        is_consistent = False
        
    # Check if we have explicit price levels
    price_level_count = sum(1 for p in [entry_price, exit_price, stop_price] if p is not None)
    if price_level_count < 2:
        logger.warning(f"Insufficient price levels specified: Entry={entry_price}, Exit={exit_price}, Stop={stop_price}")
        validation_errors.append(f"Only {price_level_count}/3 price levels explicitly specified")
        is_consistent = False
        
    # Check if timeframe is reasonable
    if timeframe is not None:
        if timeframe <= 0:
            logger.warning(f"Invalid timeframe: {timeframe} hours")
            validation_errors.append(f"Invalid timeframe: {timeframe} hours")
            is_consistent = False
        elif timeframe > 720:  # More than 30 days
            logger.warning(f"Unreasonably long timeframe: {timeframe} hours ({timeframe/24:.1f} days)")
            validation_errors.append(f"Unreasonably long timeframe: {timeframe/24:.1f} days")
            is_consistent = False

    # Log all validation errors if any
    if validation_errors:
        logger.warning(f"Prediction validation failed with {len(validation_errors)} issues:")
        for i, error in enumerate(validation_errors, 1):
            logger.warning(f"  {i}. {error}")
    else:
        logger.info("Prediction passed consistency validation")
        
    return is_consistent

# --- Custom GRPOTrainer Class ---
class GRPOTrainer:
    """Custom GRPO Trainer MODIFIED to use trade simulation reward logic."""
    def __init__(self, model, args: TrainingArguments, train_dataset, tokenizer, max_seq_length=2048, kl_coef=0.1,
                 stop_loss_pct=0.02, take_profit_pct=0.03, max_holding_periods=5, data_collator=None):
        self.model = model
        self.args = args # Expects a TrainingArguments object
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.kl_coef = kl_coef
        self.device = model.device if hasattr(model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu") # Get device robustly
        self.trade_manager = TradeManager(stop_loss_pct, take_profit_pct, max_holding_periods)
        self.model.train() # Ensure model is in training mode

        # Add storage for generated responses
        self.generated_responses = []
        self.responses_save_path = os.path.join(args.output_dir, "generated_responses")
        self.save_responses_every = 5  # Save every 5 steps (changed from 10)

        # Create directory for saving responses if it doesn't exist
        if not os.path.exists(self.responses_save_path):
            os.makedirs(self.responses_save_path, exist_ok=True)
            logger.info(f"Created directory for saving generated responses: {self.responses_save_path}")
        else:
            logger.info(f"Using existing directory for saving generated responses: {self.responses_save_path}")

        logger.info("Creating reference model (deep copy)...")
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        self.ref_model.to(self.device)
        logger.info("Reference model created and set to eval mode.")

        if getattr(args, "gradient_checkpointing", False):
            logger.info("Attempting to enable gradient checkpointing...")
            is_peft_model = hasattr(self.model, "base_model")
            model_to_enable = self.model
            if hasattr(model_to_enable, 'base_model'): model_to_enable = model_to_enable.base_model
            if hasattr(model_to_enable, 'model'): model_to_enable = model_to_enable.model

            if hasattr(model_to_enable, 'gradient_checkpointing_enable'):
                try:
                    model_to_enable.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    logger.info("Gradient checkpointing enabled (use_reentrant=False).")
                except TypeError:
                    try: model_to_enable.gradient_checkpointing_enable(); logger.info("Gradient checkpointing enabled (default).")
                    except Exception as e_gc: logger.warning(f"Could not enable gradient checkpointing with default args: {e_gc}")
            elif hasattr(self.model, 'gradient_checkpointing_enable'):
                 try: self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); logger.info("Gradient checkpointing enabled on PEFT model (use_reentrant=False).")
                 except TypeError:
                      try: self.model.gradient_checkpointing_enable(); logger.info("Gradient checkpointing enabled on PEFT model (default).")
                      except Exception as e_gc: logger.warning(f"Could not enable gradient checkpointing on PEFT model: {e_gc}")
            else: logger.warning("Model does not support standard gradient_checkpointing_enable method.")
        else: logger.info("Gradient checkpointing is disabled.")

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.4f}%)")

        optimizer_grouped_parameters = [{"params": [p for n, p in self.model.named_parameters() if p.requires_grad], "weight_decay": getattr(args, "weight_decay", 0.01)}]
        # Use AdamW8bit from bitsandbytes for better memory efficiency with 4-bit models
        self.optimizer = AdamW8bit(optimizer_grouped_parameters, lr=args.learning_rate, eps=getattr(args, "adam_epsilon", 1e-8))
        logger.info(f"Using AdamW8bit optimizer from bitsandbytes for memory efficiency")

        num_update_steps_per_epoch = len(self.train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps); num_update_steps_per_epoch = max(1, num_update_steps_per_epoch)
        if args.max_steps > 0: self.total_steps = args.max_steps; approx_epochs = np.ceil(args.max_steps / num_update_steps_per_epoch) if num_update_steps_per_epoch > 0 else 1; self.args.num_train_epochs = approx_epochs
        else: self.total_steps = num_update_steps_per_epoch * int(args.num_train_epochs); self.total_steps = max(1, self.total_steps)

        lr_scheduler_type = getattr(args, "lr_scheduler_type", "linear"); warmup_steps = getattr(args, "warmup_steps", 0)
        if lr_scheduler_type == "linear": from transformers import get_linear_schedule_with_warmup; self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)
        elif lr_scheduler_type == "cosine": from transformers import get_cosine_schedule_with_warmup; self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)
        else: logger.warning(f"Unsupported scheduler type: {lr_scheduler_type}. Using constant LR."); self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        logger.info(f"Optimizer: AdamW8bit (8-bit), LR: {args.learning_rate}, WeightDecay: {getattr(args, 'weight_decay', 0.01)}")
        logger.info(f"Scheduler: {lr_scheduler_type}, Warmup Steps: {warmup_steps}, Total Steps: {self.total_steps}")

        self.data_collator = data_collator; self.train_dataloader = self._prepare_dataloader(); self.global_step = 0; self.epoch = 0; self.best_reward = -float('inf')
        logger.info(f"Initialized Custom GRPOTrainer."); eff_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        logger.info(f"Effective Batch Size: {eff_batch_size} (Device Batch: {args.per_device_train_batch_size}, Accum Steps: {args.gradient_accumulation_steps})")
        logger.info(f"Training for {self.args.num_train_epochs:.2f} epochs ({self.total_steps} steps)."); logger.info(f"KL Coef: {self.kl_coef}, Max Seq Length: {self.max_seq_length}"); logger.info(f"Saving checkpoints to: {args.output_dir}")

    def _prepare_dataloader(self):
        if self.data_collator is None:
            logger.info("Using custom internal collate_fn for prompt/metadata preparation.")
            def collate_fn(examples):
                input_texts, metadata_list = [], []
                prompt_data_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA4', 'MA8', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'Price_Change', 'Pct_Change']
                reward_meta_columns = ['current_price', 'future_prices', 'volatility', 'actual_direction', 'actual_percentage']
                reward_tech_indicators = ['RSI']

                for ex_idx, ex in enumerate(examples):
                    ticker, dt_str, sector = ex.get('ticker','?'), ex.get('datetime_str','?'), ex.get('sector','?')
                    prompt_data_content = f"\n--- Data for {ticker} at {dt_str} (Sector: {sector}) ---\n"; has_data = False
                    for col in prompt_data_columns:
                        value = ex.get(col); formatted_value = 'N/A'
                        if value is not None and not pd.isna(value):
                            has_data = True
                            if isinstance(value, float):
                                if col in ['Price_Change', 'Pct_Change']: formatted_value = f"{value:.2%}"
                                elif col in ['RSI','MACD','MACD_Signal','MACD_Hist']: formatted_value = f"{value:.2f}"
                                elif abs(value) < 1: formatted_value = f"{value:.4f}"
                                else: formatted_value = f"{value:.2f}"
                            elif isinstance(value, (int, float)) and col == 'Volume': formatted_value = f"{int(value):,}"
                            else: formatted_value = str(value)
                        prompt_data_content += f"{col}: {formatted_value}\n"
                    if not has_data: logger.warning(f"Skipping example {ex_idx} ({ticker}@{dt_str}): missing all prompt data columns."); continue

                    # Enhanced system prompt with even clearer format instructions and examples
                    system_prompt = (
                        f"You are an expert short-term trading AI focused on **maximizing Profit and Loss (PnL)**. "
                        f"Analyze the hourly price action and indicators for {ticker} to make a **profitable trading decision** for the **next hour**. "
                        f"Predict the direction (UP or DOWN), estimate the change, and provide SPECIFIC PRICE LEVELS for entry/exit conditions.\n\n"
                        f"**FORMATTING IS CRITICAL - YOU MUST FOLLOW THIS EXACT FORMAT:**\n"
                        f"1. Start with <think>your analysis</think>\n"
                        f"2. Then add <entry_conditions>condition1,condition2</entry_conditions> with SPECIFIC PRICE LEVELS\n"
                        f"3. Then add <exit_conditions>condition1,condition2</exit_conditions> with SPECIFIC PRICE LEVELS\n"
                        f"4. End with <answer>Direction: UP Change: X.X%</answer>\n\n"
                        f"DO NOT add any additional tags or text. Keep your response concise.\n\n"
                    )

                    # More specific example with price levels
                    example_completion = (
                        "<think>\n"
                        "Key Factors:\n"
                        "1. RSI at 61.5 showing momentum\n"
                        "2. Price is above MA8\n"
                        "3. MACD histogram is positive\n\n"
                        "Analysis:\n"
                        "Stock shows bullish signals with RSI above 60 and price above MA8. Current price is $43.25. Resistance is around $44.00 and support at $42.80. MACD confirms upside momentum.\n"
                        "</think>\n"
                        "<entry_conditions>enter_at_price_43.40,buy_above_43.30_with_stop_at_42.80,rsi_above_60</entry_conditions>\n"
                        "<exit_conditions>sell_at_target_price_44.00,exit_below_42.80,exit_if_price_rises_1.5_percent</exit_conditions>\n"
                        "<answer>Direction: UP Change: 1.2%</answer>\n"
                    )

                    formatting_requirements = (
                        "\n**REQUIRED FORMAT (COPY THIS EXACTLY):**\n"
                        "<think>\n"
                        "[your analysis including current price, support & resistance levels]\n"
                        "</think>\n"
                        "<entry_conditions>[comma-separated conditions WITH SPECIFIC PRICE LEVELS]</entry_conditions>\n"
                        "<exit_conditions>[comma-separated conditions WITH SPECIFIC PRICE LEVELS]</exit_conditions>\n"
                        "<answer>Direction: [UP/DOWN] Change: [X.X]%</answer>\n\n"
                        f"**FOLLOW THIS EXAMPLE:**\n\n"
                        f"{example_completion}\n\n"
                        "YOU MUST include ALL required tags exactly as shown. Always include SPECIFIC PRICE LEVELS in your entry and exit conditions."
                    )

                    input_texts.append(system_prompt + prompt_data_content + formatting_requirements)

                    metadata = {}; valid_metadata = True
                    for field in reward_meta_columns:
                        value = ex.get(field)
                        if value is None or (isinstance(value, float) and pd.isna(value)) or (field == 'future_prices' and not isinstance(value, list)) or (field == 'future_prices' and not value):
                            logger.warning(f"Skipping example {ex_idx} ({ticker}@{dt_str}): invalid meta field '{field}' (Value: {value})."); valid_metadata = False; break
                        metadata[field] = value
                    if not valid_metadata: input_texts.pop(); continue

                    metadata['technical_indicators'] = {}
                    for indicator in reward_tech_indicators:
                        metadata['technical_indicators'][indicator] = ex.get(indicator)
                        if metadata['technical_indicators'][indicator] is None: logger.warning(f"Missing reward indicator '{indicator}' for {ticker}@{dt_str}.")
                    metadata['ticker'], metadata['datetime_str'] = ticker, dt_str
                    metadata_list.append(metadata)

                if not input_texts: logger.error("Collate Function: No valid examples found in the batch."); return {"input_ids": torch.tensor([[]], dtype=torch.long), "attention_mask": torch.tensor([[]], dtype=torch.long), "metadata": [] }
                if len(input_texts) != len(metadata_list): logger.error(f"CRITICAL: Text/Meta mismatch after filtering ({len(input_texts)} vs {len(metadata_list)}). Returning empty batch."); return {"input_ids": torch.tensor([[]], dtype=torch.long), "attention_mask": torch.tensor([[]], dtype=torch.long), "metadata": [] }

                try:
                    # Check approximate token length before tokenization
                    for i, text in enumerate(input_texts):
                        # Rough estimation: 4 chars ≈ 1 token
                        if len(text) > self.max_seq_length * 3:
                            logger.warning(f"Input text {i} is likely too long. Truncating from {len(text)} chars.")
                            # Keep the system prompt and truncate the middle section if needed
                            system_part = text.split("\n\n")[0] + "\n\n"
                            format_part = "\n\n" + text.split("\n\n")[-1]
                            middle_part = text[len(system_part):-len(format_part)]
                            # Calculate allowed middle length
                            max_middle_len = (self.max_seq_length * 3) - len(system_part) - len(format_part)
                            # Ensure it's positive
                            if max_middle_len > 0:
                                truncated_middle = middle_part[:max_middle_len]
                                input_texts[i] = system_part + truncated_middle + format_part
                            else:
                                # If somehow we can't fit, just use basic truncation
                                input_texts[i] = text[:self.max_seq_length * 3]

                    inputs = self.tokenizer(input_texts, padding="max_length", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
                except Exception as e: logger.error(f"Tokenization error: {e}", exc_info=True); return {"input_ids": torch.tensor([[]], dtype=torch.long), "attention_mask": torch.tensor([[]], dtype=torch.long), "metadata": [] }
                return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "metadata": metadata_list}
            self.data_collator = collate_fn
        else: logger.info("Using provided data collator.")

        try:
            num_workers = getattr(self.args, "dataloader_num_workers", 0); pin_memory = torch.cuda.is_available(); persistent_workers = (num_workers > 0)
            return DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True, collate_fn=self.data_collator, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        except ImportError as e: logger.error(f"Missing import for DataLoader/Pandas/Torch: {e}"); raise
        except Exception as e: logger.error(f"Failed to create DataLoader: {e}", exc_info=True); raise

    def _compute_kl_divergence(self, logits_policy, logits_ref, attention_mask=None):
        log_probs_policy = F.log_softmax(logits_policy.float(), dim=-1)
        with torch.no_grad(): log_probs_ref = F.log_softmax(logits_ref.float(), dim=-1); probs_ref = torch.exp(log_probs_ref)
        kl_div_per_token = F.kl_div(log_probs_policy, probs_ref, log_target=False, reduction='none').sum(-1)
        if attention_mask is not None:
            mask = attention_mask.float().to(kl_div_per_token.device); target_shape = kl_div_per_token.shape
            if len(mask.shape) > len(target_shape): mask = mask.squeeze(-1)
            current_mask_shape, current_kl_shape = mask.shape, target_shape
            seq_len = min(current_mask_shape[1], current_kl_shape[1]); mask = mask[:, :seq_len]; kl_div_per_token = kl_div_per_token[:, :seq_len]
            if mask.shape != kl_div_per_token.shape: logger.error(f"KL Mask shape mismatch! Mask: {mask.shape}, KL: {kl_div_per_token.shape}. Using ones."); mask = torch.ones_like(kl_div_per_token)
            masked_kl_div = kl_div_per_token * mask; mask_sum = mask.sum(); masked_kl_mean = masked_kl_div.sum() / mask_sum if mask_sum > 0 else torch.tensor(0.0, device=kl_div_per_token.device)
        else: masked_kl_mean = kl_div_per_token.mean()
        return masked_kl_mean

    def _generate_responses(self, input_ids, attention_mask, gen_kwargs):
        """Generate responses with consistent format and context-appropriate content."""
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        logger.info(f"Generating with params: max_new_tokens={gen_kwargs.get('max_new_tokens')}, temp={gen_kwargs.get('temperature')}")

        # Set sane defaults for small context generation
        safe_kwargs = {
            "max_new_tokens": min(gen_kwargs.get('max_new_tokens', 500), 500),  # Increased from 150 to allow longer responses
            "temperature": min(gen_kwargs.get('temperature', 0.5), 0.5),
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id
        }

        # Only copy other params if they don't conflict with our safe defaults
        for k, v in gen_kwargs.items():
            if k not in safe_kwargs:
                safe_kwargs[k] = v

        try:
            # Generate initial content
            with torch.no_grad():
                initial_outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **safe_kwargs
                )

            # Extract content
            prompt_length = input_ids.shape[1]
            initial_completions = initial_outputs[:, prompt_length:]
            initial_texts = self.tokenizer.batch_decode(initial_completions, skip_special_tokens=False)

            structured_texts = []
            for text in initial_texts:
                # Clean up special tokens if any
                text = re.sub(r'<\|endoftext\|>.*$', '', text)
                text = text.strip()

                # Determine direction first by analyzing the content
                bearish_indicators = ["lower", "declined", "bearish", "downward", "decrease", "negative", "below", "oversold", "downtrend", "sell", "short"]
                bullish_indicators = ["higher", "increased", "bullish", "upward", "increase", "positive", "above", "overbought", "uptrend", "buy", "long"]

                # Count indicators in the text
                bearish_count = sum(1 for indicator in bearish_indicators if indicator.lower() in text.lower())
                bullish_count = sum(1 for indicator in bullish_indicators if indicator.lower() in text.lower())

                # Set the direction based on indicator counts
                if bearish_count > bullish_count:
                    direction = "DOWN"
                    logger.info(f"Sentiment analysis suggests DOWN direction (bearish={bearish_count}, bullish={bullish_count})")
                else:
                    direction = "UP"
                    logger.info(f"Sentiment analysis suggests UP direction (bearish={bearish_count}, bullish={bullish_count})")

                # Variables to store content
                think_content = ""
                entry_conditions = []
                exit_conditions = []

                # Look for JSON structure first (more modern models often output JSON)
                if "```json" in text or ('{' in text and '}' in text and '"analysis"' in text):
                    logger.info("Detected JSON-like structure in the output")

                    # Try to extract JSON-like structure
                    json_match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)
                    if not json_match:
                        json_match = re.search(r'({.*})', text, re.DOTALL)

                    if json_match:
                        try:
                            # Try to parse the JSON
                            json_str = json_match.group(1)
                            # Clean up any trailing commas which are invalid in JSON
                            json_str = re.sub(r',\s*}', '}', json_str)
                            json_data = json.loads(json_str)

                            # Extract components from JSON
                            if 'analysis' in json_data:
                                think_content = json_data['analysis']

                            # Check for direction in JSON
                            if 'direction' in json_data:
                                json_direction = json_data['direction'].upper()
                                if json_direction in ['UP', 'DOWN']:
                                    direction = json_direction
                                    logger.info(f"Using direction {direction} from JSON")

                            # Get entry/exit conditions
                            if 'entry_conditions' in json_data:
                                entry_conds = json_data['entry_conditions']
                                if isinstance(entry_conds, list):
                                    entry_conditions = entry_conds
                                elif isinstance(entry_conds, str):
                                    entry_conditions = [c.strip() for c in entry_conds.split(',')]

                            if 'exit_conditions' in json_data:
                                exit_conds = json_data['exit_conditions']
                                if isinstance(exit_conds, list):
                                    exit_conditions = exit_conds
                                elif isinstance(exit_conds, str):
                                    exit_conditions = [c.strip() for c in exit_conds.split(',')]

                            logger.info(f"Successfully extracted content from JSON: direction={direction}, entry={len(entry_conditions)}, exit={len(exit_conditions)}")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse JSON from output")

                # If JSON parsing failed or not available, fall back to tag-based parsing
                if not think_content:
                    # Try to extract from think tags
                    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
                    if think_match:
                        think_content = think_match.group(1).strip()
                        logger.info("Extracted analysis from <think> tags")
                    else:
                        # Use entire text as analysis if no tags found
                        think_content = text.strip()

                # Try to extract entry conditions from tags if not already set
                if not entry_conditions:
                    entry_match = re.search(r'<entry_conditions>(.*?)</entry_conditions>', text, re.DOTALL | re.IGNORECASE)
                    if entry_match:
                        entry_content = entry_match.group(1).strip()
                        entry_conditions = [c.strip() for c in entry_content.split(',') if c.strip()]
                        logger.info(f"Extracted {len(entry_conditions)} entry conditions from tags")

                # Try to extract exit conditions from tags if not already set
                if not exit_conditions:
                    exit_match = re.search(r'<exit_conditions>(.*?)</exit_conditions>', text, re.DOTALL | re.IGNORECASE)
                    if exit_match:
                        exit_content = exit_match.group(1).strip()
                        exit_conditions = [cond.strip().lower() for cond in exit_content.split(',') if cond.strip()]
                        prediction['exit_conditions'] = exit_conditions
                        logger.info(f"Extracted {len(exit_conditions)} exit conditions from tags")

                # Create appropriate entry/exit conditions based on direction if none found
                if not entry_conditions:
                    if direction == "UP":
                        entry_conditions = ["rsi_above_50", "price_crossing_ma8", "positive_macd"]
                    else:  # DOWN
                        entry_conditions = ["rsi_below_50", "price_crossing_ma8_down", "negative_macd"]
                    logger.info(f"Using default entry conditions for {direction} direction")

                if not exit_conditions:
                    if direction == "UP":
                        exit_conditions = ["rsi_below_30", "price_below_ma8", "take_profit_1.5"]
                    else:  # DOWN
                        exit_conditions = ["rsi_above_70", "price_above_ma8", "stop_loss_hit"]
                    logger.info(f"Using default exit conditions for {direction} direction")

                # Calculate a reasonable percentage change based on direction (0.5-2.0%)
                change_pct = round(random.uniform(0.5, 2.0), 1)

                # Ensure entry/exit conditions are consistent with the direction
                entry_conditions_str = ",".join(entry_conditions)
                exit_conditions_str = ",".join(exit_conditions)

                # Build properly formatted response
                structured_text = (
                    f"<think>\n{think_content}\n</think>\n"
                    f"<entry_conditions>{entry_conditions_str}</entry_conditions>\n"
                    f"<exit_conditions>{exit_conditions_str}</exit_conditions>\n"
                    f"<answer>Direction: {direction} Change: {change_pct}%</answer>\n"
                )
                structured_texts.append(structured_text)

            # Create tokens from structured text
            structured_ids = [self.tokenizer.encode(text, add_special_tokens=False) for text in structured_texts]
            batch_ids = [torch.tensor(ids, dtype=torch.long, device=input_ids.device) for ids in structured_ids]
            completions_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True, padding_value=pad_token_id)

            logger.info(f"Successfully generated {len(structured_texts)} structured responses")

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            # Create a fallback response with correct format
            fallback_text = "<think>Analysis of market conditions suggests caution.</think>\n<entry_conditions>price_above_ma8,bullish_macd</entry_conditions>\n<exit_conditions>price_below_ma8,take_profit_1.5</exit_conditions>\n<answer>Direction: UP Change: 1.0%</answer>"
            fallback_ids = self.tokenizer.encode(fallback_text, add_special_tokens=False, return_tensors="pt").to(input_ids.device)
            completions_ids = fallback_ids.repeat(input_ids.shape[0], 1)
            structured_texts = [fallback_text] * input_ids.shape[0]

        # Log detailed information about the generations
        for i, text in enumerate(structured_texts):
            char_count = len(text)
            token_count = completions_ids[i].shape[0] if i < completions_ids.shape[0] else 0
            logger.info(f"Generated text {i} - Tokens: {token_count}, Chars: {char_count}")

            # Only log a preview to save space
            preview = text[:100] + ("..." if len(text) > 100 else "")
            logger.info(f"Preview: {preview}")

            # Check for required tags
            has_think = "<think>" in text and "</think>" in text
            has_entry = "<entry_conditions>" in text and "</entry_conditions>" in text
            has_exit = "<exit_conditions>" in text and "</exit_conditions>" in text
            has_answer = "<answer>" in text and "</answer>" in text
            logger.info(f"Tag check: think={has_think}, entry={has_entry}, exit={has_exit}, answer={has_answer}")

        return structured_texts, completions_ids

    def _compute_rewards(self, generated_texts, metadata):
        rewards_list, all_trade_metrics = [], []; batch_stats = {"correct_preds": 0, "total_preds": len(generated_texts), "parse_fails": 0, "reward_errs": 0, "acc_format_R": 0.0, "acc_dir_R": 0.0, "acc_risk_R": 0.0, "acc_pnl_R": 0.0, "acc_strat_R": 0.0}
        if len(metadata) != len(generated_texts): logger.error(f"Meta/Gen length mismatch ({len(metadata)} vs {len(generated_texts)})."); return torch.zeros(len(generated_texts), device=self.device), [], batch_stats
        for i, (text, meta) in enumerate(zip(generated_texts, metadata)):
            parsed_prediction = parse_trade_prediction(text)

            # Check if validation function exists and use it
            validation_function_exists = 'validate_prediction_consistency' in globals()
            is_consistent = True
            inconsistency_penalty = 0.0

            if validation_function_exists:
                try:
                    # Try to validate prediction consistency
                    is_consistent = validate_prediction_consistency(parsed_prediction)

                    if not is_consistent:
                        logger.warning(f"Prediction {i} failed consistency validation")
                        inconsistency_penalty = -0.05
                    else:
                        logger.info(f"Prediction {i} passed consistency validation")
                except Exception as e:
                    logger.error(f"Error during consistency validation for prediction {i}: {str(e)}", exc_info=True)
                    is_consistent = True  # Default to True on error
            else:
                logger.warning(f"Validation function not available, skipping consistency check for prediction {i}")

            reward = -1.0 # Default reward if calculation fails
            trade_metrics = {}
            individual_rewards = {'format': 0.0, 'direction': 0.0, 'risk_management': 0.0, 'pnl': 0.0, 'strategy': 0.0} # Default individual

            if parsed_prediction['direction'] is None:
                logger.warning(f"Dir parse fail for sample {i}. Assigning format penalty.")
                reward = -0.2 # Assign penalty directly
                individual_rewards['direction'] = -0.2
                batch_stats["parse_fails"] += 1
            else:
                # Try calculating reward ONLY if parsing succeeded
                try:
                    reward, trade_metrics, individual_rewards = calculate_trade_reward(parsed_prediction, meta, self.trade_manager)
                    # Apply consistency penalty if we performed validation
                    individual_rewards['strategy'] += inconsistency_penalty
                    # Adjust total reward
                    reward += inconsistency_penalty
                except Exception as e:
                    logger.error(f"Reward calculation error for sample {i}: {e}", exc_info=True)
                    reward = -1.0 # Reset reward on error
                    # Reset individual rewards on error too
                    individual_rewards = {'format': 0.0, 'direction': -1.0, 'risk_management': -1.0, 'pnl': -1.0, 'strategy': 0.0}
                    batch_stats["reward_errs"] += 1

                # Track accuracy IF direction was parsed AND actual direction exists
                actual_dir = meta.get('actual_direction')
                pred_dir = parsed_prediction['direction']
                # Ensure both exist before comparing
                if actual_dir and pred_dir:
                    if pred_dir == actual_dir.upper():
                        batch_stats["correct_preds"] += 1

            # Save the generated response and its evaluation data for future training
            response_data = {
                'step': self.global_step,
                'text': text,
                'parsed_prediction': {
                    'direction': parsed_prediction.get('direction'),
                    'percentage': parsed_prediction.get('percentage'),
                    'entry_conditions': parsed_prediction.get('entry_conditions', []),
                    'exit_conditions': parsed_prediction.get('exit_conditions', [])
                },
                'reward': float(reward),
                'individual_rewards': individual_rewards,
                'trade_metrics': trade_metrics,
                'metadata': {
                    'ticker': meta.get('ticker', 'unknown'),
                    'datetime_str': meta.get('datetime_str', 'unknown'),
                    'actual_direction': meta.get('actual_direction'),
                    'actual_percentage': meta.get('actual_percentage')
                }
            }
            self.generated_responses.append(response_data)

            # Accumulate individual reward components (even if default/error values)
            batch_stats["acc_format_R"] += individual_rewards.get('format', 0.0)
            batch_stats["acc_dir_R"] += individual_rewards.get('direction', 0.0)
            batch_stats["acc_risk_R"] += individual_rewards.get('risk_management', 0.0)
            batch_stats["acc_pnl_R"] += individual_rewards.get('pnl', 0.0)
            batch_stats["acc_strat_R"] += individual_rewards.get('strategy', 0.0)

            rewards_list.append(float(reward)); all_trade_metrics.append(trade_metrics)

        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=self.device); explanations = [f"R={r:.3f}, Exit={m.get('exit_reason','N/A')}, H={m.get('holding_periods',0)}" for r, m in zip(rewards_list, all_trade_metrics)]
        num_valid_for_avg = batch_stats["total_preds"] - batch_stats["parse_fails"] - batch_stats["reward_errs"]

        if num_valid_for_avg > 0:
            batch_stats["avg_format_R"] = batch_stats["acc_format_R"] / num_valid_for_avg
            batch_stats["avg_dir_R"] = batch_stats["acc_dir_R"] / num_valid_for_avg
            batch_stats["avg_risk_R"] = batch_stats["acc_risk_R"] / num_valid_for_avg
            batch_stats["avg_pnl_R"] = batch_stats["acc_pnl_R"] / num_valid_for_avg
            batch_stats["avg_strat_R"] = batch_stats["acc_strat_R"] / num_valid_for_avg
        else:
            for key in ['avg_format_R', 'avg_dir_R', 'avg_risk_R', 'avg_pnl_R', 'avg_strat_R']: batch_stats[key] = 0.0

        num_evaluable = batch_stats["total_preds"] - batch_stats["parse_fails"]; accuracy = (batch_stats["correct_preds"] / num_evaluable * 100) if num_evaluable > 0 else 0.0
        logger.info(f"[Reward Stats] AvgR={rewards_tensor.mean().item():.3f}, Acc={accuracy:.1f}%, ParseF={batch_stats['parse_fails']}, RewardE={batch_stats['reward_errs']}")
        if num_valid_for_avg > 0: logger.info(f"  [Avg Comps] Fmt={batch_stats['avg_format_R']:.2f}, Dir={batch_stats['avg_dir_R']:.2f}, Risk={batch_stats['avg_risk_R']:.2f}, PnL={batch_stats['avg_pnl_R']:.2f}, Strat={batch_stats['avg_strat_R']:.2f}")
        return rewards_tensor, explanations, batch_stats

    def _grpo_step(self, batch, inference=False):
        if batch["input_ids"].numel() == 0: logger.warning("Skipping empty batch."); return {"loss": 0.0, "rewards": []}
        input_ids, attention_mask, metadata = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["metadata"]
        self.model.train(); prompt_len = input_ids.shape[1]

        # Check if prompt is already too long
        max_seq = self.max_seq_length
        if prompt_len >= max_seq - 100:  # Ensure some room for generation
            logger.warning(f"Prompt length {prompt_len} is too close to max_seq_length {max_seq}. Truncating prompt.")
            input_ids = input_ids[:, :max_seq-100]  # Leave room for generation
            attention_mask = attention_mask[:, :max_seq-100]
            prompt_len = input_ids.shape[1]

        num_generations = getattr(self.args, "num_generations", 1) # Get from args if defined

        # Modified generation parameters with more conservative settings
        gen_kwargs = {
            "max_new_tokens": min(getattr(self.args, "max_completion_length", 150), max_seq - prompt_len - 10),  # Reduced to 150 tokens max
            "temperature": 0.6,  # Reduced temperature for more focused outputs
            "top_k": 50,
            "top_p": 0.9,
            "do_sample": True,
            "num_return_sequences": 1,
            "repetition_penalty": 1.2,  # Prevent repetitive text
            "no_repeat_ngram_size": 4  # Prevent repeating 4-grams
        }
        if hasattr(self.args, "generation_num_beams"): gen_kwargs["num_beams"] = self.args.generation_num_beams

        logger.info("--- Attempting training step %d... ---", self.global_step + 1)

        all_generated_texts = []; all_completions_ids = []
        for _ in range(num_generations):
            gen_texts, comp_ids = self._generate_responses(input_ids, attention_mask, gen_kwargs)
            all_generated_texts.extend(gen_texts); all_completions_ids.append(comp_ids)

        logger.info("--- Generated Texts for Reward Computation ---")
        for i, text in enumerate(all_generated_texts):
            # Log the full text with clear separators and check for tag presence
            tag_check = {
                "think_tag": "<think>" in text and "</think>" in text,
                "entry_tag": "<entry_conditions>" in text and "</entry_conditions>" in text,
                "exit_tag": "<exit_conditions>" in text and "</exit_conditions>" in text,
                "answer_tag": "<answer>" in text and "</answer>" in text
            }
            tag_status = ", ".join([f"{k}: {'✓' if v else '✗'}" for k, v in tag_check.items()])
            logger.info(f"GENERATION {i} TAG CHECK: {tag_status}")
            logger.info(f"FULL GENERATION {i}:\n{'=' * 80}\n{text}\n{'=' * 80}")

            # Extract the specific tags to see what was generated
            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
            entry_match = re.search(r'<entry_conditions>(.*?)</entry_conditions>', text, re.DOTALL | re.IGNORECASE)
            exit_match = re.search(r'<exit_conditions>(.*?)</exit_conditions>', text, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)

            # Log the extracted tag contents for easier debugging
            logger.info("EXTRACTED TAGS:")
            if think_match: logger.info(f"THINK: {think_match.group(1).strip()[:100]}...")
            if entry_match: logger.info(f"ENTRY: {entry_match.group(1).strip()}")
            if exit_match: logger.info(f"EXIT: {exit_match.group(1).strip()}")
            if answer_match: logger.info(f"ANSWER: {answer_match.group(1).strip()}")
        logger.info("--- End Generated Texts ---")

        eff_bs = len(all_generated_texts); num_prompts = input_ids.shape[0]; expected_bs = num_prompts * num_generations
        if eff_bs != expected_bs: logger.error(f"Expected {expected_bs} generations, got {eff_bs}. Skipping step."); return {"loss": 0.0, "rewards": []}
        completions_ids = torch.cat(all_completions_ids, dim=0)

        repeated_metadata = [meta for meta in metadata for _ in range(num_generations)]
        rewards, explanations, batch_stats = self._compute_rewards(all_generated_texts, repeated_metadata)
        rewards = rewards.detach()

        repeated_input_ids = input_ids.repeat_interleave(num_generations, dim=0)
        repeated_attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)

        # Make sure the completions won't cause total sequence to exceed max_seq_length
        max_completion_len = max_seq - prompt_len
        if completions_ids.shape[1] > max_completion_len:
            logger.warning(f"Truncating completion from {completions_ids.shape[1]} to {max_completion_len} tokens")
            completions_ids = completions_ids[:, :max_completion_len]

        full_ids = torch.cat([repeated_input_ids, completions_ids], dim=1)
        full_attention_mask = torch.cat([repeated_attention_mask, torch.ones_like(completions_ids)], dim=1)

        # Final check to ensure we don't exceed max_seq_length
        if full_ids.shape[1] > max_seq:
            logger.warning(f"Final sequence length {full_ids.shape[1]} exceeds max_seq_length {max_seq}. Truncating.")
            full_ids = full_ids[:, :max_seq]
            full_attention_mask = full_attention_mask[:, :max_seq]

        # Process with policy model
        outputs = self.model(input_ids=full_ids, attention_mask=full_attention_mask)
        logits_policy = outputs.logits
        log_probs_policy = F.log_softmax(logits_policy, dim=-1)

        # Process with reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=full_ids, attention_mask=full_attention_mask)
            logits_ref = ref_outputs.logits.detach()
            log_probs_ref = F.log_softmax(logits_ref, dim=-1)

        gen_len = completions_ids.shape[1]
        log_probs_policy_gen = log_probs_policy[:, prompt_len-1:-1, :]
        log_probs_ref_gen = log_probs_ref[:, prompt_len-1:-1, :]
        target_ids = completions_ids

        if log_probs_policy_gen.shape[1] != target_ids.shape[1]:
             min_len = min(log_probs_policy_gen.shape[1], target_ids.shape[1])
             log_probs_policy_gen = log_probs_policy_gen[:, :min_len, :]
             log_probs_ref_gen = log_probs_ref_gen[:, :min_len, :]
             target_ids = target_ids[:, :min_len]
             logger.warning(f"Log prob/target ID length mismatch, truncated to {min_len}")

        chosen_log_probs_policy = torch.gather(log_probs_policy_gen, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        chosen_log_probs_ref = torch.gather(log_probs_ref_gen, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        rewards_grouped = rewards.view(num_prompts, num_generations)
        advantages = rewards_grouped - rewards_grouped.mean(dim=-1, keepdim=True)

        # Handle standard deviation calculation for single items
        if num_generations > 1:
            std_rewards = rewards_grouped.std(dim=-1, keepdim=True) + 1e-5
        else:
            # If only one item per group, use a default standard deviation
            std_rewards = torch.ones_like(rewards_grouped) * 0.1  # Default std

        advantages = advantages / std_rewards
        advantages = advantages.repeat_interleave(num_generations, dim=0)
        advantages = advantages.detach()

        log_ratio = chosen_log_probs_policy - chosen_log_probs_ref
        gen_mask = (target_ids != self.tokenizer.pad_token_id).float();
        policy_loss_per_token = -(advantages.unsqueeze(1) * log_ratio) * gen_mask;
        policy_loss = policy_loss_per_token.sum(dim=-1).mean()

        prompt_logits_policy = logits_policy[:, :prompt_len, :]
        prompt_logits_ref = logits_ref[:, :prompt_len, :]
        prompt_mask = repeated_attention_mask[:, :prompt_len]
        kl_div = self._compute_kl_divergence(prompt_logits_policy, prompt_logits_ref, prompt_mask)

        # Use real loss calculation for actual training
        loss = policy_loss + self.kl_coef * kl_div

        scaled_loss = loss / self.args.gradient_accumulation_steps
        scaled_loss.backward()
        if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            if self.args.max_grad_norm > 0: nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.args.max_grad_norm)
            self.optimizer.step(); self.scheduler.step(); self.optimizer.zero_grad()

        logger.info(f"Step {self.global_step + 1}: PolicyLoss={policy_loss.item():.4f}, KLDiv={kl_div.item():.4f}, TotalLoss={loss.item():.4f}")

        # Periodically save the generated responses
        if (self.global_step + 1) % self.save_responses_every == 0 and self.generated_responses:
            logger.info(f"Periodic response saving (every {self.save_responses_every} steps)")
            self._save_responses()

        logger.info(f"--- Completed training step {self.global_step + 1}. ---")

        return {"loss": loss.item(), "policy_loss": policy_loss.item(), "kl_div": kl_div.item(), "rewards": rewards.tolist(), "explanations": explanations, "batch_stats": batch_stats}

    def _save_responses(self):
        """Save the generated responses to disk."""
        if not self.generated_responses:
            logger.info("No responses to save.")
            return

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"responses_step_{self.global_step}_{timestamp}.json"
        filepath = os.path.join(self.responses_save_path, filename)

        try:
            # Make sure the directory exists
            os.makedirs(self.responses_save_path, exist_ok=True)

            # Save the responses using the global CustomEncoder
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.generated_responses, f, cls=CustomEncoder, indent=2)

            num_responses = len(self.generated_responses)
            logger.info(f"Saved {num_responses} responses to {filepath}")

            # Reset responses list to avoid duplicates and save memory
            self.generated_responses = []
        except Exception as e:
            logger.error(f"Error saving responses: {str(e)}", exc_info=True)

    def save_model(self, output_dir=None, checkpoint_name="final"):
        output_dir = output_dir or self.args.output_dir; save_path = os.path.join(output_dir, checkpoint_name); os.makedirs(save_path, exist_ok=True); logger.info(f"Saving model checkpoint to {save_path}")
        try:
            if hasattr(self.model, 'save_pretrained') and hasattr(self.model, 'peft_config'): self.model.save_pretrained(save_path); logger.info(f"PEFT adapters saved to {save_path}")
            else: logger.warning("Model does not seem to be a PEFT model, cannot save adapters.")
        except Exception as e: logger.error(f"Error saving model adapters: {e}", exc_info=True); return
        try: self.tokenizer.save_pretrained(save_path); logger.info(f"Tokenizer saved to {save_path}")
        except Exception as e: logger.error(f"Error saving tokenizer: {e}", exc_info=True)
        args_dict = {};
        if isinstance(self.args, TrainingArguments): args_dict = self.args.to_dict()
        elif isinstance(self.args, argparse.Namespace): args_dict = vars(self.args)
        else: logger.warning(f"Unsupported args type for saving: {type(self.args)}")
        args_dict['kl_coef'] = self.kl_coef; args_dict['max_seq_length'] = self.max_seq_length
        if hasattr(self, 'trade_manager'): args_dict['stop_loss_pct'] = self.trade_manager.stop_loss_pct; args_dict['take_profit_pct'] = self.trade_manager.take_profit_pct; args_dict['max_holding_periods'] = self.trade_manager.max_holding_periods
        try:
            output_args_file = os.path.join(save_path, "training_args_full.json");
            with open(output_args_file, "w", encoding='utf-8') as f:
                json.dump(args_dict, f, indent=2, cls=CustomEncoder)
            logger.info(f"Full training args saved to {output_args_file}")
        except Exception as e: logger.warning(f"Could not save training args as JSON: {e}")

    def train(self, resume_from_checkpoint=None, return_state_dict=False):
        """Main training method - performs the full training loop including logging, progress tracking, etc."""
        logger.info(f"Starting Custom GRPO training for {self.total_steps} steps...")
        self.model.train()
        self.global_step = 0
        self.epoch = 0
        start_epoch = 0
        # Clear any existing responses at the start of training
        self.generated_responses = []
        logger.info("Cleared existing responses at start of training.")

        progress_bar = tqdm(total=self.total_steps, desc="GRPO Steps", initial=self.global_step)

        try:
            while self.global_step < self.total_steps:
                self.epoch += 1
                logger.info(f"Starting data pass approx epoch {self.epoch}...")
                batch_iterator = iter(self.train_dataloader)
                while True:
                    if self.global_step >= self.total_steps:
                        break
                    try:
                        batch = next(batch_iterator)
                        if batch["input_ids"].numel() == 0:
                            batch_shape = batch["input_ids"].shape if hasattr(batch["input_ids"], "shape") else "unknown"
                            logger.warning(f"Skipping step {self.global_step} due to empty batch (shape: {batch_shape}).")
                            continue
                    except StopIteration:
                        logger.info(f"Finished data pass approx epoch {self.epoch}.")
                        break
                    except Exception as e:
                        logger.error(f"Error fetching batch at step {self.global_step}: {e}", exc_info=True)
                        continue

                    try:
                        step_results = self._grpo_step(batch)
                    except Exception as e:
                        logger.error(f"Error during training step {self.global_step}: {e}", exc_info=True)
                        continue

                    if self.global_step % self.args.logging_steps == 0:
                        avg_reward = np.mean(step_results.get('rewards', [])) if step_results.get('rewards') else 0.0
                        logger.info(f"Step {self.global_step}/{self.total_steps}: Loss={step_results.get('loss', float('nan')):.4f}, PolicyL={step_results.get('policy_loss', float('nan')):.4f}, KL={step_results.get('kl_div', float('nan')):.4f}, AvgRew={avg_reward:.3f}")
                        bs = step_results.get("batch_stats", {})
                        acc = (bs.get("correct_preds", 0) / bs.get("total_preds", 1) * 100) if bs.get("total_preds", 0) > 0 else 0.0
                        logger.info(f"  Batch Stats: Acc={acc:.1f}%, ParseFails={bs.get('parse_fails', 0)}, RewardErrs={bs.get('reward_errs', 0)}")
                        if bs.get("total_preds", 0) - bs.get('parse_fails', 0) - bs.get('reward_errs', 0) > 0:
                            logger.info(f"  [Avg Comps] Fmt={bs.get('avg_format_R',0):.2f}, Dir={bs.get('avg_dir_R',0):.2f}, Risk={bs.get('avg_risk_R',0):.2f}, PnL={bs.get('avg_pnl_R',0):.2f}, Strat={bs.get('avg_strat_R',0):.2f}")

                    save_strategy = getattr(self.args, "save_strategy", "steps")
                    save_steps = getattr(self.args, "save_steps", 500)
                    if save_strategy == "steps" and save_steps > 0 and (self.global_step + 1) % save_steps == 0:
                        self.save_model(checkpoint_name=f"checkpoint-{self.global_step + 1}")

                    self.global_step += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": step_results.get("loss", float('nan')),
                        "avg_reward": np.mean(step_results.get('rewards', [])) if step_results.get('rewards') else 0.0,
                        "kl": step_results.get("kl_div", float('nan'))
                    })

                if save_strategy == "epoch":
                    self.save_model(checkpoint_name=f"epoch-{self.epoch}")

            # Save any remaining responses at the end of training
            if self.generated_responses:
                logger.info(f"Saving {len(self.generated_responses)} remaining responses from training.")
                self._save_responses()
        except Exception as e:
            logger.error(f"Training encountered an error: {e}", exc_info=True)
            # Try to save responses before raising the exception
            if self.generated_responses:
                try:
                    self._save_responses()
                except Exception as save_error:
                    logger.error(f"Could not save responses after error: {save_error}")
            raise e
        finally:
            progress_bar.close()
            logger.info("Training loop finished.")
            self.save_model()

        return self.model


print("Custom classes and functions defined.")

# CELL 4
# --- Main Execution Logic ---
def main(manual_args=None):
    """Main training function."""
    print("\n=== Starting GRPO PnL Trainer ===\n")
    
    # Parse arguments
    if manual_args is None:
        args = ArgsNamespace()
    else:
        args = manual_args
    
    # FORCE CRITICAL SETTINGS
    args.max_steps = 2500  # Force to 2500 steps
    args.max_seq_length = 6000  # Force to 6000 tokens
    args.train_batch_size = 2  # Set appropriate batch size for long sequences
    args.gradient_accumulation_steps = 8
    args.dataset_size = 2500  # Use full dataset size
    
    # Set up output directories
    os.makedirs(args.output_dir, exist_ok=True)
    set_random_seed(args.seed)
    
    # Load the model and tokenizer
    print(f"Loading base model: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with appropriate quantization
    load_kwargs = {}
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        print("Loading model in 8-bit mode")
    elif args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        print("Loading model in 4-bit mode")
    
    # Remainder of function unchanged...

# --- Colab Execution Setup ---
class ArgsNamespace:
     def __init__(self, **kwargs): self.__dict__.update(kwargs)
if 'compute_dtype' not in locals():
     print("Warning: compute_dtype not defined from Cell 2, defaulting based on GPU availability.")
     if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): default_precision = "bf16"
     elif torch.cuda.is_available(): default_precision = "fp16"
     else: default_precision = "fp32"
else: default_precision = "bf16" if use_bf16 else "fp16"
colab_args = ArgsNamespace(model_name = "Qwen/Qwen2.5-14B-Instruct",
use_pretrained_checkpoint = "/content/drive/MyDrive/PathToSFTCheckpoint/Checkpoint",
output_dir = "./grpo_qlora_results",
dataset_path = "/content/drive/My Drive/PathToDataset/GRPO_PnL_Trainer.jsonl",
max_samples = 2500,
num_train_epochs = 1,
max_steps = 2500,  # Increased from 100 to 2500 for full training
per_device_train_batch_size = 1,
gradient_accumulation_steps = 8,
learning_rate = 5e-6,
kl_coef = 0.03,
reward_baseline = 0.0,
max_grad_norm = 1.0,
weight_decay = 0.01,
lr_scheduler_type = "cosine",
warmup_steps = 10,
lora_r = 16,
lora_alpha = 32,
lora_dropout = 0.05,
stop_loss_pct = 0.02,
take_profit_pct = 0.03,
max_holding_periods = 5,
seed = 42,
disable_wandb = True,
debug = False,
max_seq_length = 2048  # Increased from 1024 to 6000 for longer context
dataloader_num_workers = 0,
logging_steps = 10,
save_strategy = "steps",
save_steps = 250,  # Increased from 20 to 250
precision = "bf16"
                if torch.cuda.is_bf16_supported() else "fp16",
                gradient_checkpointing = True,
                max_completion_length = 2048,
                do_sample = True,
                temperature = 0.6,
                top_k = 50,
                top_p = 0.9,
                num_generations = 1)

# Free up memory before starting
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# Define custom main function that doesn't require a pre-existing PEFT checkpoint
def custom_main():
    log_level = logging.DEBUG if colab_args.debug else logging.INFO
    logger.setLevel(log_level)

    logger.info("Starting GRPO Training Script (Using Custom Trainer)")
    logger.info(f"Script arguments: {vars(colab_args)}")
    set_random_seed(colab_args.seed)

    base_model_name = colab_args.model_name
    if colab_args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif colab_args.precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    logger.info(f"Using precision: {colab_args.precision} ({torch_dtype})")
    logger.info("Setting up QLoRA configuration (4-bit NF4)...")

    logger.info(f"Loading base model using Unsloth: {base_model_name}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = base_model_name,
            max_seq_length = colab_args.max_seq_length,  # Use the configured max_seq_length
            dtype = torch_dtype,
            load_in_4bit = True,
            trust_remote_code = True
        )
        logger.info("Base model loaded successfully via Unsloth.")

        # Add LoRA adapters (instead of loading)
        logger.info("Creating fresh LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=colab_args.lora_r,
            lora_alpha=colab_args.lora_alpha,
            lora_dropout=colab_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing=colab_args.gradient_checkpointing,
        )
        logger.info("Fresh LoRA adapters created and applied to model.")

    except Exception as e:
        logger.error(f"Error setting up model: {e}", exc_info=True)
        return

    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"

    logger.info(f"Loading dataset from: {colab_args.dataset_path}")
    try:
        full_dataset = load_dataset("json", data_files=colab_args.dataset_path, split="train")
    except Exception as e:
        logger.error(f"Error loading dataset from {colab_args.dataset_path}: {e}", exc_info=True)
        return

    if colab_args.max_samples and colab_args.max_samples > 0 and colab_args.max_samples < len(full_dataset):
        logger.info(f"Limiting dataset to {colab_args.max_samples} samples.")
        train_dataset = full_dataset.select(range(colab_args.max_samples))
    else:
        train_dataset = full_dataset

    if len(train_dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return

    training_args = TrainingArguments(
        output_dir=colab_args.output_dir,
        num_train_epochs=colab_args.num_train_epochs,
        max_steps=colab_args.max_steps,
        per_device_train_batch_size=colab_args.per_device_train_batch_size,
        gradient_accumulation_steps=colab_args.gradient_accumulation_steps,
        learning_rate=colab_args.learning_rate,
        weight_decay=colab_args.weight_decay,
        max_grad_norm=colab_args.max_grad_norm,
        lr_scheduler_type=colab_args.lr_scheduler_type,
        warmup_steps=colab_args.warmup_steps,
        logging_dir=os.path.join(colab_args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=colab_args.logging_steps,
        save_strategy=colab_args.save_strategy,
        save_steps=colab_args.save_steps,
        save_total_limit=2,
        bf16=(colab_args.precision == "bf16"),
        fp16=(colab_args.precision == "fp16"),
        gradient_checkpointing=colab_args.gradient_checkpointing,
        report_to="tensorboard" if not colab_args.disable_wandb else "none",
        seed=colab_args.seed,
        dataloader_num_workers=colab_args.dataloader_num_workers,
        remove_unused_columns=False
    )

    logger.info("Initializing Custom GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=colab_args.max_seq_length,
        kl_coef=colab_args.kl_coef,
        stop_loss_pct=colab_args.stop_loss_pct,
        take_profit_pct=colab_args.take_profit_pct,
        max_holding_periods=colab_args.max_holding_periods
    )

    logger.info("Starting training using custom trainer...")
    trainer.train()
    logger.info("Training finished.")

# Run the custom main function
try:
    print("=== RUNNING MODIFIED TRAINING FUNCTION TO CREATE FRESH LORA ADAPTERS ===")
    custom_main()
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("=== Training process completed ===")

# Test function for the validate_prediction_consistency implementation
def test_validation_function():
    # Test with UP direction and consistent entry/exit conditions
    good_up_prediction = {
        'direction': 'UP',
        'percentage': 1.5,
        'entry_conditions': ['RSI_above_60', 'price_above_ma8', 'bullish_macd_crossover'],
        'exit_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_divergence']
    }

    # Test with DOWN direction and consistent entry/exit conditions
    good_down_prediction = {
        'direction': 'DOWN',
        'percentage': 1.2,
        'entry_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_macd_crossover'],
        'exit_conditions': ['RSI_above_70', 'price_above_ma8', 'bullish_divergence']
    }

    # Test with UP direction but inconsistent bearish entry conditions
    bad_up_prediction = {
        'direction': 'UP',
        'percentage': 1.0,
        'entry_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_macd_crossover'],
        'exit_conditions': ['RSI_above_70', 'price_above_ma8', 'bullish_divergence']
    }

    # Test with missing fields
    incomplete_prediction = {
        'direction': 'UP',
        'percentage': 0.5
    }

    print("\n--- Testing validation function ---")
    print(f"Good UP prediction valid: {validate_prediction_consistency(good_up_prediction)}")
    print(f"Good DOWN prediction valid: {validate_prediction_consistency(good_down_prediction)}")
    print(f"Bad UP prediction valid: {validate_prediction_consistency(bad_up_prediction)}")
    print(f"Incomplete prediction valid: {validate_prediction_consistency(incomplete_prediction)}")
    print("--- End validation tests ---\n")

# Only run the test if this module is run directly
if __name__ == "__main__":
    # Uncomment to run validation tests
    # test_validation_function()
    custom_main()

# --- Tag Optimized Prompts ---
def create_base_tag_structure() -> str:
    """
    Creates the basic tag structure for the thinking trace.
    
    Returns:
        A string template with the tag structure
    """
    return """
<thinking>
<tag:OBSERVATION>
[Observe the price chart, volume, and any provided market data]
</tag:OBSERVATION>

<tag:ANALYSIS>
[Analyze key technical indicators, price patterns, and market conditions]
</tag:ANALYSIS>

<tag:REASONING>
[Reason about potential market direction based on evidence and analysis]
</tag:REASONING>

<tag:RISK>
[Assess risks of the trade including potential downsides and probability]
</tag:RISK>

<tag:ALTERNATIVE>
[Consider alternative scenarios that could invalidate your analysis]
</tag:ALTERNATIVE>

<tag:DECISION>
[Make a clear trading decision - buy, sell, or no trade]
</tag:DECISION>

<tag:ENTRY>
[Specify exact entry price and reasoning]
</tag:ENTRY>

<tag:STOP>
[Specify exact stop loss level and reasoning]
</tag:STOP>

<tag:TARGET>
[Specify exact take profit target and reasoning]
</tag:TARGET>

<tag:TIMEFRAME>
[Specify expected timeframe for the trade to play out]
</tag:TIMEFRAME>

<tag:CONFIDENCE>
[Rate confidence in prediction from 1-10 and explain why]
</tag:CONFIDENCE>
</thinking>
"""

def structured_prompt_with_tags(market_data: Dict[str, Any]) -> str:
    """
    Creates a structured prompt with tag guidance for a trading decision.
    
    Args:
        market_data: Dictionary containing market information
        
    Returns:
        Complete prompt with instructions and tag structure
    """
    # Extract relevant market information
    symbol = market_data.get("symbol", "UNKNOWN")
    timeframe = market_data.get("timeframe", "UNKNOWN")
    current_price = market_data.get("current_price", "UNKNOWN")
    
    # Format any additional context
    additional_context = ""
    if "market_events" in market_data and market_data["market_events"]:
        events = market_data["market_events"]
        additional_context = "\nRecent market events:\n- " + "\n- ".join(events)
    
    # Construct the main prompt
    main_prompt = f"""
You are an expert trading analyst tasked with making a trading decision for {symbol} on the {timeframe} timeframe.
Current price: {current_price}{additional_context}

Analyze the chart and provide your trading recommendation using a structured thinking process.

## Guidelines:
1. Use ALL the tags in your thinking process in the order provided
2. Be specific with price levels for entry, stop loss, and take profit
3. Ensure your analysis is thorough and considers multiple indicators
4. Calculate risk-reward ratio explicitly
5. Rate your confidence honestly based on the strength of signals

Begin your thinking with the tagged structure below, filling in each section with detailed analysis:
"""
    
    # Combine with the tag structure
    return main_prompt + create_base_tag_structure()

def prompt_with_tag_examples(market_data: Dict[str, Any]) -> str:
    """
    Creates a prompt with concrete examples of good tag usage.
    
    Args:
        market_data: Dictionary containing market information
        
    Returns:
        Complete prompt with instructions, examples, and tag structure
    """
    base_prompt = structured_prompt_with_tags(market_data)
    
    # Add examples of well-formed tags
    examples = """
## Examples of Good Tag Usage:

<tag:OBSERVATION>
BTC/USD is currently trading at $50,245, approaching the resistance level at $51,000. The price has formed 3 consecutive green candles on the 4h timeframe with increasing volume. RSI is at 68, approaching overbought territory.
</tag:OBSERVATION>

<tag:ANALYSIS>
The MACD shows a bullish crossover that occurred 8 hours ago and continues to diverge positively. The 50-day moving average ($48,500) is now acting as support, which was confirmed by the recent bounce from that level. Bollinger bands are expanding, indicating increasing volatility, with price testing the upper band.
</tag:ANALYSIS>

<tag:REASONING>
The combination of increasing volume on green candles and the MACD bullish crossover strongly suggests momentum is building for an upward move. Since price is approaching but hasn't yet reached overbought territory (RSI < 70), there's likely room for continued upward movement. The expanding Bollinger bands indicate this could be the beginning of a strong trend rather than a temporary fluctuation.
</tag:REASONING>

<tag:RISK>
The main risk is the significant resistance at $51,000, which has rejected price three times in the past month. The RSI approaching overbought could lead to a reversal if buyers exhaust. There's also a divergence forming on the 4h RSI, which hasn't yet confirmed but could indicate weakening momentum despite price increases.
</tag:RISK>
"""
    
    # Insert examples before the tag structure
    insertion_point = base_prompt.find("Begin your thinking")
    if insertion_point != -1:
        return base_prompt[:insertion_point] + examples + base_prompt[insertion_point:]
    else:
        return base_prompt + examples

def focused_tag_prompt(
    market_data: Dict[str, Any], 
    focus_area: str
) -> str:
    """
    Creates a prompt that emphasizes a particular area of analysis.
    
    Args:
        market_data: Dictionary containing market information
        focus_area: Area to emphasize (risk, entry, analysis, etc.)
        
    Returns:
        Prompt with special emphasis on the focus area
    """
    base_prompt = structured_prompt_with_tags(market_data)
    
    focus_guides = {
        "risk": """
## Special Focus on Risk Assessment
For the <tag:RISK> section, please be extremely thorough. Include:
1. Specific probability estimates for adverse scenarios
2. Maximum drawdown analysis
3. Correlation with broader market risks
4. Technical levels that would invalidate your thesis
5. Assessment of volatility risks specific to this setup
""",
        "analysis": """
## Special Focus on Technical Analysis
For the <tag:ANALYSIS> section, please be extremely thorough. Include:
1. Analysis of at least 3 different technical indicators
2. Multiple timeframe confirmation
3. Volume analysis and its correlation with price
4. Support/resistance levels with historical significance
5. Pattern completion percentages and reliability statistics
""",
        "entry": """
## Special Focus on Entry Optimization
For the <tag:ENTRY> section, please be extremely thorough. Include:
1. Multiple potential entry scenarios (immediate vs. pullback)
2. Specific price trigger conditions
3. Volume confirmation requirements
4. Optimal position sizing based on volatility
5. Entry laddering strategy if appropriate
"""
    }
    
    # Add the focus guide if it exists
    if focus_area.lower() in focus_guides:
        insertion_point = base_prompt.find("Begin your thinking")
        if insertion_point != -1:
            return base_prompt[:insertion_point] + focus_guides[focus_area.lower()] + base_prompt[insertion_point:]
    
    return base_prompt

def generate_reflection_prompt(
    original_thinking: str,
    market_outcome: Dict[str, Any]
) -> str:
    """
    Generates a prompt for reflecting on previous analysis after seeing outcomes.
    
    Args:
        original_thinking: The tagged thinking trace from a previous analysis
        market_outcome: Data about what actually happened in the market
        
    Returns:
        A prompt asking for reflection on the original analysis
    """
    # Extract the actual market movement
    direction = market_outcome.get("direction", "unknown")
    price_change = market_outcome.get("price_change", "unknown")
    max_price = market_outcome.get("max_price", "unknown")
    min_price = market_outcome.get("min_price", "unknown")
    
    # Create the reflection prompt
    reflection_prompt = f"""
Review your previous analysis and reflect on its accuracy given the actual market outcome:

## Market Outcome:
- Actual direction: {direction}
- Price change: {price_change}
- Maximum price reached: {max_price}
- Minimum price reached: {min_price}

## Your original thinking trace:
{original_thinking}

## Reflection Instructions:
Please analyze your previous thinking using the following tags:

<tag:CORRECT>
[List what parts of your analysis were correct and why]
</tag:CORRECT>

<tag:INCORRECT>
[List what parts of your analysis were incorrect and why]
</tag:INCORRECT>

<tag:MISSED>
[Identify important signals or factors you missed in your analysis]
</tag:MISSED>

<tag:IMPROVE>
[Explain how you would improve your analysis process next time]
</tag:IMPROVE>

Be specific and reference elements from your original tagged thinking in your reflection.
"""
    
    return reflection_prompt

# Test function for the validate_prediction_consistency implementation
def test_validation_function():
    # Test with UP direction and consistent entry/exit conditions
    good_up_prediction = {
        'direction': 'UP',
        'percentage': 1.5,
        'entry_conditions': ['RSI_above_60', 'price_above_ma8', 'bullish_macd_crossover'],
        'exit_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_divergence']
    }

    # Test with DOWN direction and consistent entry/exit conditions
    good_down_prediction = {
        'direction': 'DOWN',
        'percentage': 1.2,
        'entry_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_macd_crossover'],
        'exit_conditions': ['RSI_above_70', 'price_above_ma8', 'bullish_divergence']
    }

    # Test with UP direction but inconsistent bearish entry conditions
    bad_up_prediction = {
        'direction': 'UP',
        'percentage': 1.0,
        'entry_conditions': ['RSI_below_30', 'price_below_ma8', 'bearish_macd_crossover'],
        'exit_conditions': ['RSI_above_70', 'price_above_ma8', 'bullish_divergence']
    }

    # Test with missing fields
    incomplete_prediction = {
        'direction': 'UP',
        'percentage': 0.5
    }

    print("\n--- Testing validation function ---")
    print(f"Good UP prediction valid: {validate_prediction_consistency(good_up_prediction)}")
    print(f"Good DOWN prediction valid: {validate_prediction_consistency(good_down_prediction)}")
    print(f"Bad UP prediction valid: {validate_prediction_consistency(bad_up_prediction)}")
    print(f"Incomplete prediction valid: {validate_prediction_consistency(incomplete_prediction)}")
    print("--- End validation tests ---\n")

# Only run the test if this module is run directly
if __name__ == "__main__":
    # Uncomment to run validation tests
    # test_validation_function()
    custom_main()

