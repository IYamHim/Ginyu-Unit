import os
import sys
import json
import torch
import random
import logging
import argparse
import numpy as np
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from typing import List, Dict, Any, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sft_training.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def format_dataset_for_sft(dataset):
    """Format the dataset for SFT training, including validation of critical elements."""
    formatted_data = []
    logger.info(f"Starting to format {len(dataset)} examples for SFT training")
    
    for item in dataset:
        try:
            # Get the text which contains thinking and answer
            text = item.get('text', '')
            
            if not text:
                # Try to get text from user_prompt
                text = item.get('user_prompt', '')
            
            if not text:
                logger.warning("Skipping example: No text found")
                continue
                
            # Clean up the thinking section
            if '<think>' in text and '</think>' in text:
                # Extract the thinking section
                think_pattern = r'<think>(.*?)</think>'
                think_match = re.search(think_pattern, text, re.DOTALL)
                
                if think_match:
                    thinking = think_match.group(1)
                    thinking = '<think>\n' + thinking.strip() + '\n</think>'
                else:
                    logger.warning("Skipping example: Could not extract thinking section")
                    continue
                    
                # Extract the answer section
                answer_pattern = r'<answer>(.*?)</answer>'
                answer_match = re.search(answer_pattern, text, re.DOTALL)
                
                if answer_match:
                    answer = answer_match.group(1).strip()
                else:
                    logger.warning("Skipping example: Could not extract answer section")
                    continue
                
                # Extract direction and percentage from the answer
                direction_match = re.search(r'direction:\s*(up|down)', answer, re.IGNORECASE)
                percentage_match = re.search(r'change:\s*(\d+\.\d+)%', answer)
                
                if direction_match and percentage_match:
                    direction = direction_match.group(1).lower()
                    percentage = float(percentage_match.group(1))
                    
                    # Format the answer tag correctly
                    answer_formatted = f"<answer>direction: {direction} change: {percentage:.2f}%</answer>"
                    
                    # Combine into final format
                    formatted_text = f"{thinking}\n\n{answer_formatted}"
                    
                    # Add to formatted dataset
                    formatted_data.append({
                        "text": formatted_text
                    })
                else:
                    logger.warning(f"Skipping example: Could not extract prediction direction and percentage: {answer}")
                    continue
            else:
                # If there's no thinking tags but there's a direction and percentage in the text
                direction_match = re.search(r'My final prediction is (UP|DOWN) with an estimated change of (\d+\.\d+)%'
                                            , text, re.IGNORECASE)
                if direction_match:
                    direction = direction_match.group(1).lower()
                    percentage = float(direction_match.group(2))
                    
                    # Add artificial thinking tags if not present
                    if '<think>' not in text:
                        text = '<think>\n' + text
                    if '</think>' not in text:
                        text = text + '\n</think>'
                    
                    # Add formatted answer
                    answer_formatted = f"<answer>direction: {direction} change: {percentage:.2f}%</answer>"
                    formatted_text = f"{text}\n\n{answer_formatted}"
                    
                    formatted_data.append({
                        "text": formatted_text
                    })
                else:
                    logger.warning("Skipping example: Could not extract prediction direction and percentage")
                    continue
            
            # Log progress for large datasets
            if len(formatted_data) % 100 == 0:
                logger.info(f"Processed {len(formatted_data)} examples out of {len(dataset)}")
        
        except Exception as e:
            logger.warning(f"Error processing example: {str(e)}")
            continue
    
    logger.info(f"Formatted {len(formatted_data)} examples for SFT training")
    return formatted_data

def validate_prediction_format(analysis_text):
    """Validate that the analysis contains required UP/DOWN prediction format."""
    # Check for think tag
    if not (analysis_text.startswith('<think>') and '</think>' in analysis_text):
        return False
    
    # Check for Key Factors section
    if 'Key Factors:' not in analysis_text:
        return False
    
    # Check for Analysis section
    if 'Analysis:' not in analysis_text:
        return False
    
    # Check for Final Decision section and prediction format
    prediction_pattern = r'My final prediction is (UP|DOWN) with an estimated change of (\d+\.\d+)%'
    if not re.search(prediction_pattern, analysis_text, re.IGNORECASE):
        return False
    
    return True

def create_example_analysis(ticker, direction, percentage, price_data, financials, news):
    """Create an example analysis for SFT training with diverse, realistic thinking steps."""
    is_up = direction.lower() == "up"
    
    # Extract real values from data where available
    current_price = price_data.get('open', random.uniform(50, 200)) if isinstance(price_data, dict) else random.uniform(50, 200)
    previous_close = price_data.get('close_previous', current_price * random.uniform(0.95, 1.05)) if isinstance(price_data, dict) else current_price * random.uniform(0.95, 1.05)
    
    # Format as floats for calculations
    current_price = float(current_price)
    previous_close = float(previous_close)
    
    # Generate realistic technical indicators based on actual price movement
    rsi_value = random.uniform(65, 80) if is_up else random.uniform(20, 35)
    macd_value = random.uniform(0.1, 0.8) if is_up else random.uniform(-0.8, -0.1)
    volume_change = random.uniform(5, 30) if is_up else random.uniform(-30, -5)
    
    # Extract financial data or generate realistic values
    if isinstance(financials, dict):
        revenue_growth = financials.get('revenue_growth', random.uniform(5, 15) if is_up else random.uniform(-10, 3))
        profit_margin = financials.get('profit_margin', random.uniform(10, 25) if is_up else random.uniform(5, 15))
        eps_value = financials.get('eps', random.uniform(1.5, 4.0) if is_up else random.uniform(0.5, 1.5))
        pe_ratio = financials.get('pe_ratio', random.uniform(15, 25) if is_up else random.uniform(10, 20))
    else:
        revenue_growth = random.uniform(5, 15) if is_up else random.uniform(-10, 3)
        profit_margin = random.uniform(10, 25) if is_up else random.uniform(5, 15)
        eps_value = random.uniform(1.5, 4.0) if is_up else random.uniform(0.5, 1.5)
        pe_ratio = random.uniform(15, 25) if is_up else random.uniform(10, 20)

    # Check for headlines
    has_positive_news = False
    has_negative_news = False
    if isinstance(news, dict) and 'headlines' in news and isinstance(news['headlines'], list):
        for headline in news['headlines']:
            if isinstance(headline, dict) and 'headline' in headline:
                headline_text = headline['headline']
                positive_keywords = ['beat', 'growth', 'increase', 'positive', 'upgrade', 'strong', 'exceed', 'higher']
                negative_keywords = ['miss', 'decline', 'decrease', 'negative', 'downgrade', 'weak', 'lower', 'below']
                
                if any(word in headline_text.lower() for word in positive_keywords):
                    has_positive_news = True
                if any(word in headline_text.lower() for word in negative_keywords):
                    has_negative_news = True
    
    # Create positive factors with variations based on actual data
    positive_factors = [
        f"Recent price momentum shows {ticker} has been trending upward with a {random.uniform(1.2, 4.5):.1f}% gain in the last trading sessions",
        f"RSI indicator at {rsi_value:.1f} shows growing buying pressure {'without being overbought' if rsi_value < 70 else 'though approaching overbought territory'}",
        f"Revenue growth of {revenue_growth:.1f}% {'exceeds' if revenue_growth > 5 else 'aligns with'} industry average expectations",
        f"Positive analyst coverage with recent price target increases to ${current_price * random.uniform(1.1, 1.3):.2f}",
        f"Strong sector performance with peer companies also trending upward by {random.uniform(1.0, 3.5):.1f}%",
        f"MACD indicator value of {macd_value:.2f} shows a bullish crossover signal",
        f"Price has broken above key resistance level at ${previous_close:.2f}",
        f"Trading volume increased by {volume_change:.1f}% compared to 30-day average, indicating strong buying interest",
        f"EPS of ${eps_value:.2f} {'beat' if is_up else 'met'} analyst expectations",
        f"P/E ratio of {pe_ratio:.1f} is {'favorable' if pe_ratio < 22 else 'elevated but justified'} compared to sector average",
        f"Support from the 50-day moving average at ${current_price * random.uniform(0.92, 0.98):.2f}",
        f"Short interest has decreased by {random.uniform(5, 15):.1f}%, reducing downward pressure",
        f"Profit margin of {profit_margin:.1f}% shows operational efficiency"
    ]
    
    # Add news-based factors if available
    if has_positive_news:
        positive_factors.append(f"Recent headlines contain positive news about {ticker}'s performance and outlook")
    
    # Create negative factors with variations based on actual data
    negative_factors = [
        f"Recent price action shows {ticker} has declined by {random.uniform(1.2, 4.5):.1f}% over the past trading sessions",
        f"RSI indicator at {rsi_value:.1f} shows weakening momentum {'and approaching oversold territory' if rsi_value < 30 else ''}",
        f"Revenue decline of {abs(revenue_growth):.1f}% compared to industry average growth of {random.uniform(2, 5):.1f}%",
        f"Recent negative analyst reports with downgraded price targets to ${current_price * random.uniform(0.7, 0.9):.2f}",
        f"Sector weakness with multiple peer companies also showing declines of {random.uniform(1.5, 4.0):.1f}%",
        f"MACD indicator value of {macd_value:.2f} shows a bearish crossover signal",
        f"Price has broken below key support level at ${previous_close:.2f}",
        f"Trading volume decreased by {abs(volume_change):.1f}% compared to 30-day average, indicating lack of buying interest",
        f"EPS of ${eps_value:.2f} {'missed' if not is_up else 'barely met'} analyst expectations",
        f"P/E ratio of {pe_ratio:.1f} is {'concerning' if pe_ratio > 20 else 'reasonable but under pressure'} compared to sector average",
        f"Resistance at the 50-day moving average at ${current_price * random.uniform(1.02, 1.08):.2f}",
        f"Short interest has increased by {random.uniform(5, 20):.1f}%, adding downward pressure",
        f"Profit margin contraction to {profit_margin:.1f}% raises concerns about cost management"
    ]
    
    # Add news-based factors if available
    if has_negative_news:
        negative_factors.append(f"Recent headlines contain negative news about {ticker}'s challenges and outlook")
    
    # Select factors based on the prediction direction
    if is_up:
        main_factors = positive_factors
        counter_factors = negative_factors
    else:
        main_factors = negative_factors
        counter_factors = positive_factors
    
    # Select 7-8 main factors and 2-3 counter factors
    random.shuffle(main_factors)
    random.shuffle(counter_factors)
    num_main = random.randint(7, 8)
    num_counter = random.randint(2, 3)
    selected_factors = main_factors[:num_main] + counter_factors[:num_counter]
    random.shuffle(selected_factors)
    
    # Format the analysis text
    analysis_text = f"""<think>
Key Factors:
{chr(10).join(f'- {factor}' for factor in selected_factors)}

Analysis:
Based on the above factors, {'several positive indicators suggest upward momentum' if is_up else 'multiple concerning signals indicate downward pressure'}. 
{'The combination of strong technical indicators and positive fundamentals supports a bullish outlook.' if is_up else 'The mix of weak technical signals and concerning fundamentals points to bearish sentiment.'}
{'While some counterpoints exist, the overall evidence suggests continued upward movement.' if is_up else 'Despite some positive factors, the weight of evidence suggests further decline.'}

My final prediction is {direction.upper()} with an estimated change of {percentage:.2f}%
</think>"""

    return analysis_text

def train_model(
    base_model: str,
    dataset_path: str,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_steps: Optional[int] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 2048,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 100,
    warmup_steps: int = 100,
    local_rank: int = -1,
    resume_from_checkpoint: Optional[str] = None,
    use_8bit: bool = False
):
    """Train the model using SFT with LoRA."""
    logger.info(f"Starting training with base model: {base_model}")
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_dataset = json.load(f)
        dataset = Dataset.from_list(raw_dataset)
    else:
        dataset = load_dataset(dataset_path)['train']
    
    # Format dataset for SFT
    formatted_data = format_dataset_for_sft(dataset)
    train_dataset = Dataset.from_list(formatted_data)
    
    logger.info(f"Dataset size: {len(train_dataset)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=use_8bit
    )
    
    # Prepare model for k-bit training if needed
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info(f"Configuring LoRA with r={lora_r}, alpha={lora_alpha}")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        local_rank=local_rank,
        save_total_limit=3,
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text"
    )
    
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train a stock prediction model using SFT.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model to use")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    
    args = parser.parse_args()
    
    train_model(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_8bit=args.use_8bit
    )

if __name__ == "__main__":
    main() 