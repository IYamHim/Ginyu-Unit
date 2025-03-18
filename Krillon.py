import os
import sys
import json
import torch
import random
import logging
import argparse
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
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
    """Format the dataset for SFT training, including 3-day price histories."""
    system_message = """You are an advanced AI stock market analyst specialized in pattern recognition and quantitative analysis.
Your task is to analyze financial data and predict whether a stock will go UP or DOWN, along with an estimated percentage change.

You MUST use EXACTLY this format:

<think>
Key Factors:
1. [Factor 1]: [Analysis and impact]
2. [Factor 2]: [Analysis and impact]
3. [Factor 3]: [Analysis and impact]
4. [Factor 4]: [Analysis and impact]
5. [Factor 5]: [Analysis and impact]
6. [Factor 6]: [Analysis and impact]
7. [Factor 7]: [Analysis and impact]
8. [Factor 8]: [Analysis and impact]
9. [Factor 9]: [Analysis and impact]
10. [Factor 10]: [Analysis and impact]
...

Analysis:
[Detailed analysis of the factors and their interactions]

Overall Market Assessment:
[Summarize the overall market conditions and how they affect this stock]

Self-Review:
- Is this a tricky stock with contradicting signals?
- Are there any unusual patterns I should consider?
- Am I confident in my prediction direction?
- How significant do I expect the price movement to be?

Final Decision:
Based on my analysis, I predict:
- UP scenario: [XX.XX%] increase if [positive factors prevail]
- DOWN scenario: [XX.XX%] decrease if [negative factors prevail]
My final prediction is [UP/DOWN] with an estimated change of [XX.XX%].

Wait, let me make sure that's correct.
[Brief reconsideration of the key evidence and confirmation of your final decision] [If decision is made, thinking is complete,answer with the final prediction]
</think>

<answer>direction: up change: XX.XX%</answer> OR <answer>direction: down change: XX.XX%</answer>

IMPORTANT GUIDELINES:
1. You MUST include the <think> and <answer> tags exactly as shown
2. Your answer MUST include "direction: up/down" and "change: XX.XX%" - no other variations
3. Consider ALL relevant financial factors, not just a few obvious ones
4. Be willing to make bold predictions when the evidence supports it
5. Don't be afraid to predict larger percentage changes (1.50% to 5.00%) when appropriate
6. DOWN predictions are just as valid as UP predictions - don't be biased toward UP or DOWN
7. Make your final decision based on the overall weight of evidence, not just a simple scoring system
8. Always consider BOTH UP and DOWN scenarios before making your final prediction"""

    formatted_data = []
    
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
        
        # Create expected response (for SFT, we're teaching the model to make the correct prediction)
        actual_direction = "up" if price_change_pct > 0 else "down"
        abs_change = abs(price_change_pct)
        abs_change = max(0.1, min(10.0, abs_change))  # Cap it between 0.1% and 10% for realism
        
        # Create example response text for SFT training
        example_analysis = create_example_analysis(item.get('ticker', 'STOCK'), actual_direction, abs_change, price_data, financials, news)
        expected_response = f"{example_analysis}\n\n<answer>direction: {actual_direction} change: {abs_change:.2f}%</answer>"
        
        # Format for SFT training
        formatted_data.append({
            "text": f"{system_message}\n\n{prompt}\n\n{expected_response}"
        })
    
    logger.info(f"Formatted {len(formatted_data)} examples for SFT training")
    return formatted_data

def create_example_analysis(ticker, direction, percentage, price_data, financials, news):
    """Create an example analysis for SFT training."""
    is_up = direction.lower() == "up"
    
    # Create key factors based on the direction
    key_factors = []
    
    # Create positive factors
    positive_factors = [
        f"Recent price momentum shows {ticker} has been trending upward for 3 consecutive days",
        f"RSI indicator at 62.5 shows growing buying pressure but not yet overbought",
        f"Revenue growth of 12.3% exceeds industry average of 8.1%",
        f"Positive analyst coverage with recent price target increases",
        f"Strong sector performance with peer companies also trending upward",
        f"MACD indicator shows a bullish crossover signal",
        f"Price has broken above key resistance level at ${price_data.get('close_previous', '50.00')}",
        f"Institutional buying activity has increased by 15% over the past month",
        f"Recent product launch expected to drive new revenue streams",
        f"Technical indicators suggest bullish continuation pattern forming"
    ]
    
    # Create negative factors
    negative_factors = [
        f"Recent price action shows {ticker} has declined over the past 3 trading sessions",
        f"RSI indicator at 32.5 shows weakening momentum and selling pressure",
        f"Revenue decline of 8.3% compared to industry average growth of 3.2%",
        f"Recent negative analyst reports with downgraded price targets",
        f"Sector weakness with multiple peer companies also showing declines",
        f"MACD indicator shows a bearish crossover signal",
        f"Price has broken below key support level at ${price_data.get('close_previous', '50.00')}",
        f"Unusual volume patterns suggesting institutional selling",
        f"Recent negative news about product delays",
        f"P/E ratio significantly above sector average indicating potential overvaluation"
    ]
    
    # Select factors based on the prediction direction
    if is_up:
        main_factors = positive_factors
        counter_factors = negative_factors
    else:
        main_factors = negative_factors
        counter_factors = positive_factors
    
    # Select 7 main factors and 3 counter factors
    random.shuffle(main_factors)
    random.shuffle(counter_factors)
    selected_factors = main_factors[:7] + counter_factors[:3]
    random.shuffle(selected_factors)
    
    # Create key factors section
    factors_text = "Key Factors:\n"
    for i, factor in enumerate(selected_factors, 1):
        factors_text += f"{i}. {factor}\n"
    
    # Create analysis section
    if is_up:
        analysis = f"""Analysis:
The technical indicators for {ticker} show positive momentum, with the RSI at 62.5 indicating strong buying interest without being overbought. The price has been trending upward for three consecutive days, showing growing investor confidence.

Revenue growth of 12.3% exceeds the industry average of 8.1%, demonstrating the company is outperforming its peers. This fundamental strength is likely to attract new investors and further drive price appreciation.

While there are some minor concerns regarding sector volatility and potential resistance levels, the overall weight of evidence suggests continued upward momentum is likely."""
    else:
        analysis = f"""Analysis:
The technical indicators for {ticker} show negative momentum, with the RSI at 32.5 indicating selling pressure. The price has declined over the past three trading sessions, showing deteriorating investor sentiment.

Revenue decline of 8.3% compared to the industry average growth of 3.2% raises significant concerns about the company's competitive position. This fundamental weakness is likely to continue putting downward pressure on the stock price.

While there are some potential support levels and possible positive catalysts from new products, the overall weight of evidence suggests continued downward pressure is likely."""
    
    # Create market assessment
    if is_up:
        market_assessment = f"""Overall Market Assessment:
The broader market environment remains conducive to growth stocks in this sector. Market sentiment is positive, with major indices trending upward, providing additional tailwind for {ticker}. Industry-specific metrics also suggest favorable conditions for continued price appreciation."""
    else:
        market_assessment = f"""Overall Market Assessment:
The broader market is showing signs of weakness in this sector, with the sector index down 1.2% this week. This creates additional headwinds for {ticker}, which is already facing company-specific challenges. The current market environment is likely to amplify the stock's downward pressure."""
    
    # Create self-review section
    self_review = f"""Self-Review:
- This stock shows {'mostly positive signals with some conflicting indicators' if is_up else 'clear negative signals across multiple indicators'}
- {'The upward trend appears sustainable based on fundamental and technical factors' if is_up else 'The downward trend is supported by both technical indicators and fundamental data'}
- I'm {'reasonably' if percentage < 2.0 else 'highly'} confident in my {direction.upper()} prediction
- I expect a {'modest' if percentage < 2.0 else 'significant'} move of around {percentage:.2f}%"""
    
    # Create final decision section
    if is_up:
        final_decision = f"""Final Decision:
Based on my analysis, I predict:
- UP scenario: {percentage:.2f}% increase if positive momentum continues and revenue growth translates to higher valuation
- DOWN scenario: 0.50% decrease if broader market experiences a pullback or profit-taking occurs
My final prediction is UP with an estimated change of {percentage:.2f}%.

Wait, let me make sure that's correct.
Reviewing the evidence, the strong revenue growth, positive technical indicators, and supportive market environment all point to upward price movement. The fundamental story remains compelling despite some minor concerns. My UP prediction of {percentage:.2f}% increase is well-supported by the data."""
    else:
        final_decision = f"""Final Decision:
Based on my analysis, I predict:
- UP scenario: 0.50% increase if the company announces positive news or market sentiment improves unexpectedly
- DOWN scenario: {percentage:.2f}% decrease if negative momentum continues and revenue decline impacts investor confidence
My final prediction is DOWN with an estimated change of {percentage:.2f}%.

Wait, let me make sure that's correct.
Looking back at the evidence, the technical indicators, financial metrics, and market conditions all suggest bearish momentum. While there is a slight chance of an upward movement due to oversold conditions, the weight of evidence strongly favors the downward scenario. My DOWN prediction of {percentage:.2f}% decline is well-supported by the data."""
    
    # Combine all sections
    return f"<think>\n{factors_text}\n{analysis}\n\n{market_assessment}\n\n{self_review}\n\n{final_decision}\n</think>"

def create_detailed_financial_prompt(ticker, company_info, price_data, financials, news):
    """Create a detailed financial prompt for analysis."""
    # Extract basic information
    company_name = company_info.get('name', ticker) if isinstance(company_info, dict) else ticker
    sector = company_info.get('sector', 'Technology') if isinstance(company_info, dict) else 'Technology'
    industry = company_info.get('industry', 'Software') if isinstance(company_info, dict) else 'Software'
    
    # Extract price data
    current_price = price_data.get('open', random.uniform(50, 200)) if isinstance(price_data, dict) else random.uniform(50, 200)
    previous_close = price_data.get('close_previous', current_price * random.uniform(0.95, 1.05)) if isinstance(price_data, dict) else current_price * random.uniform(0.95, 1.05)
    
    # Format prices to two decimal places
    current_price = round(float(current_price), 2)
    previous_close = round(float(previous_close), 2)
    
    # Generate random but realistic technical indicators
    rsi = round(random.uniform(30, 70), 2)
    macd = round(random.uniform(-2, 2), 2)
    moving_avg_50 = round(float(current_price) * random.uniform(0.9, 1.1), 2)
    moving_avg_200 = round(float(current_price) * random.uniform(0.8, 1.2), 2)
    
    # Generate random financial metrics if not available
    revenue = financials.get('revenue', f"{round(random.uniform(100, 5000), 2)}M") if isinstance(financials, dict) else f"{round(random.uniform(100, 5000), 2)}M"
    net_income = financials.get('net_income', f"{round(random.uniform(-200, 1000), 2)}M") if isinstance(financials, dict) else f"{round(random.uniform(-200, 1000), 2)}M"
    eps = financials.get('eps', round(random.uniform(-2, 10), 2)) if isinstance(financials, dict) else round(random.uniform(-2, 10), 2)
    pe_ratio = financials.get('pe_ratio', round(random.uniform(10, 40), 2)) if isinstance(financials, dict) else round(random.uniform(10, 40), 2)
    
    # Generate random news headlines if not available
    headlines = []
    if isinstance(news, dict) and 'headlines' in news and isinstance(news['headlines'], list):
        headlines = [item.get('headline', '') for item in news['headlines'][:5] if isinstance(item, dict)]
    
    # If no headlines or insufficient, create synthetic ones
    if len(headlines) < 3:
        synthetic_headlines = [
            f"{ticker} Reports {random.choice(['Strong', 'Weak', 'Mixed'])} Quarterly Results",
            f"Analysts {random.choice(['Upgrade', 'Downgrade', 'Maintain'])} Rating on {ticker}",
            f"{ticker} Announces New {random.choice(['Product Launch', 'Partnership', 'Executive Hire'])}",
            f"{company_name} {random.choice(['Expands', 'Reduces'])} Operations in {random.choice(['North America', 'Europe', 'Asia'])}",
            f"{industry} Sector Facing {random.choice(['Headwinds', 'Tailwinds'])} Due to Economic Conditions"
        ]
        random.shuffle(synthetic_headlines)
        headlines.extend(synthetic_headlines[:5 - len(headlines)])
    
    # Create news section
    news_section = "Recent News:\n"
    for headline in headlines[:5]:
        news_section += f"- {headline}\n"
    
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

Based on this information, predict whether {ticker} stock will go UP or DOWN, and estimate the percentage change. Include your analysis reasoning.
"""
    return prompt

def generate_synthetic_dataset(num_samples=1000, down_ratio=0.6):
    """Generate a synthetic dataset with the specified down/up distribution ratio."""
    logger.info(f"Generating {num_samples} synthetic samples with {down_ratio*100}% DOWN predictions")
    
    synthetic_data = []
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "TSLA", "DIS", "NVDA", "AMD", 
              "INTC", "CSCO", "ADBE", "CRM", "PYPL", "UBER", "LYFT", "SHOP", "SQ", "SNAP"]
    
    sectors = ["Technology", "Healthcare", "Financial Services", "Consumer Cyclical", 
              "Communication Services", "Industrials", "Energy", "Utilities", "Real Estate", "Materials"]
    
    industries = {
        "Technology": ["Software", "Hardware", "Semiconductors", "IT Services"],
        "Healthcare": ["Biotechnology", "Pharmaceuticals", "Medical Devices", "Healthcare Services"],
        "Financial Services": ["Banks", "Insurance", "Asset Management", "Fintech"],
        "Consumer Cyclical": ["Retail", "Automotive", "Hospitality", "Apparel"],
        "Communication Services": ["Telecom", "Media", "Social Media", "Entertainment"],
        "Industrials": ["Aerospace", "Defense", "Machinery", "Transportation"],
        "Energy": ["Oil & Gas", "Renewable Energy", "Energy Services"],
        "Utilities": ["Electric Utilities", "Water Utilities", "Gas Utilities"],
        "Real Estate": ["REITs", "Real Estate Services", "Real Estate Development"],
        "Materials": ["Chemicals", "Metals & Mining", "Construction Materials"]
    }
    
    # Determine how many UP and DOWN samples to generate
    num_down = int(num_samples * down_ratio)
    num_up = num_samples - num_down
    
    # Generate DOWN samples
    for i in range(num_down):
        ticker = random.choice(tickers)
        sector = random.choice(sectors)
        industry = random.choice(industries[sector])
        
        # Generate a price decrease
        current_price = round(random.uniform(10, 500), 2)
        percentage_change = round(random.uniform(-0.5, -5.0), 2)  # Negative percentage
        previous_price = round(current_price / (1 + percentage_change/100), 2)
        
        synthetic_data.append({
            "ticker": ticker,
            "company_info": {
                "name": f"{ticker} Inc.",
                "sector": sector,
                "industry": industry,
                "price": {
                    "open": current_price,
                    "close_previous": previous_price
                },
                "financials": {
                    "financials": json.dumps({
                        "revenue": f"{round(random.uniform(100, 5000), 2)}M",
                        "net_income": f"{round(random.uniform(-200, 1000), 2)}M",
                        "eps": round(random.uniform(-2, 10), 2),
                        "pe_ratio": round(random.uniform(10, 40), 2)
                    })
                },
                "news": {
                    "headlines": [
                        {"headline": f"{ticker} Reports {random.choice(['Weak', 'Mixed'])} Quarterly Results"},
                        {"headline": f"Analysts {random.choice(['Downgrade', 'Maintain'])} Rating on {ticker}"},
                        {"headline": f"{ticker} Faces {random.choice(['Headwinds', 'Challenges'])} in Current Market"}
                    ]
                }
            }
        })
    
    # Generate UP samples
    for i in range(num_up):
        ticker = random.choice(tickers)
        sector = random.choice(sectors)
        industry = random.choice(industries[sector])
        
        # Generate a price increase
        current_price = round(random.uniform(10, 500), 2)
        percentage_change = round(random.uniform(0.5, 5.0), 2)  # Positive percentage
        previous_price = round(current_price / (1 + percentage_change/100), 2)
        
        synthetic_data.append({
            "ticker": ticker,
            "company_info": {
                "name": f"{ticker} Inc.",
                "sector": sector,
                "industry": industry,
                "price": {
                    "open": current_price,
                    "close_previous": previous_price
                },
                "financials": {
                    "financials": json.dumps({
                        "revenue": f"{round(random.uniform(100, 5000), 2)}M",
                        "net_income": f"{round(random.uniform(0, 1000), 2)}M",
                        "eps": round(random.uniform(0, 10), 2),
                        "pe_ratio": round(random.uniform(10, 40), 2)
                    })
                },
                "news": {
                    "headlines": [
                        {"headline": f"{ticker} Reports {random.choice(['Strong', 'Better-than-expected'])} Quarterly Results"},
                        {"headline": f"Analysts {random.choice(['Upgrade', 'Maintain'])} Rating on {ticker}"},
                        {"headline": f"{ticker} Announces New {random.choice(['Product Launch', 'Partnership', 'Initiative'])}"}
                    ]
                }
            }
        })
    
    # Shuffle the data
    random.shuffle(synthetic_data)
    return synthetic_data

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="SFT training for stock prediction")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name or path")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path (if None, will use synthetic data)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_train_epochs", type=float, default=3, help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use")
    parser.add_argument("--down_ratio", type=float, default=0.6, help="Ratio of DOWN predictions in synthetic data (0-1)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of update steps to accumulate")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization for training")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load dataset (or generate synthetic data)
    if args.dataset_path:
        try:
            if args.dataset_path.startswith("2084Collective/"):
                dataset = load_dataset(args.dataset_path, split="train")
            else:
                dataset = load_dataset("json", data_files=args.dataset_path, split="train")
            
            # Limit to max_samples
            if args.max_samples and len(dataset) > args.max_samples:
                dataset = dataset.select(range(args.max_samples))
            
            logger.info(f"Loaded {len(dataset)} samples from {args.dataset_path}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Falling back to synthetic data generation")
            synthetic_data = generate_synthetic_dataset(args.max_samples, args.down_ratio)
            dataset = Dataset.from_list(synthetic_data)
    else:
        # Generate synthetic data
        synthetic_data = generate_synthetic_dataset(args.max_samples, args.down_ratio)
        dataset = Dataset.from_list(synthetic_data)
        logger.info(f"Generated {len(dataset)} synthetic samples with {args.down_ratio*100}% DOWN predictions")

    # Format dataset for SFT
    formatted_data = format_dataset_for_sft(dataset)
    sft_dataset = Dataset.from_list(formatted_data)

    # Load base model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    model_kwargs = {"device_map": "auto"}
    
    if args.use_8bit:
        model_kwargs["load_in_8bit"] = True
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
    
    # Configure LoRA for efficient fine-tuning
    if args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    logger.info(f"Model prepared with LoRA configuration: {lora_config}")
    
    # Create helper function to tokenize inputs
    def tokenize_and_prepare(samples):
        """Convert text to input_ids and attention_mask for training"""
        # Convert the examples to a list of strings
        texts = samples["text"]
        # Tokenize the examples
        tokenized = tokenizer(
            texts, 
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_special_tokens_mask=True
        )
        return tokenized
    
    # Process dataset
    logger.info("Processing dataset for training...")
    processed_dataset = sft_dataset.map(
        tokenize_and_prepare,
        batched=True,
        remove_columns=["text"]  # Remove original text column
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not masked language modeling
    )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_steps=200,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,  # Important: keep all columns for the data collator
        fp16=not args.use_8bit,  # Don't use fp16 with 8-bit quantization
        logging_steps=10,
        report_to="none",  # Disable wandb
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=processed_dataset,
    )
    
    # Start training
    logger.info("Starting SFT training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model successfully saved to {args.output_dir}")
    
if __name__ == "__main__":
    main() 
