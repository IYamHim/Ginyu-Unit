#!/usr/bin/env python3
# Namek Project - Inference Script
# Owner: ./install_AI

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Over9Thousand import extract_up_down_with_percentage

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on a stonk using the trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--ticker', type=str, required=True, help='Stonk ticker')
    parser.add_argument('--company_name', type=str, required=True, help='Company name')
    parser.add_argument('--current_price', type=float, required=True, help='Current stonk price')
    parser.add_argument('--previous_price', type=float, required=True, help='Previous stonk price')
    parser.add_argument('--revenue', type=str, help='Company revenue')
    parser.add_argument('--net_income', type=str, help='Company net income')
    parser.add_argument('--eps', type=str, help='Earnings per share')
    parser.add_argument('--pe_ratio', type=str, help='Price to earnings ratio')
    parser.add_argument('--rsi', type=float, help='RSI value')
    parser.add_argument('--macd', type=float, help='MACD value')
    parser.add_argument('--moving_avg_50', type=float, help='50-day moving average')
    parser.add_argument('--moving_avg_200', type=float, help='200-day moving average')
    parser.add_argument('--news', type=str, nargs='*', help='Recent news headlines')
    parser.add_argument('--sector', type=str, help='Company sector')
    parser.add_argument('--industry', type=str, help='Company industry')
    return parser.parse_args()

def create_prompt(args):
    """Create a detailed prompt for the model."""
    # Calculate price change
    price_change = ((args.current_price - args.previous_price) / args.previous_price) * 100
    
    prompt = f"""Analysis Request for {args.ticker} ({args.company_name})

Company Information:
Sector: {args.sector if args.sector else 'Unknown'}
Industry: {args.industry if args.industry else 'Unknown'}

Price Data:
Current Price: ${args.current_price:.2f}
Previous Close: ${args.previous_price:.2f}
Price Change: {price_change:.2f}%

Technical Indicators:
RSI (14-day): {args.rsi if args.rsi else 'N/A'}
MACD: {args.macd if args.macd else 'N/A'}
50-day Moving Average: ${args.moving_avg_50 if args.moving_avg_50 else 'N/A'}
200-day Moving Average: ${args.moving_avg_200 if args.moving_avg_200 else 'N/A'}

Financial Metrics:
Revenue: ${args.revenue if args.revenue else 'N/A'}
Net Income: ${args.net_income if args.net_income else 'N/A'}
EPS: ${args.eps if args.eps else 'N/A'}
P/E Ratio: {args.pe_ratio if args.pe_ratio else 'N/A'}

Recent News:
"""
    
    if args.news:
        for headline in args.news:
            prompt += f"- {headline}\n"
    else:
        prompt += "- No recent news available\n"
    
    prompt += """
Based on this information, predict whether the stonk price will go UP or DOWN, and estimate the percentage change. Include your analysis reasoning.

<think>Let me analyze this data carefully:
1. Price trend and momentum
2. Technical indicators
3. Financial metrics
4. News sentiment
</think>

<answer>"""
    
    return prompt

def run_inference(args):
    """Run inference on a stonk using the trained model."""
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # Create the prompt
    prompt = create_prompt(args)
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the model's prediction from the output
    prediction_text = generated_text[len(prompt):]
    
    # Print the complete raw generated text
    print("\n=== COMPLETE RAW GENERATED TEXT ===")
    print(prediction_text)
    print("=== END OF RAW GENERATED TEXT ===\n")
    
    # Extract UP/DOWN prediction and percentage
    direction_pattern = re.compile(r'direction:\s*(up|down).*?change:\s*(\d+\.?\d*)%', re.DOTALL | re.IGNORECASE)
    match = direction_pattern.search(prediction_text)
    
    if match:
        direction = match.group(1).upper()
        percentage = float(match.group(2))
        print(f"\nFinal Prediction: {direction}")
        print(f"Predicted Price Change: {percentage:.2f}%")
    else:
        # Try alternative extraction method
        prediction = extract_up_down_with_percentage(prediction_text)
        
        if prediction['direction']:
            print(f"\nFinal Prediction: {prediction['direction'].upper()}")
            if prediction['percentage']:
                print(f"Predicted Price Change: {prediction['percentage']:.2f}%")
        else:
            print("\nCould not extract a clear UP/DOWN prediction.")

def main():
    args = parse_args()
    
    try:
        # Run inference
        run_inference(args)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 