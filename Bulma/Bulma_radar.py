import os
import torch
import json
import argparse
import logging
import re
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"enhanced_model_testing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Energy sector stocks
ENERGY_STOCKS = [
    "XOM",  # ExxonMobil
    "CVX",  # Chevron
    "COP",  # ConocoPhillips
    "EOG",  # EOG Resources
    "SLB",  # Schlumberger
    "DVN",  # Devon Energy
    "OXY",  # Occidental Petroleum
    "VLO",  # Valero Energy
    "MPC",  # Marathon Petroleum
    "PSX",  # Phillips 66
    "KMI",  # Kinder Morgan
    "WMB",  # Williams Companies
    "OKE",  # ONEOK
    "HAL",  # Halliburton
    "BKR",  # Baker Hughes
    "FANG", # Diamondback Energy
    "PXD",  # Pioneer Natural Resources
    "HES",  # Hess Corporation
    "MRO",  # Marathon Oil
    "APA",  # APA Corporation
]

def get_historical_data(ticker, days_ago=1):
    """Get historical stock data for backtesting."""
    end_date = datetime.datetime.now()  # today
    start_date = end_date - datetime.timedelta(days=days_ago + 10)  # more days for context
    
    ticker_data = yf.Ticker(ticker)
    hist = ticker_data.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if hist.empty or len(hist) < 3:  # Need at least 3 days of data
        raise ValueError(f"Insufficient historical data for ticker {ticker}")
    
    # Get the date we want to predict from
    valid_dates = hist.index.tolist()
    valid_dates.sort()  # Ensure chronological order
    
    if len(valid_dates) <= days_ago + 1:
        raise ValueError(f"Not enough trading days in history for {ticker}")
    
    # Get the indices for the prediction date and verification date
    prediction_date_idx = len(valid_dates) - days_ago - 2  # -1 for today, -1 for predicting from
    verification_date_idx = len(valid_dates) - days_ago - 1
    
    if prediction_date_idx < 0 or verification_date_idx < 0:
        raise ValueError(f"Not enough trading days in history for {ticker}")
    
    prediction_date = valid_dates[prediction_date_idx]
    verification_date = valid_dates[verification_date_idx]
    
    # Get the prices
    prediction_date_data = hist.loc[prediction_date]
    verification_date_data = hist.loc[verification_date]
    
    # Get the previous close for context
    previous_close = None
    if prediction_date_idx > 0:
        previous_date = valid_dates[prediction_date_idx - 1]
        previous_close = hist.loc[previous_date]['Close']
    
    # Get sector data for context
    try:
        xle = yf.Ticker("XLE")  # Energy sector ETF
        xle_hist = xle.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        sector_change = ((xle_hist.iloc[-1]['Close'] / xle_hist.iloc[-2]['Close']) - 1) * 100
    except:
        sector_change = None
    
    # Get oil price data for context
    try:
        oil = yf.Ticker("CL=F")  # Crude oil futures
        oil_hist = oil.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        oil_change = ((oil_hist.iloc[-1]['Close'] / oil_hist.iloc[-2]['Close']) - 1) * 100
    except:
        oil_change = None
    
    logging.info(f"Using {prediction_date.strftime('%Y-%m-%d')} to predict {verification_date.strftime('%Y-%m-%d')} for {ticker}")
    
    return {
        'current_close': prediction_date_data['Close'],
        'previous_close': previous_close,
        'next_day_close': verification_date_data['Close'],
        'prediction_date': prediction_date.strftime('%Y-%m-%d'),
        'verification_date': verification_date.strftime('%Y-%m-%d'),
        'volume': prediction_date_data['Volume'],
        'high': prediction_date_data['High'],
        'low': prediction_date_data['Low'],
        'sector_change': sector_change,
        'oil_change': oil_change
    }

def format_enhanced_prompt(ticker, historical_data):
    """Format an enhanced prompt for stock prediction testing with more context."""
    current_close = historical_data['current_close']
    previous_close = historical_data['previous_close']
    prediction_date = historical_data['prediction_date']
    volume = historical_data['volume']
    high = historical_data['high']
    low = historical_data['low']
    
    # Calculate more technical indicators
    percent_change = ((current_close / previous_close) - 1) * 100 if previous_close else None
    high_low_range = ((high - low) / low) * 100 if low else None
    sector_change = historical_data['sector_change']
    oil_change = historical_data['oil_change']
    
    # Get day of week for temporal patterns
    day_of_week = datetime.datetime.strptime(prediction_date, '%Y-%m-%d').strftime('%A')
    
    # Format the prompt with more context
    prompt = f"""Analyze the following stock information and predict its price movement for the next trading day:

Company: {ticker}
Description: {ticker} is a company in the energy sector.

Recent Stock Prices:
- Current Close ({prediction_date}): ${current_close:.2f}
- Previous Close: ${previous_close:.2f if previous_close else 'Not available'}
- Daily High: ${high:.2f}
- Daily Low: ${low:.2f}

Additional Context:
- Trading Day: {day_of_week}
- Volume: {volume:,.0f}
- Daily Range: {high_low_range:.2f}% from low to high
- Previous Day Change: {percent_change:.2f}% from previous close
- Energy Sector Change: {sector_change:.2f if sector_change else 'N/A'}%
- Oil Price Change: {oil_change:.2f if oil_change else 'N/A'}%

Analyze the stock data and provide a detailed prediction for the next trading day.
Think through fundamental, technical, and sentiment factors in your analysis.
Then provide your verdict on whether the stock will go UP or DOWN for the next trading day, along with the percentage change amount.

Thinking Process:"""

    return prompt

def extract_answer(response):
    """Extract the predicted direction and percentage change from the model's response."""
    logging.info("Extracting answer from response")
    
    # First, look for the final answer section
    final_answer_match = re.search(r"Final Answer:\s*Direction:\s*(UP|DOWN|NEUTRAL|up|down|neutral)\s*Percentage Change:\s*([-+]?\d*\.?\d+)%", response, re.IGNORECASE | re.DOTALL)
    
    if final_answer_match:
        direction = final_answer_match.group(1).upper()
        try:
            percent_change = float(final_answer_match.group(2))
            return {"direction": direction, "percent_change": percent_change}
        except:
            logging.error(f"Error parsing percentage from final answer: {final_answer_match.group(2)}")
    
    # Fall back to other patterns if final answer not found
    direction_patterns = [
        r"Direction:\s*(UP|DOWN|NEUTRAL|up|down|neutral)",
        r"Final\s*(?:Answer|Decision|Verdict):\s*(UP|DOWN|BUY|SELL|NEUTRAL|up|down|buy|sell|neutral)",
        r"The most probable scenario.*?(UP|DOWN|BUY|SELL|NEUTRAL|up|down|buy|sell|neutral)\s*move",
        r"predicted\s*(?:to|will)?\s*(?:move)?\s*(UP|DOWN|BUY|SELL|NEUTRAL|up|down|buy|sell|neutral)",
        r"stock\s*is\s*likely\s*to\s*(UP|DOWN|BUY|SELL|NEUTRAL|up|down|buy|sell|neutral)"
    ]
    
    # Look for direction
    direction = "UNKNOWN"
    for pattern in direction_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            direction_text = match.group(1).upper()
            if direction_text in ["BUY", "UP"]:
                direction = "UP"
                break
            elif direction_text in ["SELL", "DOWN"]:
                direction = "DOWN"
                break
            elif direction_text == "NEUTRAL":
                direction = "NEUTRAL"
                break
    
    # Look for percentage
    percent_patterns = [
        r"Percentage Change:\s*([-+]?\d*\.?\d+)%",
        r"by\s*([-+]?\d*\.?\d+)%",
        r"([-+]?\d*\.?\d+)%\s*(?:change|move)",
        r"approximately\s*([-+]?\d*\.?\d+)%",
        r"roughly\s*([-+]?\d*\.?\d+)%"
    ]
    
    percent = 0.0
    for pattern in percent_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                percent = float(match.group(1))
                break
            except:
                logging.error(f"Error parsing percentage: {match.group(1)}")
    
    logging.info(f"Extracted direction: {direction}, percent_change: {percent}")
    return {"direction": direction, "percent_change": percent}

def test_historical_prediction(ticker, model_path, days_ago=1, temperature=0.1):
    """Test the model's prediction against historical data."""
    logging.info(f"Testing historical prediction for {ticker} using model at {model_path}")
    
    # Get historical data
    historical_data = get_historical_data(ticker, days_ago)
    logging.info(f"Retrieved historical data for {ticker}")
    
    # Format the prompt based on historical data
    prompt = format_enhanced_prompt(ticker, historical_data)
    logging.info(f"Generated historical prompt for {ticker}")
    
    # Load the model and tokenizer
    logging.info(f"Loading model from {model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Generate prediction
    logging.info("Generating prediction")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=2000,
        temperature=temperature,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer
    answer = extract_answer(response)
    
    # Calculate the actual change
    current_close = historical_data['current_close']
    next_day_close = historical_data['next_day_close']
    actual_change_pct = ((next_day_close - current_close) / current_close) * 100
    actual_direction = "UP" if next_day_close > current_close else "DOWN"
    
    # Calculate prediction accuracy
    direction_correct = answer['direction'] == actual_direction
    percent_error = abs(answer['percent_change'] - actual_change_pct)
    
    # Print results
    prediction_date = historical_data['prediction_date']
    verification_date = historical_data['verification_date']
    
    print(f"\n=== HISTORICAL PREDICTION FOR {ticker} ===")
    print(f"Prediction Date: {prediction_date}")
    print(f"Verification Date: {verification_date}")
    print(f"Price on {prediction_date}: ${current_close:.2f}")
    print(f"Predicted: {answer['direction']} by {answer['percent_change']:.2f}%")
    print(f"Predicted Price: ${current_close * (1 + answer['percent_change']/100):.2f}")
    print(f"\nActual Results:")
    print(f"Price on {verification_date}: ${next_day_close:.2f}")
    print(f"Actual: {actual_direction} by {actual_change_pct:.2f}%")
    print(f"Direction Correct: {direction_correct}")
    print(f"Percentage Error: {percent_error:.2f}%")
    
    # Also print the model's thinking process for analysis
    thinking_match = re.search(r"Thinking Process:(.*?)(?:Final Answer:|$)", response, re.DOTALL | re.IGNORECASE)
    thinking = thinking_match.group(1).strip() if thinking_match else "No thinking process found"
    
    print(f"\n=== MODEL'S THINKING PROCESS ===")
    print(thinking)
    
    return {
        "ticker": ticker,
        "prediction_date": prediction_date,
        "verification_date": verification_date,
        "price_on_prediction_date": current_close,
        "predicted_direction": answer['direction'],
        "predicted_change": answer['percent_change'],
        "predicted_price": current_close * (1 + answer['percent_change']/100),
        "actual_price": next_day_close,
        "actual_direction": actual_direction,
        "actual_change": actual_change_pct,
        "direction_correct": direction_correct,
        "percent_error": percent_error,
        "full_response": response,
        "thinking": thinking
    }

def test_multiple_tickers(tickers, model_path, days_ago=1, temperature=0.1, output_file=None):
    """Test the model on multiple tickers and compile results."""
    results = []
    
    for ticker in tickers:
        try:
            logging.info(f"Processing {ticker}...")
            result = test_historical_prediction(ticker, model_path, days_ago, temperature)
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            print(f"{ticker}: Failed to generate prediction - {e}")
    
    # Calculate overall stats
    if results:
        correct_directions = sum(1 for r in results if r['direction_correct'])
        direction_accuracy = (correct_directions / len(results)) * 100
        avg_percent_error = sum(r['percent_error'] for r in results) / len(results)
        
        # Create confusion matrix
        y_true = [r['actual_direction'] for r in results]
        y_pred = [r['predicted_direction'] for r in results]
        
        labels = ["UP", "DOWN"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Calculate additional metrics
        classification_metrics = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        
        # Print results
        print("\n=== OVERALL RESULTS ===")
        print(f"Total Tickers: {len(results)}")
        print(f"Correct Directions: {correct_directions}/{len(results)} ({direction_accuracy:.2f}%)")
        print(f"Average Percentage Error: {avg_percent_error:.2f}%")
        
        print("\n=== CONFUSION MATRIX ===")
        print("              Predicted")
        print("              UP    DOWN")
        print(f"Actual  UP   {cm[0][0]}     {cm[0][1]}")
        print(f"        DOWN {cm[1][0]}     {cm[1][1]}")
        
        print("\n=== CLASSIFICATION METRICS ===")
        for label in labels:
            print(f"{label} Precision: {classification_metrics[label]['precision']:.2f}")
            print(f"{label} Recall: {classification_metrics[label]['recall']:.2f}")
            print(f"{label} F1-score: {classification_metrics[label]['f1-score']:.2f}")
            print("")
        
        # Check for UP/DOWN bias
        up_predictions = sum(1 for r in results if r['predicted_direction'] == "UP")
        up_percentage = (up_predictions / len(results)) * 100
        print(f"UP Predictions: {up_predictions}/{len(results)} ({up_percentage:.2f}%)")
        print(f"DOWN Predictions: {len(results) - up_predictions}/{len(results)} ({100 - up_percentage:.2f}%)")
        
        # Save to CSV if specified
        if output_file:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            logging.info(f"Saved results to {output_file}")
            print(f"\nDetailed results saved to {output_file}")
            
            # Create plots
            plot_path = output_file.replace('.csv', '_plots.png')
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Direction accuracy by ticker
            ticker_accuracy = df.groupby('ticker')['direction_correct'].mean() * 100
            ticker_accuracy.plot(kind='bar', ax=axs[0, 0], color='skyblue')
            axs[0, 0].set_title('Direction Accuracy by Ticker')
            axs[0, 0].set_ylabel('Accuracy (%)')
            axs[0, 0].set_ylim(0, 100)
            
            # Plot 2: Actual vs predicted percentage change
            axs[0, 1].scatter(df['actual_change'], df['predicted_change'], alpha=0.7)
            axs[0, 1].set_title('Actual vs Predicted Percentage Change')
            axs[0, 1].set_xlabel('Actual Change (%)')
            axs[0, 1].set_ylabel('Predicted Change (%)')
            max_val = max(abs(df['actual_change'].max()), abs(df['predicted_change'].max())) * 1.1
            axs[0, 1].set_xlim(-max_val, max_val)
            axs[0, 1].set_ylim(-max_val, max_val)
            axs[0, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axs[0, 1].axvline(x=0, color='r', linestyle='-', alpha=0.3)
            axs[0, 1].plot([-max_val, max_val], [-max_val, max_val], 'g--', alpha=0.3)
            
            # Plot 3: Percentage error distribution
            axs[1, 0].hist(df['percent_error'], bins=20, color='lightgreen', edgecolor='black')
            axs[1, 0].set_title('Percentage Error Distribution')
            axs[1, 0].set_xlabel('Percentage Error')
            axs[1, 0].set_ylabel('Count')
            
            # Plot 4: Confusion matrix
            im = axs[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axs[1, 1].set_title('Confusion Matrix')
            axs[1, 1].set_xticks([0, 1])
            axs[1, 1].set_yticks([0, 1])
            axs[1, 1].set_xticklabels(labels)
            axs[1, 1].set_yticklabels(labels)
            axs[1, 1].set_xlabel('Predicted')
            axs[1, 1].set_ylabel('Actual')
            
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axs[1, 1].text(j, i, cm[i, j], 
                        ha="center", va="center", 
                        color="white" if cm[i, j] > cm.max()/2 else "black")
            
            plt.tight_layout()
            plt.savefig(plot_path)
            logging.info(f"Saved plots to {plot_path}")
            print(f"Visualization plots saved to {plot_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test the enhanced directional model against historical data.")
    parser.add_argument("--ticker", type=str, help="Single stock ticker to test")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers to test")
    parser.add_argument("--model_path", type=str, default="output/enhanced_directional_model", help="Path to the model")
    parser.add_argument("--days_ago", type=int, default=1, help="Days ago to test (default: 1 for yesterday)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--output", type=str, default="enhanced_directional_predictions.csv", help="Output CSV file for results")
    parser.add_argument("--test_all", action="store_true", help="Test all energy stocks")
    
    args = parser.parse_args()
    
    if args.test_all:
        # Test all energy stocks
        test_multiple_tickers(ENERGY_STOCKS, args.model_path, args.days_ago, args.temperature, args.output)
    elif args.tickers:
        # Test a comma-separated list of tickers
        ticker_list = [t.strip() for t in args.tickers.split(',')]
        test_multiple_tickers(ticker_list, args.model_path, args.days_ago, args.temperature, args.output)
    elif args.ticker:
        # Test a single ticker
        test_historical_prediction(args.ticker, args.model_path, args.days_ago, args.temperature)
    else:
        # Default to testing all stocks if no specific ticker is provided
        test_multiple_tickers(ENERGY_STOCKS, args.model_path, args.days_ago, args.temperature, args.output)

if __name__ == "__main__":
    main() 