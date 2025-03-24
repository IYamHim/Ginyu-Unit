import os
import json
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Energy sector stocks
ENERGY_STOCKS = [
    "XOM", "CVX", "SLB", "COP", "EOG", "PXD", "PSX", "MPC", "OXY", "VLO",
    "KMI", "WMB", "DVN", "HAL", "OKE", "BKR", "FANG", "MRO", "APA"
]

def select_ticker_with_distribution(stock_distribution, total_examples):
    """Select a ticker with balanced distribution."""
    min_examples_per_stock = 40  # Ensure at least 40 examples per stock
    
    # Find stocks with fewer than min_examples_per_stock
    eligible_stocks = [
        ticker for ticker in ENERGY_STOCKS 
        if stock_distribution[ticker] < min_examples_per_stock
    ]
    
    # If all stocks have reached minimum, select based on overall distribution
    if not eligible_stocks:
        return random.choice(ENERGY_STOCKS)
    
    # Prioritize underrepresented stocks
    return random.choice(eligible_stocks)

def generate_bearish_technicals():
    """Generate bearish technical indicators."""
    
    # Death Cross scenario (50-day MA below 200-day MA)
    death_cross_chance = random.random() < 0.6
    
    # Overbought RSI scenario
    rsi_value = random.randint(70, 85) if random.random() < 0.7 else random.randint(45, 69)
    
    # MACD bearish divergence
    macd_bearish = random.random() < 0.7
    
    # Bearish candlestick patterns
    candlestick_patterns = [
        "Shooting Star",
        "Bearish Engulfing",
        "Evening Star",
        "Dark Cloud Cover",
        "Bearish Harami"
    ]
    bearish_pattern = random.choice(candlestick_patterns) if random.random() < 0.6 else None
    
    # Bollinger Band indicators
    upper_band_touch = random.random() < 0.65
    
    # Volume indicators
    volume_increase = random.randint(15, 50) if random.random() < 0.65 else random.randint(1, 10)
    
    return {
        "death_cross": death_cross_chance,
        "rsi": rsi_value,
        "macd_bearish": macd_bearish,
        "candlestick_pattern": bearish_pattern,
        "upper_band_touch": upper_band_touch,
        "volume_increase": volume_increase
    }

def generate_negative_fundamentals():
    """Generate negative fundamental factors."""
    
    # Earnings miss
    earnings_miss_percent = round(random.uniform(1.5, 8.5), 1) if random.random() < 0.7 else 0
    
    # Declining production
    production_decline = round(random.uniform(0.8, 3.5), 1) if random.random() < 0.7 else 0
    
    # Profit margin contraction
    margin_contraction = round(random.uniform(0.5, 3.2), 1) if random.random() < 0.65 else 0
    
    # Increased debt levels
    debt_increase = round(random.uniform(2.0, 7.5), 1) if random.random() < 0.55 else 0
    
    # Downward analyst revisions
    analyst_revision = round(random.uniform(-5.0, -1.0), 1) if random.random() < 0.7 else 0
    
    return {
        "earnings_miss": earnings_miss_percent,
        "production_decline": production_decline,
        "margin_contraction": margin_contraction,
        "debt_increase": debt_increase,
        "analyst_revision": analyst_revision
    }

def generate_bearish_market():
    """Generate bearish market context."""
    
    # Energy sector underperformance
    sector_underperformance = round(random.uniform(0.8, 3.0), 1) if random.random() < 0.75 else 0
    
    # Oil price movement (negative for energy stocks)
    oil_decline = round(random.uniform(0.5, 2.5), 1) if random.random() < 0.7 else 0
    
    # Oversupply concerns
    oversupply_concern = random.random() < 0.65
    
    # Rising interest rates (negative for capital-intensive energy companies)
    interest_rate_increase = random.random() < 0.6
    
    # Global economic slowdown
    economic_slowdown = random.random() < 0.55
    
    return {
        "sector_underperformance": sector_underperformance,
        "oil_decline": oil_decline,
        "oversupply_concern": oversupply_concern,
        "interest_rate_increase": interest_rate_increase,
        "economic_slowdown": economic_slowdown
    }

def generate_negative_sentiment():
    """Generate negative market sentiment factors."""
    
    # Bearish analyst coverage
    bearish_analysts = random.randint(2, 7) if random.random() < 0.7 else random.randint(0, 1)
    
    # Social media sentiment (negative)
    social_sentiment_score = round(random.uniform(-0.7, -0.2), 2) if random.random() < 0.65 else round(random.uniform(-0.1, 0.1), 2)
    
    # Increased short interest
    short_interest_increase = round(random.uniform(5.0, 15.0), 1) if random.random() < 0.7 else 0
    
    # Negative news coverage
    negative_news_count = random.randint(2, 5) if random.random() < 0.6 else 0
    
    return {
        "bearish_analysts": bearish_analysts,
        "social_sentiment": social_sentiment_score,
        "short_interest_increase": short_interest_increase,
        "negative_news": negative_news_count
    }

def generate_bullish_technicals():
    """Generate bullish technical indicators for UP scenarios."""
    
    # Golden Cross scenario (50-day MA above 200-day MA)
    golden_cross = random.random() < 0.65
    
    # Oversold RSI scenario (bullish)
    rsi_value = random.randint(25, 35) if random.random() < 0.7 else random.randint(36, 55)
    
    # MACD bullish divergence
    macd_bullish = random.random() < 0.7
    
    # Bullish candlestick patterns
    candlestick_patterns = [
        "Hammer",
        "Bullish Engulfing",
        "Morning Star",
        "Piercing Pattern",
        "Bullish Harami"
    ]
    bullish_pattern = random.choice(candlestick_patterns) if random.random() < 0.6 else None
    
    # Bollinger Band indicators
    lower_band_bounce = random.random() < 0.65
    
    # Volume indicators
    volume_increase = random.randint(15, 50) if random.random() < 0.65 else random.randint(1, 10)
    
    return {
        "golden_cross": golden_cross,
        "rsi": rsi_value,
        "macd_bullish": macd_bullish,
        "candlestick_pattern": bullish_pattern,
        "lower_band_bounce": lower_band_bounce,
        "volume_increase": volume_increase
    }

def generate_positive_fundamentals():
    """Generate positive fundamental factors for UP scenarios."""
    
    # Earnings beat
    earnings_beat_percent = round(random.uniform(1.5, 8.5), 1) if random.random() < 0.7 else 0
    
    # Increasing production
    production_increase = round(random.uniform(0.8, 3.5), 1) if random.random() < 0.7 else 0
    
    # Profit margin expansion
    margin_expansion = round(random.uniform(0.5, 3.2), 1) if random.random() < 0.65 else 0
    
    # Decreased debt levels
    debt_decrease = round(random.uniform(2.0, 7.5), 1) if random.random() < 0.55 else 0
    
    # Upward analyst revisions
    analyst_revision = round(random.uniform(1.0, 5.0), 1) if random.random() < 0.7 else 0
    
    return {
        "earnings_beat": earnings_beat_percent,
        "production_increase": production_increase,
        "margin_expansion": margin_expansion,
        "debt_decrease": debt_decrease,
        "analyst_revision": analyst_revision
    }

def generate_bullish_market():
    """Generate bullish market context for UP scenarios."""
    
    # Energy sector outperformance
    sector_outperformance = round(random.uniform(0.8, 3.0), 1) if random.random() < 0.75 else 0
    
    # Oil price movement (positive for energy stocks)
    oil_increase = round(random.uniform(0.5, 2.5), 1) if random.random() < 0.7 else 0
    
    # Supply constraint concerns
    supply_constraint = random.random() < 0.65
    
    # Declining interest rates (positive for capital-intensive energy companies)
    interest_rate_decrease = random.random() < 0.6
    
    # Global economic recovery
    economic_recovery = random.random() < 0.55
    
    return {
        "sector_outperformance": sector_outperformance,
        "oil_increase": oil_increase,
        "supply_constraint": supply_constraint,
        "interest_rate_decrease": interest_rate_decrease,
        "economic_recovery": economic_recovery
    }

def generate_positive_sentiment():
    """Generate positive market sentiment factors for UP scenarios."""
    
    # Bullish analyst coverage
    bullish_analysts = random.randint(2, 7) if random.random() < 0.7 else random.randint(0, 1)
    
    # Social media sentiment (positive)
    social_sentiment_score = round(random.uniform(0.2, 0.7), 2) if random.random() < 0.65 else round(random.uniform(-0.1, 0.1), 2)
    
    # Decreased short interest
    short_interest_decrease = round(random.uniform(5.0, 15.0), 1) if random.random() < 0.7 else 0
    
    # Positive news coverage
    positive_news_count = random.randint(2, 5) if random.random() < 0.6 else 0
    
    return {
        "bullish_analysts": bullish_analysts,
        "social_sentiment": social_sentiment_score,
        "short_interest_decrease": short_interest_decrease,
        "positive_news": positive_news_count
    }

def create_coherent_prediction(is_bearish=True):
    """Create a coherent prediction based on scenario."""
    
    if is_bearish:
        # Generate bearish scenario
        direction = "DOWN"
        # Percentage change more likely to be larger for DOWN predictions
        percentage = round(random.uniform(0.3, 3.5), 2)
    else:
        # Generate bullish scenario
        direction = "UP"
        # Percentage change more constrained for UP predictions
        percentage = round(random.uniform(0.2, 2.5), 2)
        
    return direction, percentage

def format_bearish_thinking_process(ticker, signals):
    """Create a detailed bearish thinking process with concrete indicators."""
    
    technicals = signals["technicals"]
    fundamentals = signals["fundamentals"]
    market = signals["market"]
    sentiment = signals["sentiment"]
    
    # Build the death cross description
    death_cross_text = "Death Cross: 50-day MA crossed below 200-day MA, signaling bearish momentum" if technicals["death_cross"] else "No Death Cross pattern present"
    
    # Build the RSI description
    if technicals["rsi"] > 70:
        rsi_text = f"RSI showing overbought conditions at {technicals['rsi']}, suggesting potential reversal"
    else:
        rsi_text = f"RSI at {technicals['rsi']}, showing neutral momentum"
    
    # Build the MACD description
    macd_text = "MACD showing bearish divergence, confirming downtrend" if technicals["macd_bearish"] else "MACD showing mixed signals"
    
    # Build the candlestick pattern description
    candlestick_text = f"Bearish {technicals['candlestick_pattern']} pattern identified" if technicals["candlestick_pattern"] else "No clear bearish candlestick pattern"
    
    # Build the earnings miss description
    earnings_text = f"Recent earnings miss by {fundamentals['earnings_miss']}%" if fundamentals["earnings_miss"] > 0 else "Earnings in line with expectations"
    
    # Build the production decline description
    production_text = f"Declining production volume by {fundamentals['production_decline']}%" if fundamentals["production_decline"] > 0 else "Production volume stable"
    
    # Build the margin contraction description
    margin_text = f"Narrowing profit margins (down {fundamentals['margin_contraction']}%)" if fundamentals["margin_contraction"] > 0 else "Profit margins stable"
    
    # Build the sector performance description
    sector_text = f"Energy sector underperforming broader market by {market['sector_underperformance']}%" if market["sector_underperformance"] > 0 else "Energy sector performance in line with market"
    
    # Build the oil price description
    oil_text = f"Declining oil futures pointing to lower prices (down {market['oil_decline']}%)" if market["oil_decline"] > 0 else "Oil price relatively stable"
    
    # Build the supply concern description
    supply_text = "Rising inventory levels suggesting demand weakness" if market["oversupply_concern"] else "Supply and demand relatively balanced"
    
    # Create the full thinking process
    process = f"""
1. Bearish Technical Indicators:
   - {death_cross_text}
   - {rsi_text}
   - {macd_text}
   - {candlestick_text}
   
2. Negative Fundamental Outlook:
   - {earnings_text}
   - {production_text}
   - {margin_text}
   - Debt levels {f"increased by {fundamentals['debt_increase']}%" if fundamentals['debt_increase'] > 0 else "remain stable"}
   
3. Sector Weakness:
   - {sector_text}
   - {oil_text}
   - {supply_text}
   - {"Interest rates trending higher, increasing borrowing costs" if market['interest_rate_increase'] else "Interest rate environment stable"}
   
4. Bearish Sentiment:
   - {f"{sentiment['bearish_analysts']} analysts with bearish outlook" if sentiment['bearish_analysts'] > 1 else "Limited bearish analyst coverage"}
   - Social media sentiment negative (score: {sentiment['social_sentiment']})
   - {f"Short interest increased by {sentiment['short_interest_increase']}%" if sentiment['short_interest_increase'] > 0 else "Short interest stable"}
   - {f"{sentiment['negative_news']} negative news stories in past week" if sentiment['negative_news'] > 0 else "Limited negative news coverage"}

Conclusion:
After analyzing the technical indicators, fundamental factors, sector dynamics, and market sentiment, I anticipate {ticker} will experience downward momentum in the next trading session.

Verdict: Directional Prediction: DOWN - {random.uniform(0.3, 2.5):.2f}%
The combination of {random.choice(["bearish technical patterns", "negative fundamentals", "sector weakness", "negative sentiment"])} and {random.choice(["bearish technical patterns", "negative fundamentals", "sector weakness", "negative sentiment"])} points to a decline of approximately {random.uniform(0.3, 2.5):.2f}% for tomorrow's session.
"""
    
    return process

def format_bullish_thinking_process(ticker, signals):
    """Create a detailed bullish thinking process with concrete indicators."""
    
    technicals = signals["technicals"]
    fundamentals = signals["fundamentals"]
    market = signals["market"]
    sentiment = signals["sentiment"]
    
    # Build the golden cross description
    golden_cross_text = "Golden Cross: 50-day MA crossed above 200-day MA, signaling bullish momentum" if technicals["golden_cross"] else "No Golden Cross pattern present"
    
    # Build the RSI description
    if technicals["rsi"] < 36:
        rsi_text = f"RSI showing oversold conditions at {technicals['rsi']}, suggesting potential rebound"
    else:
        rsi_text = f"RSI at {technicals['rsi']}, showing neutral momentum"
    
    # Build the MACD description
    macd_text = "MACD showing bullish divergence, confirming uptrend" if technicals["macd_bullish"] else "MACD showing mixed signals"
    
    # Build the candlestick pattern description
    candlestick_text = f"Bullish {technicals['candlestick_pattern']} pattern identified" if technicals["candlestick_pattern"] else "No clear bullish candlestick pattern"
    
    # Build the earnings beat description
    earnings_text = f"Recent earnings beat by {fundamentals['earnings_beat']}%" if fundamentals["earnings_beat"] > 0 else "Earnings in line with expectations"
    
    # Build the production increase description
    production_text = f"Increasing production volume by {fundamentals['production_increase']}%" if fundamentals["production_increase"] > 0 else "Production volume stable"
    
    # Build the margin expansion description
    margin_text = f"Expanding profit margins (up {fundamentals['margin_expansion']}%)" if fundamentals["margin_expansion"] > 0 else "Profit margins stable"
    
    # Build the sector performance description
    sector_text = f"Energy sector outperforming broader market by {market['sector_outperformance']}%" if market["sector_outperformance"] > 0 else "Energy sector performance in line with market"
    
    # Build the oil price description
    oil_text = f"Rising oil futures pointing to higher prices (up {market['oil_increase']}%)" if market["oil_increase"] > 0 else "Oil price relatively stable"
    
    # Build the supply constraint description
    supply_text = "Supply constraints suggesting higher energy prices" if market["supply_constraint"] else "Supply and demand relatively balanced"
    
    # Create the full thinking process
    process = f"""
1. Bullish Technical Indicators:
   - {golden_cross_text}
   - {rsi_text}
   - {macd_text}
   - {candlestick_text}
   
2. Positive Fundamental Outlook:
   - {earnings_text}
   - {production_text}
   - {margin_text}
   - Debt levels {f"decreased by {fundamentals['debt_decrease']}%" if fundamentals['debt_decrease'] > 0 else "remain stable"}
   
3. Sector Strength:
   - {sector_text}
   - {oil_text}
   - {supply_text}
   - {"Interest rates trending lower, decreasing borrowing costs" if market['interest_rate_decrease'] else "Interest rate environment stable"}
   
4. Bullish Sentiment:
   - {f"{sentiment['bullish_analysts']} analysts with bullish outlook" if sentiment['bullish_analysts'] > 1 else "Limited bullish analyst coverage"}
   - Social media sentiment positive (score: {sentiment['social_sentiment']})
   - {f"Short interest decreased by {sentiment['short_interest_decrease']}%" if sentiment['short_interest_decrease'] > 0 else "Short interest stable"}
   - {f"{sentiment['positive_news']} positive news stories in past week" if sentiment['positive_news'] > 0 else "Limited positive news coverage"}

Conclusion:
After analyzing the technical indicators, fundamental factors, sector dynamics, and market sentiment, I anticipate {ticker} will experience upward momentum in the next trading session.

Verdict: Directional Prediction: UP - {random.uniform(0.2, 2.0):.2f}%
The combination of {random.choice(["bullish technical patterns", "positive fundamentals", "sector strength", "positive sentiment"])} and {random.choice(["bullish technical patterns", "positive fundamentals", "sector strength", "positive sentiment"])} points to an increase of approximately {random.uniform(0.2, 2.0):.2f}% for tomorrow's session.
"""
    
    return process

def generate_mixed_signals_bearish_scenario():
    """Generate a scenario with mixed signals but ultimately bearish outcome."""
    
    # Mixed technical indicators (some bullish, some bearish)
    technical_signals = {
        # Bearish signals
        "death_cross": random.random() < 0.5,
        "macd_bearish": random.random() < 0.5,
        
        # Bullish signals
        "rsi": random.randint(35, 45),  # Neutral to slightly oversold (bullish)
        "golden_cross": random.random() < 0.3,  # Sometimes present (contradicting death cross)
        
        # Neutral signals
        "volume_increase": random.randint(5, 15)  # Moderate volume
    }
    
    # Mixed fundamentals (some positive, some negative)
    fundamentals = {
        # Positive signals
        "earnings_beat": random.random() < 0.6 and round(random.uniform(0.5, 2.5), 1) or 0,
        "positive_guidance": random.random() < 0.5,
        
        # Negative signals
        "margin_contraction": random.random() < 0.7 and round(random.uniform(0.8, 3.0), 1) or 0,
        "debt_increase": random.random() < 0.6 and round(random.uniform(1.5, 5.0), 1) or 0,
        "analyst_revision": round(random.uniform(-3.0, -0.5), 1)  # Downward revision
    }
    
    # Mixed market context (some positive, some negative)
    market = {
        # Positive signals
        "market_rally": random.random() < 0.6,  # Overall market is up
        "peer_strength": random.random() < 0.5,  # Some peers showing strength
        
        # Negative signals
        "sector_underperformance": round(random.uniform(0.3, 1.5), 1),
        "oil_decline": random.random() < 0.7 and round(random.uniform(0.5, 2.0), 1) or 0,
        "high_volatility": random.random() < 0.7  # High volatility (negative for energy)
    }
    
    # Mixed sentiment (some positive, some negative)
    sentiment = {
        # Mixed analyst coverage
        "bullish_analysts": random.randint(1, 3),
        "bearish_analysts": random.randint(2, 5),
        
        # Negative social sentiment but with some positive chatter
        "social_sentiment": round(random.uniform(-0.4, -0.1), 2),
        "mixed_news": True
    }
    
    return {
        "technicals": technical_signals,
        "fundamentals": fundamentals,
        "market": market,
        "sentiment": sentiment
    }

def format_mixed_signals_bearish_thinking(ticker, signals):
    """Create a thinking process for ambiguous but ultimately bearish scenario."""
    
    technicals = signals["technicals"]
    fundamentals = signals["fundamentals"]
    market = signals["market"]
    sentiment = signals["sentiment"]
    
    # Create competing narratives that ultimately lead to DOWN
    process = f"""
1. Mixed Technical Indicators:
   - {"Death Cross present, suggesting bearish momentum" if technicals.get("death_cross") else "No Death Cross pattern visible"}
   - {"Golden Cross recently formed, typically a bullish signal" if technicals.get("golden_cross") else "No Golden Cross formation"}
   - RSI at {technicals.get("rsi", "N/A")}, showing neither overbought nor oversold conditions
   - {"MACD shows bearish divergence" if technicals.get("macd_bearish") else "MACD shows mixed signals"}
   - Volume increase of {technicals.get("volume_increase", "N/A")}% is inconclusive
   
2. Contradicting Fundamental Factors:
   - {"Recent earnings beat expectations by "+str(fundamentals.get("earnings_beat"))+"%," if fundamentals.get("earnings_beat") else "Earnings in line with expectations,"} {"with positive forward guidance" if fundamentals.get("positive_guidance") else "but guidance was neutral"}
   - However, profit margins contracted by {fundamentals.get("margin_contraction", "N/A")}%
   - Debt levels {"increased by "+str(fundamentals.get("debt_increase"))+"%," if fundamentals.get("debt_increase") else "remained stable,"}
   - Analyst consensus has been revised downward by {fundamentals.get("analyst_revision", "N/A")}%
   
3. Conflicting Market Context:
   - {"Broader market is rallying, which typically lifts all boats" if market.get("market_rally") else "Broader market is neutral to slightly negative"}
   - {"Some peers showing strength" if market.get("peer_strength") else "Peers generally weak"}
   - Yet energy sector is underperforming the broader market by {market.get("sector_underperformance", "N/A")}%
   - {"Oil prices declining by "+str(market.get("oil_decline"))+"%," if market.get("oil_decline") else "Oil prices relatively stable,"} {"with high volatility" if market.get("high_volatility") else "with normal volatility"}
   
4. Mixed Investor Sentiment:
   - {sentiment.get("bullish_analysts", 0)} analysts with bullish outlook vs. {sentiment.get("bearish_analysts", 0)} with bearish stance
   - Social media sentiment slightly negative (score: {sentiment.get("social_sentiment", "N/A")})
   - Mixed news cycle with both positive and negative headlines

Conclusion:
Despite some positive signals, the weight of evidence suggests {ticker} will face downward pressure in the near term. The combination of margin contraction, analyst downgrades, sector underperformance, and negative sentiment outweighs the positive earnings results and broader market strength.

Verdict: Directional Prediction: DOWN - {random.uniform(0.2, 1.0):.2f}%
While this is a close call with conflicting indicators, we expect a modest decline tomorrow as the negative catalysts assert more influence than the positive ones.
"""
    
    return process

def generate_enhanced_next_day_dataset(num_examples, output_dir, down_percent=0.60):
    """Generate an enhanced dataset with 60% DOWN bias for next-day stock prediction."""
    
    examples = []
    stock_distribution = defaultdict(int)
    up_count = down_count = 0
    ambiguous_down_count = 0
    
    # Calculate target counts based on down_percent
    target_down = int(num_examples * down_percent)
    target_up = num_examples - target_down
    
    # Calculate target for ambiguous DOWN examples (25% of DOWN examples)
    target_ambiguous_down = int(target_down * 0.25)
    
    logging.info(f"Generating dataset with {down_percent*100:.1f}% DOWN examples ({target_down} DOWN, {target_up} UP)")
    logging.info(f"Including {target_ambiguous_down} ambiguous DOWN examples with mixed signals")
    
    # Generate examples
    for _ in tqdm(range(num_examples)):
        # Select a ticker with proper distribution
        ticker = select_ticker_with_distribution(stock_distribution, num_examples)
        stock_distribution[ticker] += 1
        
        # Determine if this example should be bearish based on current counts
        should_be_bearish = down_count < target_down and (
            up_count >= target_up or random.random() < down_percent
        )
        
        if should_be_bearish:
            # Determine if this should be an ambiguous DOWN example
            should_be_ambiguous = ambiguous_down_count < target_ambiguous_down and random.random() < 0.5
            
            if should_be_ambiguous:
                # Generate ambiguous but bearish scenario
                signals = generate_mixed_signals_bearish_scenario()
                direction, percentage = create_coherent_prediction(is_bearish=True)
                thinking_process = format_mixed_signals_bearish_thinking(ticker, signals)
                down_count += 1
                ambiguous_down_count += 1
            else:
                # Generate clearly bearish scenario
                technicals = generate_bearish_technicals()
                fundamentals = generate_negative_fundamentals()
                market_context = generate_bearish_market()
                sentiment = generate_negative_sentiment()
                
                signals = {
                    "technicals": technicals,
                    "fundamentals": fundamentals,
                    "market": market_context,
                    "sentiment": sentiment
                }
                
                direction, percentage = create_coherent_prediction(is_bearish=True)
                thinking_process = format_bearish_thinking_process(ticker, signals)
                down_count += 1
        else:
            # Generate bullish scenario
            technicals = generate_bullish_technicals()
            fundamentals = generate_positive_fundamentals()
            market_context = generate_bullish_market()
            sentiment = generate_positive_sentiment()
            
            signals = {
                "technicals": technicals,
                "fundamentals": fundamentals,
                "market": market_context,
                "sentiment": sentiment
            }
            
            direction, percentage = create_coherent_prediction(is_bearish=False)
            thinking_process = format_bullish_thinking_process(ticker, signals)
            up_count += 1
        
        # Create current price and previous close values
        current_price = round(random.uniform(20.0, 200.0), 2)
        previous_close = round(current_price * (1 - random.uniform(-0.015, 0.015)), 2)
        
        # Create the example
        example = {
            "company": ticker,
            "description": f"{ticker} is a company in the energy sector.",
            "current_price": current_price,
            "previous_close": previous_close,
            "thinking_process": thinking_process,
            "direction": direction,
            "percentage_change": percentage
        }
        
        examples.append(example)
    
    # Generate the text format examples
    text_examples = []
    for example in examples:
        text = f"""
<example>
<company>{example["company"]}</company>
<description>{example["description"]}</description>
<current_price>${example["current_price"]:.2f}</current_price>
<previous_close>${example["previous_close"]:.2f}</previous_close>

<thinking>
{example["thinking_process"]}
</thinking>

<answer>
Direction: {example["direction"]}
Percentage Change: {example["percentage_change"]}%
</answer>
</example>
"""
        text_examples.append({"text": text})
    
    # Save the examples
    output_path = os.path.join(output_dir, "down_enhanced_dataset.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    # Save the text examples
    text_output_path = os.path.join(output_dir, "down_enhanced_dataset_text.jsonl")
    with open(text_output_path, "w") as f:
        for example in text_examples:
            f.write(json.dumps(example) + "\n")
    
    # Log statistics
    logging.info(f"Generated {len(examples)} examples")
    logging.info(f"UP predictions: {up_count} ({up_count/num_examples*100:.2f}%)")
    logging.info(f"DOWN predictions: {down_count} ({down_count/num_examples*100:.2f}%)")
    logging.info(f"Ambiguous DOWN predictions: {ambiguous_down_count} ({ambiguous_down_count/num_examples*100:.2f}%)")
    
    # Log distribution by stock
    logging.info("Distribution by stock:")
    for stock, count in sorted(stock_distribution.items()):
        logging.info(f"  {stock}: {count} examples")
    
    logging.info(f"Saved examples to {output_path}")
    logging.info(f"Saved text examples to {text_output_path}")
    
    return examples, up_count, down_count, output_path, text_output_path

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced next day prediction dataset with DOWN bias")
    parser.add_argument("--num_examples", type=int, default=3000, help="Number of examples to generate")
    parser.add_argument("--output_dir", type=str, default="down_enhanced_dataset", help="Output directory")
    parser.add_argument("--down_percent", type=float, default=0.60, help="Percentage of DOWN examples")
    args = parser.parse_args()
    
    generate_enhanced_next_day_dataset(
        args.num_examples, 
        args.output_dir,
        args.down_percent
    )

if __name__ == "__main__":
    main() 