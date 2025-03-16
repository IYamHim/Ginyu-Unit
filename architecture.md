# Ginyu-Uint Architecture

## Overview

The Namek Project is a sophisticated deep learning system for stock market prediction using transformer models. It employs a three-stage training pipeline to progressively enhance a base language model's ability to analyze financial data and make accurate predictions about stock price movements.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Stage 1      │     │    Stage 2      │     │    Stage 3      │
│  SFT Training   │────▶│  GRPO Training  │────▶│ Advanced GRPO   │
│   (Krillon.py)  │     │  (Piccolo.py)   │     │  (Kakarot.py)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Stage 1 Model  │     │  Stage 2 Model  │     │  Stage 3 Model  │
│    Checkpoint   │     │    Checkpoint   │     │    Checkpoint   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   Inference     │
                                               │   (Vegeta.py)   │
                                               └─────────────────┘
```

## Core Components

### 1. Base Model
- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Type**: Transformer-based language model
- **Size**: 1.5 billion parameters
- **Capabilities**: Pre-trained on general text data with instruction-following abilities

### 2. Training Pipeline

#### Stage 1: Supervised Fine-Tuning (Krillon.py)
- **Purpose**: Teach the model proper format and basic prediction capabilities
- **Method**: Supervised Fine-Tuning (SFT)
- **Learning Rate**: 2e-5
- **Batch Size**: 4
- **Training Steps**: 1000
- **Checkpoint Frequency**: Every 50 steps
- **Dataset Size**: 5000 samples
- **Input**: Financial data from S&P 500 companies
- **Output**: Formatted analysis with prediction

#### Stage 2: Balanced GRPO Training (Piccolo.py)
- **Purpose**: Improve prediction accuracy with balanced dataset
- **Method**: Guided Reinforcement from Policy Optimization (GRPO)
- **Learning Rate**: 1e-6
- **Batch Size**: 2
- **Training Steps**: 10000 (matching sample size)
- **Checkpoint Frequency**: Every 5 steps
- **Dataset Size**: 10000 samples (50% UP, 50% DOWN)
- **Input**: Stage 1 model + balanced financial data
- **Output**: Refined model with improved prediction accuracy

#### Stage 3: Advanced GRPO Training (Kakarot.py)
- **Purpose**: Further refine prediction accuracy and analysis quality
- **Method**: Advanced GRPO with enhanced rewards
- **Learning Rate**: 1e-6
- **Batch Size**: 1
- **Training Steps**: 20000 (sufficient for full dataset)
- **Checkpoint Frequency**: Every 5 steps
- **Dataset Size**: Full dataset (no sample limit)
- **Input**: Stage 2 model + comprehensive financial data
- **Output**: Final model with high-quality predictions and analysis

### 3. Inference Engine (Vegeta.py)
- **Purpose**: Generate predictions using trained models
- **Input**: Financial data for a specific company
- **Output**: Structured prediction with analysis and percentage change

### 4. Core Library (Over9Thousand.py)
- **Purpose**: Provides shared functionality for all stages
- **Components**:
  - GRPO training implementation
  - Reward functions
  - Data processing utilities
  - Model configuration helpers

## Data Flow

1. **Data Ingestion**:
   - Financial data from the 2084Collective/deepstock-sp500-companies-with-info-and-user-prompt dataset
   - Company information (ticker, name, sector, industry)
   - Price data (current price, previous close)
   - Technical indicators (RSI, MACD, moving averages)
   - Financial metrics (revenue, net income, EPS, P/E ratio)
   - Recent news headlines

2. **Data Processing**:
   - Formatting into structured prompts
   - Calculation of actual percentage changes for training
   - Tokenization for model input

3. **Training Flow**:
   - Stage 1: SFT training with 5,000 samples
   - Stage 2: GRPO training with 10,000 samples (balanced dataset)
   - Stage 3: Advanced GRPO training with the full dataset

4. **Inference Flow**:
   - Input financial data
   - Generate model prediction
   - Extract direction and percentage
   - Return structured analysis

## Reward System

The Namek project implements a progressive reward system that evolves across the three training stages:

### Stage 1: Supervised Fine-Tuning (Krillon.py)
- **Approach**: Uses standard cross-entropy loss without explicit rewards
- **Focus**: Teaching the model proper format and basic prediction capabilities
- **Mechanism**: Maximizes likelihood of generating correct tokens in the expected format

### Stage 2: Balanced GRPO Training (Piccolo.py)
- **Direction Prediction Rewards**:
  - Correct UP prediction: **+3.5**
  - Incorrect UP prediction: **-1.5**
  - Correct DOWN prediction: **+3.5**
  - Incorrect DOWN prediction: **-1.5**
  - Invalid/missing prediction: **-2.5**

- **Format Compliance Rewards**:
  - Including `<think>` tag: **+0.4**
  - Missing `<think>` tag: **-0.4**
  - Having Key Factors section: **+0.3**
  - Having Analysis section: **+0.3**
  - Detailed thinking section: **+0.3**
  - Short thinking section: **-0.3**

- **Percentage Accuracy Rewards**:
  - Very accurate (within 0.5%): **+1.0**
  - Reasonably accurate (within 1.0%): **+0.5**
  - Very inaccurate (off by >5.0%): **-0.5**

- **Constraints**: Rewards capped between -5.0 and +5.0

### Stage 3: Advanced GRPO Training (Kakarot.py)
- **Direction Prediction Rewards**:
  - Correct UP prediction: **+3.8** (slightly reduced to favor DOWN predictions)
  - Incorrect UP prediction: **-2.0** (stronger penalty)
  - Correct DOWN prediction: **+4.0** (higher reward to favor safer DOWN predictions)
  - Incorrect DOWN prediction: **-2.0**
  - Invalid/missing prediction: **-3.0**

- **Format Compliance Rewards**:
  - Both required tags present: **+2.0**
  - Missing tags: **-2.0** each
  - Having Key Factors section: **+1.0**
  - Having numbered factors (≥3): **+1.0**
  - Having Analysis section: **+1.0**
  - Detailed thinking section: **+0.5**
  - Short thinking section: **-1.5**
  - Multiple think/answer tags: **-2.0** each

- **Analysis Quality Rewards**:
  - Mentioning ≥3 financial metrics: **+1.5**
  - Mentioning 1-2 financial metrics: **+0.5**
  - Mentioning ≥2 technical indicators: **+1.0**
  - Mentioning 1 technical indicator: **+0.5**
  - Balanced analysis (positive and negative factors): **+1.0**

- **Minimum Reward Floor**:
  - Correct predictions always get at least **+1.0** reward

This progressive reward system guides the model through increasingly sophisticated levels of financial analysis and prediction accuracy, with each stage building upon skills learned in the previous stage. The Stage 3 reward system is specifically designed to slightly favor DOWN predictions as a safer approach in uncertain market conditions.

## Output Format

The model generates predictions in a structured format:

```
<think>
Key Factors:
1. [First key observation]
2. [Second key observation]
3. [Third key observation]

Analysis:
[Detailed analysis of technical indicators, financial metrics, and market sentiment]
</think>

<answer>direction: up change: X.X%</answer>
```

## Technical Implementation Details

### Model Adaptation
- **Parameter-Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) to efficiently fine-tune the model
- **Quantization**: Supports 4-bit and 8-bit quantization for memory efficiency
- **Gradient Checkpointing**: Optional memory optimization technique

### Training Optimization
- **Gradient Accumulation**: Used to simulate larger batch sizes
- **Learning Rate Scheduling**: Warmup steps to stabilize early training
- **Checkpoint Management**: Saves the best models and limits total checkpoints

### Inference Optimization
- **Structured Output Parsing**: Regex-based extraction of predictions
- **Format Validation**: Ensures predictions follow the required format
- **Confidence Calibration**: Adjusts prediction confidence based on market conditions

## Deployment

The system is designed to be run in a Python environment with PyTorch and Hugging Face Transformers. The training scripts can be executed directly or via the provided shell scripts:

- `train_stage1.sh`: For Stage 1 SFT training
- `train_stage2.sh`: For Stage 2 GRPO training
- `train_stage3.sh`: For Stage 3 Advanced GRPO training
- `test_inference.sh`: For testing model predictions

## Performance Metrics

The system evaluates model performance based on:

1. **Direction Accuracy**: Percentage of correct UP/DOWN predictions
2. **Percentage Accuracy**: How close the predicted percentage is to the actual change
3. **Format Compliance**: Adherence to the required output format
4. **Analysis Quality**: Depth and balance of the financial analysis

## Future Extensions

The architecture is designed to be extensible in several ways:

1. **Model Scaling**: Support for larger model variants
2. **Multi-day Predictions**: Extending to longer-term forecasts
3. **Market Regime Awareness**: Adapting predictions based on overall market conditions
4. **Ensemble Methods**: Combining predictions from multiple model checkpoints
5. **Confidence Intervals**: Adding uncertainty estimates to predictions 
