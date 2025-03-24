# Down-Prediction Enhancement Plan

## 1. Enhanced Dataset Generation

### Dataset Specifications
- **Size**: 3,000 examples (up from 1,000)
- **Class Balance**: 60% DOWN / 40% UP (reversed from current distribution)
- **Structure**: Same JSON format with improved thinking process
- **Focus Areas**: Add more examples with:
  - Technical indicators signaling bearish trends
  - Negative earnings surprises and revenue misses
  - Sector-wide energy downturns
  - Commodity price drops affecting energy stocks
  - Bearish analyst sentiment

### Implementation Steps
1. Modify `generate_enhanced_next_day_dataset.py`:
   ```python
   # Update target distribution
   target_down_percent = 0.60  # 60% DOWN examples
   
   # Add more sophisticated DOWN patterns
   def generate_bearish_scenario():
      # Generate complex bearish scenarios
      return {
         "technical_indicators": generate_bearish_technicals(),
         "fundamental_factors": generate_negative_fundamentals(),
         "market_context": generate_bearish_market(),
         "sentiment": generate_negative_sentiment()
      }
   ```

2. Generate more nuanced thinking processes:
   ```python
   def format_bearish_thinking_process(ticker, signals):
      # Create more detailed bearish reasoning with concrete technical indicators
      process = f"""
      1. Bearish Technical Indicators:
         - Death Cross: 50-day MA crossed below 200-day MA
         - RSI showing overbought conditions at 78
         - MACD showing bearish divergence
         
      2. Negative Fundamental Outlook:
         - Recent earnings miss by 3.2%
         - Declining production volume by 1.8%
         - Narrowing profit margins (down 2.1%)
         
      3. Sector Weakness:
         - Energy sector underperforming broader market by 2.3%
         - Declining oil futures pointing to lower prices
         - Rising inventory levels suggesting demand weakness
      """
      return process
   ```

## 2. Model & Training Improvements

### Model Options
- **Option 1**: Continue with current architecture + enhanced training
   - Base: Qwen2.5-1.5B-Instruct with improved LoRA settings
   - LoRA: r=32, alpha=64 (double current values)
   
- **Option 2**: Upgrade to 3B model (compute permitting)
   - Base: Qwen2.5-3B-Instruct or Mistral-3B
   - LoRA: r=32, alpha=64

### Custom Loss Function Implementation
1. Create a weighted directional loss:
   ```python
   class DirectionalWeightedLoss(nn.Module):
       def __init__(self, down_weight=2.0, up_weight=1.0):
           super().__init__()
           self.down_weight = down_weight
           self.up_weight = up_weight
           self.base_loss = nn.CrossEntropyLoss(reduction='none')
           
       def forward(self, logits, labels, direction_masks):
           base_loss = self.base_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
           
           # Apply direction-based weights
           weighted_loss = torch.where(
               direction_masks.view(-1),  # TRUE for DOWN examples
               base_loss * self.down_weight,
               base_loss * self.up_weight
           )
           
           return weighted_loss.mean()
   ```

2. Create directional labels for training:
   ```python
   def prepare_directional_labels(dataset):
       # Create binary masks for DOWN predictions
       down_masks = []
       for example in dataset:
           down_masks.append(example["direction"].upper() == "DOWN")
       return torch.tensor(down_masks, dtype=torch.bool)
   ```

### Training Configuration
1. Update `train_enhanced_directional_model.py`:
   ```python
   # Enhanced training arguments
   training_args = TrainingArguments(
       output_dir="output/down_enhanced_model",
       num_train_epochs=15,  # Increased from 10
       per_device_train_batch_size=4,
       gradient_accumulation_steps=4,
       learning_rate=2e-5,
       warmup_ratio=0.1,
       weight_decay=0.01,
       logging_steps=10,
       save_strategy="epoch",
       fp16=True,
   )
   
   # Create trainer with custom loss
   class DirectionalTrainer(Trainer):
       def __init__(self, *args, direction_masks=None, **kwargs):
           super().__init__(*args, **kwargs)
           self.direction_masks = direction_masks
           
       def compute_loss(self, model, inputs, return_outputs=False):
           labels = inputs.pop("labels")
           outputs = model(**inputs)
           logits = outputs.logits
           
           # Use custom loss function
           loss_fct = DirectionalWeightedLoss(down_weight=2.0, up_weight=1.0)
           loss = loss_fct(logits, labels, self.direction_masks)
           
           return (loss, outputs) if return_outputs else loss
   ```

## 3. Evaluation Framework Enhancements

### Improved Metrics
1. Add directional-specific evaluation:
   ```python
   def evaluate_directional_performance(predictions, actuals):
       # Calculate standard metrics
       accuracy, precision_up, recall_up, f1_up = calculate_metrics(predictions, actuals)
       
       # Add DOWN-specific metrics
       down_precision = precision_score(
           [1 if d == "DOWN" else 0 for d in actuals],
           [1 if d == "DOWN" else 0 for d in predictions],
           zero_division=0
       )
       
       down_recall = recall_score(
           [1 if d == "DOWN" else 0 for d in actuals],
           [1 if d == "DOWN" else 0 for d in predictions],
           zero_division=0
       )
       
       down_f1 = f1_score(
           [1 if d == "DOWN" else 0 for d in actuals],
           [1 if d == "DOWN" else 0 for d in predictions],
           zero_division=0
       )
       
       # Calculate DOWN-specific improvements
       relative_improvement = {
           "down_precision_relative": (down_precision - 0.33) / 0.33 * 100,
           "down_recall_relative": (down_recall - 0.20) / 0.20 * 100
       }
       
       return {
           "accuracy": accuracy,
           "up_precision": precision_up,
           "up_recall": recall_up,
           "up_f1": f1_up,
           "down_precision": down_precision,
           "down_recall": down_recall,
           "down_f1": down_f1,
           "relative_improvements": relative_improvement
       }
   ```

### Backtesting Framework
1. Implement extended historical backtesting:
   ```python
   def extended_backtest(model_path, tickers, start_date, end_date):
       """Test model across a longer time period to evaluate consistency."""
       results = []
       for ticker in tickers:
           ticker_results = []
           for test_date in generate_trading_dates(start_date, end_date):
               prediction = predict_for_date(model_path, ticker, test_date)
               actual = get_actual_movement(ticker, test_date)
               ticker_results.append({
                   "date": test_date,
                   "prediction": prediction,
                   "actual": actual,
                   "correct": prediction["direction"] == actual["direction"]
               })
           results.append({
               "ticker": ticker,
               "predictions": ticker_results,
               "accuracy": sum(r["correct"] for r in ticker_results) / len(ticker_results)
           })
       return results
   ```

## 4. Implementation Timeline

### Week 1: Dataset & Loss Function
- Day 1-2: Enhance dataset generator with bearish scenarios
- Day 3-4: Generate 3,000 examples with 60% DOWN bias
- Day 5-7: Implement and test custom directional loss function

### Week 2: Model Training
- Day 1-2: Set up training with enhanced hyperparameters 
- Day 3-7: Train model for 15 epochs with monitoring

### Week 3: Evaluation & Fine-tuning
- Day 1-3: Comprehensive evaluation across all energy stocks
- Day 4-5: Fine-tune model based on error analysis
- Day 6-7: Extended backtesting over 3-month period

## 5. Expected Improvements

### Current Baseline
- Directional Accuracy: 61.11%
- DOWN Precision: 0.33
- DOWN Recall: 0.20
- Average Percentage Error: 0.69%

### Target Metrics
- Directional Accuracy: 70-75%
- DOWN Precision: 0.60-0.70
- DOWN Recall: 0.50-0.60
- Average Percentage Error: <0.65%

## 6. Future Research Directions

1. **Contrastive Learning**: Implement techniques to better distinguish UP vs DOWN patterns
2. **Ensemble Methods**: Combine specialized UP and DOWN models 
3. **Adaptive Prompting**: Dynamically adjust prompts based on market conditions
4. **Alternative Architectures**: Experiment with different model architectures beyond transformer-based LLMs 