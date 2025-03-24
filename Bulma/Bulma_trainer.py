import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class DirectionalWeightedLoss(nn.Module):
    """
    Custom loss function that applies higher weights to DOWN examples.
    This helps the model learn better discrimination for DOWN predictions.
    """
    def __init__(self, down_weight=2.0, up_weight=1.0):
        super().__init__()
        self.down_weight = down_weight
        self.up_weight = up_weight
        self.base_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, labels, direction_masks):
        # Calculate base loss for each token
        base_loss = self.base_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Reshape direction masks to match loss shape
        # Each sample in the batch gets its own direction mask value
        batch_size = logits.size(0)
        seq_length = logits.size(1)
        
        # Ensure direction_masks is on the same device as base_loss
        direction_masks = direction_masks.to(base_loss.device)
        
        # Repeat the direction mask for each token in the sequence
        expanded_masks = direction_masks.repeat_interleave(seq_length)
        
        # Apply direction-based weights
        weighted_loss = torch.where(
            expanded_masks,  # TRUE for DOWN examples
            base_loss * self.down_weight,
            base_loss * self.up_weight
        )
        
        return weighted_loss.mean()

class DirectionalTrainer(Trainer):
    """
    Custom trainer that uses the DirectionalWeightedLoss
    """
    def __init__(self, direction_masks=None, down_weight=2.0, up_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction_masks = direction_masks
        self.down_weight = down_weight
        self.up_weight = up_weight
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get the batch indices for the current batch
        batch_indices = inputs.get("batch_indices", None)
        if batch_indices is None:
            # If batch indices not provided, assume sequential batching
            batch_size = labels.size(0)
            batch_indices = list(range(batch_size))
            
        # Get direction masks for this batch and move to the same device as logits
        batch_direction_masks = self.direction_masks[batch_indices].to(logits.device)
        
        # Use custom loss function
        loss_fct = DirectionalWeightedLoss(
            down_weight=self.down_weight,
            up_weight=self.up_weight
        )
        loss = loss_fct(logits, labels, batch_direction_masks)
        
        return (loss, outputs) if return_outputs else loss

def prepare_directional_labels(jsonl_file):
    """
    Extract direction labels from dataset file to create directional masks.
    """
    directions = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            directions.append(example["direction"].upper() == "DOWN")
    
    return torch.tensor(directions, dtype=torch.bool)

def load_and_prepare_dataset(dataset_path):
    """
    Load and prepare dataset for training.
    """
    # Load dataset from JSONL file
    logging.info(f"Loading dataset from {dataset_path}")
    
    # Check if we have the text version (preferred for training)
    text_path = dataset_path.replace('.jsonl', '_text.jsonl')
    if os.path.exists(text_path):
        dataset_path = text_path
        logging.info(f"Using text-formatted dataset: {text_path}")
    
    # Load as HF dataset
    hf_dataset = load_dataset('json', data_files=dataset_path)['train']
    
    logging.info(f"Loaded {len(hf_dataset)} examples")
    
    return hf_dataset

def process_dataset(hf_dataset, tokenizer, max_length=2048):
    """
    Process and tokenize the dataset for training.
    """
    # Define tokenization function
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "labels": outputs["input_ids"].copy(),
        }
    
    # Tokenize dataset
    logging.info("Tokenizing dataset")
    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing dataset",
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def setup_model_and_tokenizer(model_name):
    """
    Set up the model and tokenizer for training.
    """
    logging.info(f"Loading base model from {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Handle tokenizer quirks
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization for efficient training
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_peft_config(lora_r, lora_alpha):
    """
    Set up the PEFT (LoRA) configuration.
    """
    logging.info(f"Setting up LoRA with rank={lora_r}, alpha={lora_alpha}")
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
    )
    
    return peft_config

def train_model(model, tokenizer, training_args, tokenized_dataset, direction_masks, 
                down_weight=2.0, up_weight=1.0, lora_r=24, lora_alpha=48):
    """
    Train the model with directional weighted loss.
    """
    # Apply LoRA
    peft_config = setup_peft_config(lora_r, lora_alpha)
    model = get_peft_model(model, peft_config)
    
    # Create a custom trainer with directional weighted loss
    trainer = DirectionalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        direction_masks=direction_masks,
        down_weight=down_weight,
        up_weight=up_weight,
    )
    
    # Train the model
    logging.info("Starting training")
    trainer.train()
    
    # Save the trained model
    logging.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a directional-enhanced model with DOWN-weighted loss")
    
    # Dataset and output arguments
    parser.add_argument("--dataset", type=str, default="down_enhanced_dataset/down_enhanced_dataset.jsonl",
                        help="Path to the enhanced dataset JSONL file")
    parser.add_argument("--base_model", type=str, default="output/enhanced_directional_model",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="output/down_enhanced_model",
                        help="Output directory for the trained model")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    
    # Loss weighting arguments
    parser.add_argument("--down_weight", type=float, default=2.0,
                        help="Weight for DOWN examples in loss function")
    parser.add_argument("--up_weight", type=float, default=1.0,
                        help="Weight for UP examples in loss function")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Log training parameters
    logging.info(f"Starting training with the following parameters:")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Base model: {args.base_model}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Training epochs: {args.epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(f"LoRA rank: {args.lora_r}")
    logging.info(f"LoRA alpha: {args.lora_alpha}")
    logging.info(f"DOWN weight: {args.down_weight}")
    logging.info(f"UP weight: {args.up_weight}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Prepare directional labels for weighted loss
    direction_masks = prepare_directional_labels(args.dataset)
    logging.info(f"Extracted {direction_masks.sum().item()} DOWN examples out of {len(direction_masks)}")
    
    # Load dataset
    hf_dataset = load_and_prepare_dataset(args.dataset)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.base_model)
    
    # Process dataset
    tokenized_dataset = process_dataset(hf_dataset, tokenizer)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb reporting
    )
    
    # Train the model
    train_model(
        model, 
        tokenizer, 
        training_args, 
        tokenized_dataset, 
        direction_masks,
        down_weight=args.down_weight,
        up_weight=args.up_weight,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
if __name__ == "__main__":
    main() 