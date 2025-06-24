#!/usr/bin/env python3
"""
Training script for Hebrew-Aramaic translation model using MarianMT.
This script fine-tunes a pre-trained MarianMT model on the prepared dataset.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    set_seed
)
from evaluate import load
import wandb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HebrewAramaicTranslator:
    """
    Main class for training Hebrew-Aramaic translation model.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Set random seeds for reproducibility
        set_seed(config.get('seed', 42))
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def load_dataset(self, dataset_path: str) -> Dataset:
        """
        Load the prepared dataset from disk.
        """
        logger.info(f"Loading dataset from {dataset_path}")
        
        try:
            dataset = load_from_disk(dataset_path)
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  Train: {len(dataset['train'])} examples")
            logger.info(f"  Validation: {len(dataset['validation'])} examples")
            logger.info(f"  Test: {len(dataset['test'])} examples")
            
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def load_tokenizer_and_model(self, model_name: str):
        """
        Load the tokenizer and model from Hugging Face Hub.
        """
        logger.info(f"Loading tokenizer and model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add special tokens if needed
            special_tokens = {
                'additional_special_tokens': ['<he>', '<arc>']
            }
            self.tokenizer.add_special_tokens(special_tokens)
            
            # Load model - check if it's a MarianMT model or local model
            if 'marian' in model_name.lower() or 'opus-mt' in model_name.lower() or os.path.exists(model_name):
                # For MarianMT models or local models, don't use mean_resizing
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                # For other models, use mean_resizing to avoid warning
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, 
                    mean_resizing=False
                )
            
            # Resize token embeddings if we added new tokens
            if len(special_tokens['additional_special_tokens']) > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Vocabulary size: {len(self.tokenizer)}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_function(self, examples):
        """
        Preprocess the dataset for training.
        """
        # Determine source and target based on training direction
        if self.config.get('direction', 'he2arc') == 'he2arc':
            source_texts = examples['hebrew']
            target_texts = examples['aramaic']
            source_lang = 'he'
            target_lang = 'arc'
        else:  # arc2he
            source_texts = examples['aramaic']
            target_texts = examples['hebrew']
            source_lang = 'arc'
            target_lang = 'he'
        
        # Add language prefix if specified
        if self.config.get('use_language_prefix', True):
            inputs = [f"<{source_lang}> {text}" for text in source_texts]
        else:
            inputs = source_texts
        
        # Tokenize inputs and targets using the new API
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.get('max_input_length', 512),
            truncation=True,
            padding=False,  # Don't pad here, let the data collator handle it
            return_tensors=None  # Return lists, not tensors
        )
        
        # Tokenize targets using the new API
        labels = self.tokenizer(
            text_target=target_texts,
            max_length=self.config.get('max_target_length', 512),
            truncation=True,
            padding=False,  # Don't pad here, let the data collator handle it
            return_tensors=None  # Return lists, not tensors
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def compute_metrics(self, eval_preds):
        """
        Compute evaluation metrics.
        """
        predictions, labels = eval_preds
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Load BLEU metric
        metric = load("sacrebleu")
        
        # Compute BLEU score
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Also compute character-level accuracy for Hebrew/Aramaic
        char_accuracy = self.compute_char_accuracy(decoded_preds, decoded_labels)
        
        return {
            "bleu": result["score"],
            "char_accuracy": char_accuracy
        }
    
    def compute_char_accuracy(self, predictions, references):
        """
        Compute character-level accuracy for Hebrew/Aramaic text.
        """
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            # Normalize whitespace
            pred = ' '.join(pred.split())
            ref = ' '.join(ref.split())
            
            # Count characters
            total_chars += len(ref)
            
            # Count correct characters
            for i, (p_char, r_char) in enumerate(zip(pred, ref)):
                if i < len(ref) and p_char == r_char:
                    correct_chars += 1
        
        return (correct_chars / total_chars * 100) if total_chars > 0 else 0.0
    
    def setup_training(self, dataset):
        """
        Set up the training configuration and trainer.
        """
        logger.info("Setting up training configuration")
        
        # Preprocess the datasets
        logger.info("Preprocessing datasets...")
        train_dataset = dataset["train"].map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        val_dataset = dataset["validation"].map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["validation"].column_names
        )
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if self.config.get('use_fp16', True) else None,
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config['output_dir'],
            eval_strategy="steps",
            eval_steps=self.config.get('eval_steps', 500),
            save_strategy="steps",
            save_steps=self.config.get('save_steps', 500),
            learning_rate=self.config.get('learning_rate', 2e-5),
            per_device_train_batch_size=self.config.get('batch_size', 8),
            per_device_eval_batch_size=self.config.get('batch_size', 8),
            weight_decay=self.config.get('weight_decay', 0.01),
            save_total_limit=self.config.get('save_total_limit', 3),
            num_train_epochs=self.config.get('num_epochs', 3),
            predict_with_generate=True,
            fp16=self.config.get('use_fp16', True),
            dataloader_pin_memory=False,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            report_to="wandb" if self.config.get('use_wandb', False) else "none",
            logging_steps=self.config.get('logging_steps', 100),
            warmup_steps=self.config.get('warmup_steps', 500),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            remove_unused_columns=False,
            push_to_hub=False,
        )
        
        # Create trainer with improved early stopping
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.get('early_stopping_patience', 5),
                early_stopping_threshold=self.config.get('early_stopping_threshold', 0.1)
            )]
        )
        
        logger.info("Training setup completed")
    
    def train(self):
        """
        Start the training process.
        """
        logger.info("Starting training...")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config['output_dir'])
            
            # Log training results
            logger.info("Training completed successfully!")
            logger.info(f"Training loss: {train_result.training_loss}")
            
            # Save training metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, dataset):
        """
        Evaluate the trained model on test set.
        """
        logger.info("Evaluating model on test set...")
        
        try:
            # Preprocess the test dataset
            logger.info("Preprocessing test dataset...")
            test_dataset = dataset["test"].map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["test"].column_names
            )
            
            # Evaluate on test set
            test_results = self.trainer.evaluate(
                eval_dataset=test_dataset,
                metric_key_prefix="test"
            )
            
            logger.info("Test results:")
            for key, value in test_results.items():
                logger.info(f"  {key}: {value}")
            
            # Save test metrics
            self.trainer.log_metrics("test", test_results)
            self.trainer.save_metrics("test", test_results)
            
            return test_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def save_model_info(self):
        """
        Save model information and configuration.
        """
        model_info = {
            'model_name': self.config['model_name'],
            'direction': self.config.get('direction', 'he2arc'),
            'vocabulary_size': len(self.tokenizer),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_config': self.config
        }
        
        info_path = os.path.join(self.config['output_dir'], 'model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model information saved to {info_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Hebrew-Aramaic translation model')
    
    # Data arguments
    parser.add_argument('--dataset_path', default='./hebrew_aramaic_dataset',
                       help='Path to the prepared dataset')
    parser.add_argument('--output_dir', default='./hebrew_aramaic_model',
                       help='Output directory for the trained model')
    
    # Model arguments
    parser.add_argument('--model_name', 
                       default='Helsinki-NLP/opus-mt-mul-en',
                       help='Pre-trained model to fine-tune')
    parser.add_argument('--direction', choices=['he2arc', 'arc2he'], default='he2arc',
                       help='Translation direction')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--max_input_length', type=int, default=512,
                       help='Maximum input sequence length')
    parser.add_argument('--max_target_length', type=int, default=512,
                       help='Maximum target sequence length')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluation frequency in steps')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save frequency in steps')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Number of evaluations to wait before early stopping')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.1,
                       help='Minimum improvement threshold for early stopping')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_fp16', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases logging')
    parser.add_argument('--use_language_prefix', action='store_true', default=True,
                       help='Use language prefix in input')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip evaluation on test set')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="hebrew-aramaic-translation",
            config=vars(args)
        )
    
    # Create configuration
    config = vars(args)
    
    try:
        # Initialize translator
        translator = HebrewAramaicTranslator(config)
        
        # Load dataset
        dataset = translator.load_dataset(args.dataset_path)
        
        # Load model and tokenizer
        translator.load_tokenizer_and_model(args.model_name)
        
        # Setup training
        translator.setup_training(dataset)
        
        # Train the model
        train_result = translator.train()
        
        # Evaluate the model
        if not args.skip_evaluation:
            test_results = translator.evaluate(dataset)
        
        # Save model information
        translator.save_model_info()
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main() 