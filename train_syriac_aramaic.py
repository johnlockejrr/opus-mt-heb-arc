#!/usr/bin/env python3
"""
Training script for Syriac-Aramaic translation model.
Adapted from the Hebrew-Aramaic training script.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
from transformers import (
    MarianMTModel, MarianTokenizer, MarianConfig,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyriacAramaicTranslator:
    """
    Syriac-Aramaic translation model trainer.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load dataset
        self.load_dataset()
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Set up training configuration
        self.setup_training_config()
        
    def load_dataset(self):
        """Load the Syriac-Aramaic dataset."""
        logger.info(f"Loading dataset from {self.config['dataset_path']}")
        
        try:
            dataset_dict = DatasetDict.load_from_disk(self.config['dataset_path'])
            self.train_dataset = dataset_dict['train']
            self.val_dataset = dataset_dict['validation']
            self.test_dataset = dataset_dict['test']
            
            logger.info("Dataset loaded successfully:")
            logger.info(f"  Train: {len(self.train_dataset)} examples")
            logger.info(f"  Validation: {len(self.val_dataset)} examples")
            logger.info(f"  Test: {len(self.test_dataset)} examples")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer."""
        model_name = self.config['model_name']
        logger.info(f"Loading tokenizer and model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            
            # Load model
            self.model = MarianMTModel.from_pretrained(model_name)
            
            # Add special tokens for Syriac and Aramaic
            special_tokens = {
                'additional_special_tokens': ['<syr>', '<arc>']
            }
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info("Model loaded successfully")
            logger.info(f"Vocabulary size: {len(self.tokenizer)}")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_function(self, examples):
        """Preprocess the dataset for training."""
        # Determine source and target based on training direction
        if self.config.get('direction', 'syr2arc') == 'syr2arc':
            source_texts = examples['syriac']
            target_texts = examples['aramaic']
            source_lang = 'syr'
            target_lang = 'arc'
        else:  # arc2syr
            source_texts = examples['aramaic']
            target_texts = examples['syriac']
            source_lang = 'arc'
            target_lang = 'syr'
        
        # Add language prefix if enabled
        if self.config.get('use_language_prefix', True):
            inputs = [f"<{source_lang}> {text}" for text in source_texts]
        else:
            inputs = source_texts
        
        # Tokenize inputs and targets
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config['max_input_length'],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            text_target=target_texts,
            max_length=self.config['max_target_length'],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def setup_training_config(self):
        """Set up the training configuration."""
        logger.info("Setting up training configuration")
        
        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        self.train_dataset = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        self.val_dataset = self.val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.val_dataset.column_names
        )
        
        # Set up training arguments
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.config['output_dir'],
            evaluation_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_steps=self.config['save_steps'],
            learning_rate=self.config['learning_rate'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.config['num_epochs'],
            predict_with_generate=True,
            fp16=self.config.get('use_fp16', False),
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu",
            greater_is_better=True,
            warmup_steps=self.config.get('warmup_steps', 500),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            logging_steps=100,
            report_to="wandb" if self.config.get('use_wandb', False) else None,
            seed=self.config.get('seed', 42),
        )
        
        # Set up evaluation metrics
        self.bleu_metric = evaluate.load("sacrebleu")
        self.char_accuracy_metric = evaluate.load("character")
        
        # Set up trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.get('early_stopping_patience', 5),
                early_stopping_threshold=self.config.get('early_stopping_threshold', 0.1)
            )] if not self.config.get('skip_evaluation', False) else None
        )
        
        logger.info("Training setup completed")
    
    def compute_metrics(self, eval_preds):
        """Compute evaluation metrics."""
        predictions, labels = eval_preds
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU score
        bleu_result = self.bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Compute character accuracy
        char_accuracy_result = self.char_accuracy_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels
        )
        
        return {
            "bleu": bleu_result["score"],
            "char_accuracy": char_accuracy_result["character_accuracy"] * 100
        }
    
    def train(self):
        """Train the model."""
        logger.info("Starting training...")
        
        try:
            self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            
            # Log final training loss
            train_loss = self.trainer.state.log_history[-1].get('train_loss', 'N/A')
            logger.info(f"Training completed successfully!")
            logger.info(f"Training loss: {train_loss}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(self):
        """Evaluate the model on test set."""
        if self.config.get('skip_evaluation', False):
            logger.info("Skipping evaluation as requested")
            return
        
        logger.info("Evaluating model on test set...")
        
        try:
            # Preprocess test dataset
            logger.info("Preprocessing test dataset...")
            self.test_dataset = self.test_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=self.test_dataset.column_names
            )
            
            # Evaluate
            results = self.trainer.evaluate(eval_dataset=self.test_dataset)
            
            logger.info("Test results:")
            for key, value in results.items():
                logger.info(f"  {key}: {value}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def save_model_info(self):
        """Save model information to a JSON file."""
        model_info = {
            "model_name": self.config['model_name'],
            "direction": self.config['direction'],
            "vocabulary_size": len(self.tokenizer),
            "model_parameters": self.model.num_parameters(),
            "training_config": self.config,
            "device": str(self.device),
            "dataset_sizes": {
                "train": len(self.train_dataset),
                "validation": len(self.val_dataset),
                "test": len(self.test_dataset)
            }
        }
        
        info_file = os.path.join(self.config['output_dir'], 'model_info.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model information saved to {info_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Syriac-Aramaic translation model')
    
    # Dataset and model arguments
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset directory')
    parser.add_argument('--output_dir', required=True, help='Output directory for the model')
    parser.add_argument('--model_name', default='Helsinki-NLP/opus-mt-mul-en', 
                       help='Pre-trained model to fine-tune')
    parser.add_argument('--direction', choices=['syr2arc', 'arc2syr'], default='syr2arc',
                       help='Translation direction')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_input_length', type=int, default=512, help='Maximum input length')
    parser.add_argument('--max_target_length', type=int, default=512, help='Maximum target length')
    
    # Evaluation arguments
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation steps')
    parser.add_argument('--save_steps', type=int, default=500, help='Save steps')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    
    # Early stopping arguments
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.1, help='Early stopping threshold')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16 training')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--use_language_prefix', action='store_true', default=True, 
                       help='Use language prefix in input')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation')
    
    args = parser.parse_args()
    
    # Convert args to config dictionary
    config = vars(args)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=2))
    
    try:
        # Create translator instance
        translator = SyriacAramaicTranslator(config)
        
        # Train the model
        success = translator.train()
        
        if success:
            # Evaluate the model
            translator.evaluate()
            
            # Save model information
            translator.save_model_info()
            
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Model saved to: {config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Error during training pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 