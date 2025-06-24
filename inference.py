#!/usr/bin/env python3
"""
Inference script for Hebrew-Aramaic translation model.
This script loads a trained model and provides translation functionality.
"""

import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HebrewAramaicTranslator:
    """
    Inference class for Hebrew-Aramaic translation.
    """
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load model info if available
        self.model_info = self.load_model_info()
        
        logger.info("Model loaded successfully")
    
    def load_model_info(self):
        """
        Load model information from the saved JSON file.
        """
        try:
            info_path = f"{self.model_path}/model_info.json"
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Model info file not found")
            return {}
    
    def translate(self, text: str, direction: str = None, max_length: int = 512):
        """
        Translate text from Hebrew to Aramaic or vice versa.
        
        Args:
            text: Input text to translate
            direction: Translation direction ('he2ar' or 'ar2he'). If None, auto-detect
            max_length: Maximum length of generated translation
        
        Returns:
            Translated text
        """
        # Auto-detect direction if not specified
        if direction is None:
            direction = self.model_info.get('direction', 'he2ar')
        
        # Add language prefix if the model was trained with it
        if self.model_info.get('training_config', {}).get('use_language_prefix', True):
            if direction == 'he2arc':
                input_text = f"<he> {text}"
            else:  # arc2he
                input_text = f"<arc> {text}"
        else:
            input_text = text
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode output
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation
    
    def batch_translate(self, texts: list, direction: str = None, max_length: int = 512):
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            direction: Translation direction
            max_length: Maximum length of generated translation
        
        Returns:
            List of translated texts
        """
        translations = []
        
        for text in texts:
            translation = self.translate(text, direction, max_length)
            translations.append(translation)
        
        return translations

def main():
    parser = argparse.ArgumentParser(description='Hebrew-Aramaic translation inference')
    parser.add_argument('--model_path', required=True,
                       help='Path to the trained model')
    parser.add_argument('--text', 
                       help='Text to translate')
    parser.add_argument('--input_file',
                       help='Input file with texts to translate (one per line)')
    parser.add_argument('--output_file',
                       help='Output file for translations')
    parser.add_argument('--direction', choices=['he2ar', 'ar2he'],
                       help='Translation direction (auto-detect if not specified)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum length of generated translation')
    parser.add_argument('--device',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = HebrewAramaicTranslator(args.model_path, args.device)
    
    # Print model info
    if translator.model_info:
        logger.info("Model information:")
        for key, value in translator.model_info.items():
            if key != 'training_config':
                logger.info(f"  {key}: {value}")
    
    # Handle single text translation
    if args.text:
        logger.info(f"Translating: {args.text}")
        translation = translator.translate(args.text, args.direction, args.max_length)
        print(f"Translation: {translation}")
    
    # Handle batch translation from file
    elif args.input_file:
        logger.info(f"Translating texts from {args.input_file}")
        
        # Read input texts
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        # Translate
        translations = translator.batch_translate(texts, args.direction, args.max_length)
        
        # Write output
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for text, translation in zip(texts, translations):
                    f.write(f"Source: {text}\n")
                    f.write(f"Translation: {translation}\n")
                    f.write("-" * 50 + "\n")
            logger.info(f"Translations saved to {args.output_file}")
        else:
            # Print to console
            for text, translation in zip(texts, translations):
                print(f"Source: {text}")
                print(f"Translation: {translation}")
                print("-" * 50)
    
    # Interactive mode
    else:
        logger.info("Entering interactive mode. Type 'quit' to exit.")
        logger.info(f"Default direction: {translator.model_info.get('direction', 'he2ar')}")
        
        while True:
            try:
                text = input("\nEnter text to translate: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if text:
                    translation = translator.translate(text, args.direction, args.max_length)
                    print(f"Translation: {translation}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Translation error: {e}")

if __name__ == "__main__":
    main() 