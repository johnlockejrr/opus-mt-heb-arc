#!/usr/bin/env python3
"""
Dataset preparation script for Syriac-Aramaic translation model.
This script processes a TSV file with Syriac-Aramaic parallel texts and prepares it for training.
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import re
from sklearn.model_selection import train_test_split
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean and normalize text for training.
    """
    if pd.isna(text) or text == '':
        return None
    
    # Convert to string if not already
    text = str(text).strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text if text else None

def filter_valid_pairs(syriac_text, aramaic_text):
    """
    Filter out invalid translation pairs.
    """
    # Both texts must be non-empty and cleaned
    syriac_clean = clean_text(syriac_text)
    aramaic_clean = clean_text(aramaic_text)
    
    if syriac_clean is None or aramaic_clean is None:
        return False, None, None
    
    # Filter out very short or very long texts
    if len(syriac_clean) < 5 or len(aramaic_clean) < 5:
        return False, None, None
    
    if len(syriac_clean) > 1000 or len(aramaic_clean) > 1000:
        return False, None, None
    
    # Filter out texts that are too different in length (likely not good translations)
    length_ratio = len(syriac_clean) / len(aramaic_clean)
    if length_ratio < 0.3 or length_ratio > 3.0:
        return False, None, None
    
    return True, syriac_clean, aramaic_clean

def handle_duplicates(df, remove_duplicates=True):
    """
    Handle duplicate entries in the dataset.
    """
    logger.info(f"Original dataset size: {len(df)}")
    
    # Check for different types of duplicates
    total_duplicates = len(df) - len(df.drop_duplicates())
    syriac_duplicates = df['Syriac'].duplicated().sum()
    aramaic_duplicates = df['Aramaic'].duplicated().sum()
    verse_duplicates = df[['Book', 'Chapter', 'Verse']].duplicated().sum()
    
    logger.info(f"Duplicate analysis:")
    logger.info(f"  - Total row duplicates: {total_duplicates}")
    logger.info(f"  - Syriac text duplicates: {syriac_duplicates}")
    logger.info(f"  - Aramaic text duplicates: {aramaic_duplicates}")
    logger.info(f"  - Verse reference duplicates: {verse_duplicates}")
    
    if remove_duplicates:
        # Remove exact row duplicates first
        df_clean = df.drop_duplicates()
        logger.info(f"Removed {len(df) - len(df_clean)} exact row duplicates")
        
        # For verse reference duplicates, keep the first occurrence
        df_clean = df_clean.drop_duplicates(subset=['Book', 'Chapter', 'Verse'], keep='first')
        logger.info(f"Removed {len(df.drop_duplicates()) - len(df_clean)} verse reference duplicates")
        
        # For text duplicates, we keep them as they might be legitimate repetitions
        logger.info(f"Keeping text duplicates as they may be legitimate repetitions")
        
        logger.info(f"Final dataset size after duplicate removal: {len(df_clean)}")
        return df_clean
    else:
        # Just log the duplicates but keep them
        logger.info("Keeping all duplicates as requested")
        return df

def load_and_prepare_data(file_path, test_size=0.1, val_size=0.1, random_state=42, remove_duplicates=True):
    """
    Load the TSV file and prepare the dataset for training.
    """
    logger.info(f"Loading data from {file_path}")
    
    # Load the TSV file
    df = pd.read_csv(file_path, sep='|', encoding='utf-8')
    
    logger.info(f"Loaded {len(df)} rows from the dataset")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Verify we have the expected columns
    if 'Syriac' not in df.columns or 'Aramaic' not in df.columns:
        raise ValueError("Expected columns 'Syriac' and 'Aramaic' not found in the dataset")
    
    # Handle duplicates
    df = handle_duplicates(df, remove_duplicates=remove_duplicates)
    
    # Filter and clean the data
    valid_pairs = []
    filtered_out = 0
    
    for idx, row in df.iterrows():
        syriac_text = row['Syriac']
        aramaic_text = row['Aramaic']
        
        is_valid, syriac_clean, aramaic_clean = filter_valid_pairs(syriac_text, aramaic_text)
        
        if is_valid:
            valid_pairs.append({
                'syriac': syriac_clean,
                'aramaic': aramaic_clean,
                'book': row.get('Book', ''),
                'chapter': row.get('Chapter', ''),
                'verse': row.get('Verse', '')
            })
        else:
            filtered_out += 1
    
    logger.info(f"Filtered out {filtered_out} invalid pairs")
    logger.info(f"Kept {len(valid_pairs)} valid translation pairs")
    
    if len(valid_pairs) == 0:
        raise ValueError("No valid translation pairs found after filtering")
    
    # Convert to DataFrame for easier splitting
    df_clean = pd.DataFrame(valid_pairs)
    
    # Split the data
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df_clean, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_clean['book'] if len(df_clean['book'].unique()) > 1 else None
    )
    
    # Second split: separate validation set from training
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size/(1-test_size), 
        random_state=random_state,
        stratify=train_val_df['book'] if len(train_val_df['book'].unique()) > 1 else None
    )
    
    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict

def save_dataset(dataset_dict, output_dir):
    """
    Save the prepared dataset to disk.
    """
    logger.info(f"Saving dataset to {output_dir}")
    dataset_dict.save_to_disk(output_dir)
    
    # Also save some statistics
    stats = {
        'train_size': len(dataset_dict['train']),
        'validation_size': len(dataset_dict['validation']),
        'test_size': len(dataset_dict['test']),
        'total_size': len(dataset_dict['train']) + len(dataset_dict['validation']) + len(dataset_dict['test'])
    }
    
    logger.info("Dataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Prepare Syriac-Aramaic translation dataset')
    parser.add_argument('--input_file', required=True, 
                       help='Input TSV file path')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for the prepared dataset')
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion of data to use for validation')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--keep_duplicates', action='store_true',
                       help='Keep duplicate entries instead of removing them')
    
    args = parser.parse_args()
    
    try:
        # Load and prepare the dataset
        dataset_dict = load_and_prepare_data(
            args.input_file,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            remove_duplicates=not args.keep_duplicates
        )
        
        # Save the dataset
        stats = save_dataset(dataset_dict, args.output_dir)
        
        logger.info("Dataset preparation completed successfully!")
        logger.info(f"Dataset saved to: {args.output_dir}")
        
        # Print a few examples
        logger.info("\nSample training examples:")
        for i in range(min(3, len(dataset_dict['train']))):
            example = dataset_dict['train'][i]
            logger.info(f"Example {i+1}:")
            logger.info(f"  Syriac: {example['syriac'][:100]}...")
            logger.info(f"  Aramaic: {example['aramaic'][:100]}...")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error during dataset preparation: {e}")
        raise

if __name__ == "__main__":
    main() 