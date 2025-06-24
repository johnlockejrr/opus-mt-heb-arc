#!/usr/bin/env python3
"""
Script to fix inconsistent translations of Hebrew "אור" (light) to Aramaic.
This script identifies cases where "אור" should be translated vs. preserved.
"""

import pandas as pd
import re

def analyze_light_translations(corpus_file):
    """
    Analyze how "אור" (light) is translated in the corpus.
    """
    print("=== ANALYSIS: Hebrew 'אור' (light) Translations ===\n")
    
    df = pd.read_csv(corpus_file, sep='|', encoding='utf-8')
    
    # Find all instances of "אור" in Hebrew
    hebrew_light = df[df['Samaritan'].str.contains('אור', na=False)]
    print(f"Total instances of 'אור' in Hebrew: {len(hebrew_light)}")
    
    # Categorize the instances
    place_name_cases = []
    light_cases = []
    
    for idx, row in hebrew_light.iterrows():
        hebrew_text = row['Samaritan']
        aramaic_text = row['Targum']
        
        # Check if it's a place name (Ur of the Chaldees)
        if 'כשדים' in hebrew_text or 'Ur' in hebrew_text:
            place_name_cases.append((hebrew_text, aramaic_text))
        else:
            light_cases.append((hebrew_text, aramaic_text))
    
    print(f"Place name cases (should be preserved): {len(place_name_cases)}")
    print(f"Light cases (should be translated): {len(light_cases)}")
    
    return df, place_name_cases, light_cases

def examine_light_cases(light_cases):
    """
    Examine how "אור" as light is being translated.
    """
    print("\n=== EXAMINING LIGHT TRANSLATIONS ===\n")
    
    for i, (hebrew, aramaic) in enumerate(light_cases):
        print(f"Case {i+1}:")
        print(f"  Hebrew: {hebrew}")
        print(f"  Aramaic: {aramaic}")
        
        # Check if "אור" is preserved or translated
        if 'אור' in aramaic:
            print(f"  Status: ⚠️ PRESERVED (should be translated)")
        else:
            print(f"  Status: ✅ TRANSLATED")
        print()

def suggest_corrections(light_cases):
    """
    Suggest corrections for light translations.
    """
    print("=== SUGGESTED CORRECTIONS ===\n")
    
    # Hebrew→Aramaic light translations
    light_translations = {
        'אור': 'נהורא',      # light
        'האור': 'אורה',      # the light
        'לאור': 'לנהורא',    # to light
        'באור': 'בנהורא',    # in light
    }
    
    corrections = []
    
    for hebrew, aramaic in light_cases:
        corrected_aramaic = aramaic
        
        # Apply corrections
        for hebrew_word, aramaic_word in light_translations.items():
            if hebrew_word in hebrew and hebrew_word in aramaic:
                corrected_aramaic = corrected_aramaic.replace(hebrew_word, aramaic_word)
        
        if corrected_aramaic != aramaic:
            corrections.append({
                'hebrew': hebrew,
                'original_aramaic': aramaic,
                'corrected_aramaic': corrected_aramaic
            })
    
    print(f"Found {len(corrections)} cases that need correction:")
    for i, correction in enumerate(corrections):
        print(f"\nCorrection {i+1}:")
        print(f"  Hebrew: {correction['hebrew']}")
        print(f"  Original Aramaic: {correction['original_aramaic']}")
        print(f"  Corrected Aramaic: {correction['corrected_aramaic']}")
    
    return corrections

def create_improved_dataset(corpus_file, corrections):
    """
    Create an improved dataset with corrected light translations.
    """
    print("\n=== CREATING IMPROVED DATASET ===\n")
    
    df = pd.read_csv(corpus_file, sep='|', encoding='utf-8')
    improved_df = df.copy()
    
    # Apply corrections
    for correction in corrections:
        # Find the row with this Hebrew text
        mask = improved_df['Samaritan'] == correction['hebrew']
        if mask.any():
            improved_df.loc[mask, 'Targum'] = correction['corrected_aramaic']
    
    # Save improved dataset
    output_file = 'aligned_corpus_improved.tsv'
    improved_df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
    
    print(f"Improved dataset saved to: {output_file}")
    print(f"Applied {len(corrections)} corrections")
    
    return improved_df

def create_training_examples():
    """
    Create additional training examples for light translations.
    """
    print("\n=== ADDITIONAL TRAINING EXAMPLES ===\n")
    
    # Common light-related phrases
    additional_examples = [
        ("האור הגדול", "אורה רבא"),           # the great light
        ("אור העולם", "נהורא דעלמא"),         # light of the world
        ("אור השמש", "נהורא דשמשא"),         # light of the sun
        ("אור היום", "נהורא דיומא"),          # light of day
        ("אור האש", "נהורא דנורא"),          # light of fire
        ("אור הכוכבים", "נהורא דכוכביא"),    # light of the stars
        ("אור הבוקר", "נהורא דצפרה"),        # light of morning
        ("אור הערב", "נהורא דרמשא"),          # light of evening
    ]
    
    print("Suggested additional training examples:")
    for hebrew, aramaic in additional_examples:
        print(f"  {hebrew} → {aramaic}")
    
    return additional_examples

def main():
    """
    Main function to fix light translations.
    """
    corpus_file = 'aligned_corpus.tsv'
    
    try:
        # Analyze current translations
        df, place_name_cases, light_cases = analyze_light_translations(corpus_file)
        
        # Examine light cases
        examine_light_cases(light_cases)
        
        # Suggest corrections
        corrections = suggest_corrections(light_cases)
        
        # Create improved dataset
        if corrections:
            improved_df = create_improved_dataset(corpus_file, corrections)
        
        # Create additional training examples
        additional_examples = create_training_examples()
        
        print("\n=== SUMMARY ===")
        print("The issue with 'אור' (light) translations:")
        print("1. Inconsistent translation in original corpus")
        print("2. Model learned to sometimes preserve, sometimes translate")
        print("3. Need for better training data with consistent translations")
        
        print("\nRecommendations:")
        print("1. Use the improved dataset for retraining")
        print("2. Add the suggested additional examples")
        print("3. Consider post-processing rules for light translations")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 