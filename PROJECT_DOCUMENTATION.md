# Hebrew-Aramaic Biblical Translation Project Documentation

## Project Overview

This project implements a neural machine translation system for translating Hebrew biblical verses to Aramaic using MarianMT models. The system has been trained on aligned Hebrew-Aramaic biblical text pairs and achieves significant translation quality improvements.

## Key Achievements

- **BLEU Score Improvement**: From ~11.7 to ~37.8 after continued training and fine-tuning
- **Model Architecture**: MarianMT-based translation model
- **Hardware Optimization**: Configured for GTX 3060 GPU
- **Training Features**: Early stopping, gradient accumulation, and learning rate scheduling

## Project Structure

```
sam-aram/
├── back/                          # Core training and inference scripts
│   ├── inference.py
│   ├── prepare_dataset.py
│   ├── run_pipeline.sh
│   └── train_translation_model.py
├── hebrew_aramaic_dataset/        # Processed dataset
├── hebrew_aramaic_model_early_stopping/  # Model with early stopping
├── hebrew_aramaic_model_final/    # Final trained model
├── aligned_corpus.tsv             # Raw aligned corpus
├── train_syriac_aramaic.py        # Syriac→Aramaic training script
├── prepare_syriac_aramaic_dataset.py  # Syriac dataset preparation
└── requirements.txt               # Python dependencies
```

## Training Configuration

### Model Parameters
- **Base Model**: MarianMT
- **Source Language**: Hebrew (`he` - ISO 639-1)
- **Target Language**: Aramaic (`arc` - ISO 639-3)
- **Batch Size**: Optimized for GTX 3060
- **Learning Rate**: Adaptive with scheduling
- **Early Stopping**: Implemented to prevent overfitting

### Language Codes
- **Hebrew**: `he` (ISO 639-1)
- **Aramaic**: `arc` (ISO 639-3)
- **Arabic**: `ar` (ISO 639-1) - not used in this project

### Training Scripts
- `train_translation_model.py`: Main Hebrew→Aramaic training
- `train_syriac_aramaic.py`: Syriac→Aramaic training (adapted)
- `train_with_early_stopping.sh`: Automated training pipeline

## Dataset Information

### Corpus Details
- **Format**: TSV with Hebrew-Aramaic verse pairs
- **Processing**: Tokenized and preprocessed for MarianMT
- **Duplicates**: Some verse reference duplicates identified (text duplicates are normal and beneficial)

### Dataset Preparation
- `prepare_dataset.py`: Main dataset preparation
- `prepare_syriac_aramaic_dataset.py`: Syriac dataset adaptation
- Handles tokenization, splitting, and format conversion

## Technical Insights

### Translation Quality Issues

#### 1. Inconsistent Word Translations
**Problem**: Words like "אור" (light) are inconsistently translated or preserved
**Root Cause**: Limited and inconsistent training data
**Solutions**:
- Data augmentation with more examples
- Manual correction of training data
- Post-processing rules

#### 2. Untranslated Proper Nouns
**Problem**: "בראשית" kept untranslated despite "ראשית" existing in corpus
**Root Cause**: "בראשית" absent from training corpus
**Solution**: Add training examples and retrain model

### Model Training Warnings
- Deprecated parameter warnings (normal for MarianMT)
- Tokenizer warnings (expected behavior)
- Dataset suitability confirmed as typical

### Language Code Standards
- **Corrected**: Updated from `ar` (Arabic) to `arc` (Aramaic) for proper ISO 639 compliance
- **Hebrew**: Uses `he` (ISO 639-1)
- **Aramaic**: Uses `arc` (ISO 639-3)
- **Existing models**: Continue to work despite previous incorrect codes

## Syriac-Aramaic Extension

### Adaptation Requirements
- Dataset preparation with Syriac source
- Model prefix adjustment (syriac→aramaic)
- Similar training configuration to Hebrew model

### Implementation
- `prepare_syriac_aramaic_dataset.py`: Adapted dataset preparation
- `train_syriac_aramaic.py`: Modified training script
- Maintains same architecture and optimization

## Inference and Usage

### Basic Inference
```python
from inference import translate_text
result = translate_text("Hebrew text here")
```

### Pipeline Automation
- `run_pipeline.sh`: Complete training and evaluation pipeline
- Includes dataset preparation, training, and testing

## Performance Metrics

### Training Results
- **Initial BLEU**: ~11.7
- **Final BLEU**: ~37.8
- **Training Steps**: Multiple checkpoints (500, 810, 2000, 2500, 2700)
- **Early Stopping**: Implemented at checkpoint 2700

### Model Checkpoints
- `checkpoint-500/`: Early training stage
- `checkpoint-810/`: Mid-training
- `checkpoint-2000/`, `checkpoint-2500/`, `checkpoint-2700/`: Advanced training

## Technical Challenges and Solutions

### 1. GPU Memory Optimization
- Configured batch sizes for GTX 3060
- Gradient accumulation for effective larger batches
- Memory-efficient training strategies

### 2. Data Quality Issues
- Identified and addressed corpus inconsistencies
- Implemented data validation and cleaning
- Added support for multiple source languages

### 3. Model Convergence
- Early stopping to prevent overfitting
- Learning rate scheduling for stable training
- Multiple checkpoint saving for model selection

## Future Improvements

### Data Enhancement
- Expand training corpus with more Hebrew-Aramaic pairs
- Add manual corrections for inconsistent translations
- Implement data augmentation techniques

### Model Optimization
- Experiment with different model architectures
- Fine-tune hyperparameters for better performance
- Implement ensemble methods

### Quality Assurance
- Develop automated quality metrics
- Create validation sets for specific translation challenges
- Implement post-processing rules for common issues

## Dependencies

Key Python packages:
- transformers
- datasets
- torch
- sentencepiece
- wandb (for experiment tracking)

See `requirements.txt` for complete dependency list.

## Usage Instructions

1. **Setup**: Install dependencies from `requirements.txt`
2. **Dataset Preparation**: Run `prepare_dataset.py`
3. **Training**: Execute `train_translation_model.py` or use `run_pipeline.sh`
4. **Inference**: Use `inference.py` for translation
5. **Evaluation**: Check model performance in training logs

## Notes

- The project demonstrates successful application of transformer-based NMT to biblical Hebrew-Aramaic translation
- Significant BLEU score improvements achieved through iterative training
- Model handles both Hebrew→Aramaic and Syriac→Aramaic translation tasks
- Early stopping and GPU optimization ensure efficient training
- Ongoing work focuses on improving translation consistency and quality 