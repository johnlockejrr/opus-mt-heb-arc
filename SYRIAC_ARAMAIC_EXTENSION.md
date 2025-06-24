# Syriac-Aramaic Translation Extension

## Overview

This extension adapts the Hebrew-Aramaic translation model to handle Syriac→Aramaic translation tasks. The implementation maintains the same architecture and training approach while adjusting for the different source language.

## Implementation Files

### Core Scripts

#### `prepare_syriac_aramaic_dataset.py`
**Purpose**: Dataset preparation specifically for Syriac-Aramaic pairs
**Key Features**:
- Processes Syriac source texts instead of Hebrew
- Maintains same Aramaic target format
- Uses appropriate language prefixes (`syr→arc`)
- Same data splitting and validation logic as Hebrew version

**Usage**:
```bash
python prepare_syriac_aramaic_dataset.py \
    --input_file syriac_aramaic_corpus.tsv \
    --output_dir ./syriac_aramaic_dataset \
    --test_size 0.1 \
    --val_size 0.1
```

#### `train_syriac_aramaic.py`
**Purpose**: Training script adapted for Syriac→Aramaic translation
**Key Adaptations**:
- Model prefix: `syr→arc` instead of `hebrew→aramaic`
- Source language handling: Syriac text processing
- Same training configuration as Hebrew model
- Compatible with existing MarianMT architecture

**Usage**:
```bash
python train_syriac_aramaic.py \
    --dataset_path ./syriac_aramaic_dataset \
    --output_dir ./syriac_aramaic_model \
    --model_name Helsinki-NLP/opus-mt-mul-en \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --use_fp16
```

## Technical Adaptations

### Language Prefix Changes
- **Hebrew Model**: Uses `hebrew→aramaic` prefix
- **Syriac Model**: Uses `syr→arc` prefix (correct ISO codes)
- **Tokenization**: Same MarianMT tokenizer, different source language handling

### Language Codes
- **Syriac**: `syr` (ISO 639-2)
- **Aramaic**: `arc` (ISO 639-3)
- **Arabic**: `ar` (ISO 639-1) - not used in this project

### Dataset Requirements
- **Format**: TSV with Syriac-Aramaic parallel texts
- **Columns**: Similar structure to Hebrew corpus
- **Processing**: Same cleaning and validation steps
- **Splitting**: Identical train/validation/test ratios

### Model Configuration
- **Base Model**: Same MarianMT architecture
- **Training Parameters**: Identical to Hebrew model
- **Early Stopping**: Same implementation
- **Evaluation Metrics**: BLEU score and character accuracy

## Training Process

### 1. Dataset Preparation
```python
# Adapted preprocessing for Syriac source
if self.config.get('direction', 'syr2arc') == 'syr2arc':
    source_texts = examples['syriac']
    target_texts = examples['aramaic']
    source_lang = 'syr'
    target_lang = 'arc'
```

### 2. Model Training
- Same hyperparameters as Hebrew model
- Compatible with existing training infrastructure
- Can use same GPU optimization settings
- Early stopping and checkpoint saving

### 3. Inference
- Modified inference script for Syriac input
- Same translation quality metrics
- Compatible with existing evaluation pipeline

## Expected Performance

### Similarities to Hebrew Model
- **Architecture**: Identical MarianMT setup
- **Training Process**: Same optimization strategies
- **Memory Requirements**: Compatible with GTX 3060
- **Convergence**: Expected similar training patterns

### Potential Differences
- **Vocabulary**: Different source language vocabulary
- **Translation Patterns**: Syriac-specific linguistic features
- **Data Quality**: Dependent on Syriac-Aramaic corpus quality
- **Performance**: May vary based on corpus size and quality

## Integration with Main Project

### File Organization
```
sam-aram/
├── train_syriac_aramaic.py              # Syriac training script
├── prepare_syriac_aramaic_dataset.py    # Syriac dataset preparation
├── syriac_aramaic_dataset/              # Processed Syriac dataset
└── syriac_aramaic_model/                # Trained Syriac model
```

### Shared Resources
- **Requirements**: Same Python dependencies
- **Infrastructure**: Compatible with existing pipeline
- **Evaluation**: Same metrics and tools
- **Documentation**: Extends main project documentation

## Usage Workflow

### Complete Syriac-Aramaic Pipeline
```bash
# 1. Prepare Syriac dataset
python prepare_syriac_aramaic_dataset.py \
    --input_file syriac_aramaic_corpus.tsv \
    --output_dir ./syriac_aramaic_dataset

# 2. Train Syriac model
python train_syriac_aramaic.py \
    --dataset_path ./syriac_aramaic_dataset \
    --output_dir ./syriac_aramaic_model \
    --model_name Helsinki-NLP/opus-mt-mul-en \
    --batch_size 16 \
    --use_fp16

# 3. Use for translation
python inference.py \
    --model_path ./syriac_aramaic_model \
    --text "Syriac text here" \
    --direction syr2arc
```

## Advantages of This Approach

### 1. Code Reusability
- Minimal code changes required
- Same training infrastructure
- Compatible evaluation methods

### 2. Consistent Quality
- Same model architecture ensures reliability
- Proven training strategies
- Established optimization techniques

### 3. Easy Maintenance
- Single codebase for multiple language pairs
- Shared utilities and helpers
- Unified documentation and testing

## Future Extensions

### Additional Language Pairs
- Same approach can be applied to other Semitic languages
- Easy adaptation for new source languages
- Scalable architecture for multiple models

### Model Improvements
- Shared improvements benefit all language pairs
- Consistent optimization across models
- Unified evaluation and comparison

## Notes

- The Syriac-Aramaic extension demonstrates the flexibility of the original architecture
- Minimal code changes required for new language pair adaptation
- Maintains same high-quality training and evaluation standards
- Provides foundation for additional Semitic language translation models
- Uses correct ISO 639 language codes: `syr` for Syriac, `arc` for Aramaic 