# Hebrew-Aramaic Translation Model

This project provides a complete pipeline for fine-tuning MarianMT models on Hebrew-Aramaic parallel texts, specifically designed for translating between Hebrew (Samaritan) and Aramaic (Targum) texts.

## Overview

The pipeline consists of:
1. **Dataset Preparation** (`prepare_dataset.py`) - Processes the aligned corpus and splits it into train/validation/test sets
2. **Model Training** (`train_translation_model.py`) - Fine-tunes a pre-trained MarianMT model
3. **Inference** (`inference.py`) - Provides translation functionality using the trained model

## Data Format

The input data should be in TSV format with the following columns:
- `Book` - Book identifier
- `Chapter` - Chapter number
- `Verse` - Verse number
- `Targum` - Aramaic text
- `Samaritan` - Hebrew text

Example:
```
Book|Chapter|Verse|Targum|Samaritan
1|3|2|מן אילן גנה ניכל|מפרי עץ הגן נאכל
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd sam-aram
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install CUDA for GPU acceleration if available.

## Usage

### Step 1: Prepare the Dataset

First, prepare your aligned corpus for training:

```bash
python prepare_dataset.py \
    --input_file aligned_corpus.tsv \
    --output_dir ./hebrew_aramaic_dataset \
    --test_size 0.1 \
    --val_size 0.1
```

This will:
- Load the TSV file
- Clean and filter the data
- Split into train/validation/test sets
- Save the processed dataset

### Step 2: Train the Model

Train a translation model using the prepared dataset:

```bash
python train_translation_model.py \
    --dataset_path ./hebrew_aramaic_dataset \
    --output_dir ./hebrew_aramaic_model \
    --model_name Helsinki-NLP/opus-mt-mul-en \
    --direction he2ar \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --use_fp16
```

#### Key Parameters:

- `--model_name`: Pre-trained model to fine-tune. Recommended options:
  - `Helsinki-NLP/opus-mt-mul-en` (multilingual)
  - `Helsinki-NLP/opus-mt-he-en` (Hebrew-English)
  - `Helsinki-NLP/opus-mt-ar-en` (Arabic-English)
- `--direction`: Translation direction (`he2ar` or `ar2he`)
- `--batch_size`: Training batch size (adjust based on GPU memory)
- `--learning_rate`: Learning rate for fine-tuning
- `--num_epochs`: Number of training epochs
- `--use_fp16`: Enable mixed precision training (faster, less memory)

#### Training with Weights & Biases (Optional):

```bash
python train_translation_model.py \
    --dataset_path ./hebrew_aramaic_dataset \
    --output_dir ./hebrew_aramaic_model \
    --model_name Helsinki-NLP/opus-mt-mul-en \
    --use_wandb
```

### Step 3: Use the Trained Model

#### Interactive Translation:

```bash
python inference.py --model_path ./hebrew_aramaic_model
```

#### Translate a Single Text:

```bash
python inference.py \
    --model_path ./hebrew_aramaic_model \
    --text "מפרי עץ הגן נאכל" \
    --direction he2ar
```

#### Batch Translation:

```bash
python inference.py \
    --model_path ./hebrew_aramaic_model \
    --input_file input_texts.txt \
    --output_file translations.txt \
    --direction he2ar
```

## Model Recommendations

Based on the information in `info.txt`, here are recommended pre-trained models for Hebrew-Aramaic translation:

### 1. Multilingual Models
- `Helsinki-NLP/opus-mt-mul-en` - Good starting point for multilingual fine-tuning
- `facebook/m2m100_1.2B` - Large multilingual model with Hebrew and Aramaic support

### 2. Hebrew-Related Models
- `Helsinki-NLP/opus-mt-he-en` - Hebrew to English (can be adapted)
- `Helsinki-NLP/opus-mt-heb-ara` - Hebrew to Arabic (Semitic language family)

### 3. Arabic-Related Models
- `Helsinki-NLP/opus-mt-ar-en` - Arabic to English (Aramaic is related to Arabic)
- `Helsinki-NLP/opus-mt-ar-heb` - Arabic to Hebrew

## Training Tips

### 1. Data Quality
- Ensure your parallel texts are properly aligned
- Clean the data to remove noise and inconsistencies
- Consider the length ratio between source and target texts

### 2. Model Selection
- Start with a multilingual model if available
- Consider the vocabulary overlap between your languages
- Test different pre-trained models to find the best starting point

### 3. Hyperparameter Tuning
- Use smaller batch sizes for limited GPU memory
- Start with a lower learning rate (1e-5 to 5e-5)
- Increase epochs if the model hasn't converged
- Use early stopping to prevent overfitting

### 4. Evaluation
- Monitor BLEU score during training
- Use character-level accuracy for Hebrew/Aramaic
- Test on a held-out test set

## File Structure

```
sam-aram/
├── aligned_corpus.tsv          # Input parallel corpus
├── prepare_dataset.py          # Dataset preparation script
├── train_translation_model.py  # Training script
├── inference.py               # Inference script
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── info.txt                  # Project information
├── hebrew_aramaic_dataset/   # Prepared dataset (created)
└── hebrew_aramaic_model/     # Trained model (created)
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Poor Translation Quality**: 
   - Check data quality and alignment
   - Try different pre-trained models
   - Increase training epochs
   - Adjust learning rate

3. **Tokenization Issues**: 
   - Ensure the tokenizer supports Hebrew/Aramaic scripts
   - Check for proper UTF-8 encoding

4. **Training Instability**:
   - Reduce learning rate
   - Increase warmup steps
   - Use gradient clipping

### Performance Optimization:

- Use mixed precision training (`--use_fp16`)
- Enable gradient accumulation for larger effective batch sizes
- Use multiple GPUs if available
- Consider model quantization for inference

## Evaluation Metrics

The training script computes:
- **BLEU Score**: Standard machine translation metric
- **Character Accuracy**: Character-level accuracy for Hebrew/Aramaic text

## Contributing

To improve the pipeline:
1. Test with different pre-trained models
2. Experiment with different data preprocessing techniques
3. Add more evaluation metrics
4. Optimize for specific use cases

## License

This project is provided as-is for research and educational purposes.

## References

- [MarianMT Documentation](https://huggingface.co/docs/transformers/en/model_doc/marian)
- [Helsinki-NLP Models](https://huggingface.co/Helsinki-NLP)
- [Transformers Library](https://huggingface.co/docs/transformers/)

## Language Codes

- **Hebrew**: `he` (ISO 639-1)
- **Aramaic**: `arc` (ISO 639-3)
- **Syriac**: `syr` (ISO 639-2)
- **Arabic**: `ar` (ISO 639-1, not used here)

> **Note:** Previous versions used `ar` for Aramaic, which is the code for Arabic. All scripts and models now use the correct `arc` code for Aramaic for standards compliance.

## Supported Translation Directions

- Hebrew (`he`) → Aramaic (`arc`)
- Aramaic (`arc`) → Hebrew (`he`)
- Syriac (`syr`) → Aramaic (`arc`) *(see Syriac-Aramaic extension)*

## Syriac-Aramaic Extension

This project now includes scripts and documentation for training a Syriac→Aramaic translation model. See `SYRIAC_ARAMAIC_EXTENSION.md` for details. 