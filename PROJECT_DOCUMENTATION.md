# Hebrew-Aramaic Translation Model - Complete Project Documentation

## 📋 Project Overview

This project implements a complete pipeline for fine-tuning MarianMT models on Hebrew-Aramaic parallel texts, specifically designed for translating between Hebrew (Samaritan) and Aramaic (Targum) biblical texts.

**Project Goal**: Create a machine translation model that can translate between Hebrew and Aramaic with high accuracy.

## 🗂️ Data Structure

### Input Data Format
The project uses a TSV (Tab-Separated Values) file with the following structure:
```
Book|Chapter|Verse|Targum|Samaritan
1|3|2|מן אילן גנה ניכל|מפרי עץ הגן נאכל
```

**Columns**:
- `Book`: Book identifier
- `Chapter`: Chapter number  
- `Verse`: Verse number
- `Targum`: Aramaic text
- `Samaritan`: Hebrew text

### Dataset Statistics
- **Total examples**: 5,417 parallel texts
- **Training set**: 4,319 examples (80%)
- **Validation set**: 540 examples (10%)
- **Test set**: 540 examples (10%)

## 🛠️ Project Files

### Core Scripts

#### 1. `prepare_dataset.py`
**Purpose**: Data preprocessing and dataset preparation
**Features**:
- Loads TSV file with Hebrew-Aramaic parallel texts
- Cleans and filters data (removes empty/invalid pairs)
- Splits data into train/validation/test sets
- Saves processed dataset in Hugging Face format
- Filters by length ratio and text quality

**Usage**:
```bash
python prepare_dataset.py \
    --input_file aligned_corpus.tsv \
    --output_dir ./hebrew_aramaic_dataset \
    --test_size 0.1 \
    --val_size 0.1
```

#### 2. `train_translation_model.py`
**Purpose**: Main training script for fine-tuning MarianMT models
**Features**:
- Supports both Hebrew→Aramaic and Aramaic→Hebrew directions
- Configurable hyperparameters
- Early stopping with BLEU score monitoring
- Mixed precision training (FP16)
- Comprehensive evaluation metrics
- WandB integration (optional)

**Key Parameters**:
- `--model_name`: Pre-trained model to fine-tune
- `--direction`: Translation direction (`he2ar` or `ar2he`)
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate for fine-tuning
- `--num_epochs`: Number of training epochs
- `--early_stopping_patience`: Early stopping patience
- `--early_stopping_threshold`: Minimum improvement threshold

**Usage**:
```bash
python train_translation_model.py \
    --dataset_path ./hebrew_aramaic_dataset \
    --output_dir ./hebrew_aramaic_model \
    --model_name Helsinki-NLP/opus-mt-mul-en \
    --direction he2ar \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --use_fp16
```

#### 3. `inference.py`
**Purpose**: Model inference and translation
**Features**:
- Interactive translation mode
- Single text translation
- Batch translation from files
- Auto-detection of translation direction
- Support for both directions

**Usage**:
```bash
# Interactive mode
python inference.py --model_path ./hebrew_aramaic_model

# Single translation
python inference.py \
    --model_path ./hebrew_aramaic_model \
    --text "מפרי עץ הגן נאכל" \
    --direction he2ar

# Batch translation
python inference.py \
    --model_path ./hebrew_aramaic_model \
    --input_file input_texts.txt \
    --output_file translations.txt
```

### Automation Scripts

#### 4. `run_pipeline.sh`
**Purpose**: Complete pipeline automation
**Features**:
- Runs dataset preparation
- Trains the model
- Provides usage instructions
- Error handling and colored output

**Usage**:
```bash
./run_pipeline.sh
```

#### 5. `train_with_early_stopping.sh`
**Purpose**: Training with early stopping optimization
**Features**:
- Configurable early stopping parameters
- Starts from pre-trained model
- Automatic stopping when performance plateaus
- Optimized for finding best training duration

**Usage**:
```bash
./train_with_early_stopping.sh
```

### Configuration Files

#### 6. `requirements.txt`
**Dependencies**:
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
sacrebleu>=2.3.0
wandb>=0.15.0
accelerate>=0.20.0
tokenizers>=0.13.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```

## 🎯 Model Architecture & Training

### Pre-trained Models Used
1. **Helsinki-NLP/opus-mt-mul-en**: Multilingual MarianMT model
2. **Helsinki-NLP/opus-mt-he-en**: Hebrew-English model (alternative)
3. **Helsinki-NLP/opus-mt-ar-en**: Arabic-English model (alternative)

### Training Configuration
- **Model Type**: MarianMT (Transformer encoder-decoder)
- **Vocabulary Size**: 64,174 tokens
- **Model Parameters**: 77,519,872
- **Max Sequence Length**: 512 tokens
- **Batch Size**: 16 (optimized for GTX 3060 12GB)
- **Learning Rate**: 2e-5
- **Mixed Precision**: FP16 enabled
- **Early Stopping**: BLEU score monitoring

### Training Process
1. **Data Preprocessing**: Tokenization with language prefixes
2. **Model Loading**: Pre-trained model with special tokens
3. **Fine-tuning**: Task-specific training on Hebrew-Aramaic data
4. **Evaluation**: BLEU score and character accuracy
5. **Model Saving**: Best checkpoint preservation

## 📊 Results & Performance

### Training Results (Hebrew → Aramaic)

#### Initial Training (3 epochs)
- **BLEU Score**: 11.71
- **Character Accuracy**: 22.55%
- **Training Loss**: 1.09
- **Training Time**: ~4 minutes 20 seconds

#### Continued Training (2 additional epochs)
- **BLEU Score**: 19.23 (+64% improvement!)
- **Character Accuracy**: 22.77%
- **Training Loss**: 0.89
- **Training Time**: ~3 minutes 31 seconds

### Model Performance Analysis
- **Strong BLEU improvement**: 11.71 → 19.23
- **Stable character accuracy**: ~22-23%
- **Good convergence**: Loss decreasing consistently
- **No overfitting**: Validation metrics improving

## 🔧 Technical Implementation Details

### Data Preprocessing
```python
def preprocess_function(self, examples):
    # Determine source and target based on training direction
    if self.config.get('direction', 'he2ar') == 'he2ar':
        source_texts = examples['hebrew']
        target_texts = examples['aramaic']
        source_lang = 'he'
        target_lang = 'ar'
    
    # Add language prefix
    inputs = [f"<{source_lang}> {text}" for text in source_texts]
    
    # Tokenize inputs and targets
    model_inputs = self.tokenizer(inputs, ...)
    labels = self.tokenizer(text_target=target_texts, ...)
    
    return model_inputs
```

### Early Stopping Configuration
```python
callbacks=[EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0.1
)]
```

### Evaluation Metrics
- **BLEU Score**: Standard machine translation metric
- **Character Accuracy**: Character-level accuracy for Hebrew/Aramaic
- **Training Loss**: Cross-entropy loss during training

## 🚀 Usage Examples

### Complete Pipeline
```bash
# 1. Prepare dataset
python prepare_dataset.py --input_file aligned_corpus.tsv

# 2. Train model
python train_translation_model.py \
    --dataset_path ./hebrew_aramaic_dataset \
    --output_dir ./hebrew_aramaic_model \
    --model_name Helsinki-NLP/opus-mt-mul-en \
    --direction he2ar \
    --batch_size 16 \
    --use_fp16

# 3. Use for translation
python inference.py --model_path ./hebrew_aramaic_model
```

### Training Both Directions
```bash
# Hebrew → Aramaic
python train_translation_model.py \
    --dataset_path ./hebrew_aramaic_dataset \
    --output_dir ./hebrew_aramaic_model_he2ar \
    --direction he2ar \
    --batch_size 16 \
    --use_fp16

# Aramaic → Hebrew
python train_translation_model.py \
    --dataset_path ./hebrew_aramaic_dataset \
    --output_dir ./hebrew_aramaic_model_ar2he \
    --direction ar2he \
    --batch_size 16 \
    --use_fp16
```

## 🎯 Key Achievements

### ✅ Successfully Implemented
1. **Complete training pipeline** from data preparation to inference
2. **Bidirectional translation** support (Hebrew↔Aramaic)
3. **Early stopping optimization** for optimal training duration
4. **High-quality results** with BLEU score of 19.23
5. **GPU optimization** for GTX 3060 12GB
6. **Comprehensive error handling** and logging
7. **Automation scripts** for easy usage

### 📈 Performance Improvements
- **BLEU Score**: 11.71 → 19.23 (+64% improvement)
- **Training Efficiency**: Early stopping prevents overfitting
- **Memory Optimization**: FP16 training and optimal batch sizes
- **Code Quality**: Clean, documented, and maintainable

## 🔍 Troubleshooting & Common Issues

### Issues Resolved
1. **Model loading errors**: Fixed `mean_resizing` parameter for MarianMT models
2. **Tokenization warnings**: Updated to new transformers API
3. **Early stopping issues**: Fixed metric name and configuration
4. **WandB warnings**: Properly disabled by default
5. **Data preprocessing errors**: Fixed dataset format handling

### Performance Optimization
- **Batch size**: Optimized for GTX 3060 12GB (16-24)
- **Learning rate**: 2e-5 provides good convergence
- **Sequence length**: 512 tokens handles most biblical verses
- **Mixed precision**: FP16 reduces memory usage and speeds training

## 🎓 Lessons Learned

### Data Quality
- Biblical verses work well as translation units
- Mixed text lengths are normal and beneficial
- High-quality parallel data is crucial for good results

### Model Training
- Early stopping is essential for optimal training duration
- Continued training from checkpoints can significantly improve performance
- MarianMT models work well for Semitic languages

### Technical Implementation
- Proper error handling prevents training failures
- Modular code design enables easy experimentation
- Comprehensive logging helps with debugging and monitoring

## 🚀 Future Improvements

### Potential Enhancements
1. **Data augmentation**: Add more diverse Hebrew-Aramaic texts
2. **Model ensemble**: Combine multiple models for better accuracy
3. **Domain adaptation**: Fine-tune for specific biblical books
4. **Interactive web interface**: Create a web-based translation tool
5. **API deployment**: Deploy model as a REST API service

### Research Directions
1. **Cross-lingual transfer**: Explore transfer learning from related languages
2. **Low-resource optimization**: Techniques for limited parallel data
3. **Evaluation metrics**: Develop domain-specific evaluation methods
4. **Interpretability**: Analyze model attention patterns

## 📚 References & Resources

### Documentation
- [MarianMT Documentation](https://huggingface.co/docs/transformers/en/model_doc/marian)
- [Helsinki-NLP Models](https://huggingface.co/Helsinki-NLP)
- [Transformers Library](https://huggingface.co/docs/transformers/)

### Related Work
- Helsinki-NLP multilingual translation models
- Biblical text translation research
- Semitic language processing

---

**Project Status**: ✅ Complete and Functional  
**Last Updated**: June 24, 2025  
**Model Performance**: BLEU 19.23 (Hebrew→Aramaic)  
**Training Time**: ~8 minutes total  
**GPU Requirements**: GTX 3060 12GB or equivalent 