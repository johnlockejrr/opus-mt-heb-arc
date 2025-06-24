#!/bin/bash

# Hebrew-Aramaic Translation Pipeline
# This script runs the complete pipeline from data preparation to training

set -e  # Exit on any error

echo "=== Hebrew-Aramaic Translation Pipeline ==="
echo ""

# Configuration
INPUT_FILE="aligned_corpus.tsv"
DATASET_DIR="./hebrew_aramaic_dataset"
MODEL_DIR="./hebrew_aramaic_model"
MODEL_NAME="Helsinki-NLP/opus-mt-mul-en"
DIRECTION="he2ar"
BATCH_SIZE=24  # Increased for GTX 3060 12GB
LEARNING_RATE=2e-5
NUM_EPOCHS=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file $INPUT_FILE not found!"
    exit 1
fi

print_status "Starting pipeline with configuration:"
echo "  Input file: $INPUT_FILE"
echo "  Dataset directory: $DATASET_DIR"
echo "  Model directory: $MODEL_DIR"
echo "  Model name: $MODEL_NAME"
echo "  Direction: $DIRECTION"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo ""

# Step 1: Prepare Dataset
print_status "Step 1: Preparing dataset..."
python prepare_dataset.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$DATASET_DIR" \
    --test_size 0.1 \
    --val_size 0.1

if [ $? -ne 0 ]; then
    print_error "Dataset preparation failed!"
    exit 1
fi

print_status "Dataset preparation completed successfully!"
echo ""

# Step 2: Train Model
print_status "Step 2: Training translation model..."
python train_translation_model.py \
    --dataset_path "$DATASET_DIR" \
    --output_dir "$MODEL_DIR" \
    --model_name "$MODEL_NAME" \
    --direction "$DIRECTION" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --use_fp16

if [ $? -ne 0 ]; then
    print_error "Model training failed!"
    exit 1
fi

print_status "Model training completed successfully!"
echo ""

# Step 3: Test the model
print_status "Step 3: Testing the trained model..."
echo "You can now use the model for translation:"
echo ""
echo "Interactive mode:"
echo "  python inference.py --model_path $MODEL_DIR"
echo ""
echo "Single translation:"
echo "  python inference.py --model_path $MODEL_DIR --text 'מפרי עץ הגן נאכל' --direction he2ar"
echo ""

print_status "Pipeline completed successfully!"
print_status "Model saved to: $MODEL_DIR"
print_status "Dataset saved to: $DATASET_DIR" 