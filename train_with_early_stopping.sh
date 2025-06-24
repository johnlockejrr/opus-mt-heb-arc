#!/bin/bash

# Hebrew-Aramaic Translation Training with Early Stopping
# This script trains the model for more epochs with early stopping

set -e  # Exit on any error

echo "=== Hebrew-Aramaic Translation Training with Early Stopping ==="
echo ""

# Configuration
DATASET_DIR="./hebrew_aramaic_dataset"
MODEL_DIR="./hebrew_aramaic_model_early_stopping"
BASE_MODEL="Helsinki-NLP/opus-mt-mul-en"  # Start from the original pre-trained model
DIRECTION="he2arc"
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=10  # Train for more epochs, let early stopping decide when to stop
EARLY_STOPPING_PATIENCE=3  # Stop if no improvement for 3 evaluations
EARLY_STOPPING_THRESHOLD=0.5  # Minimum BLEU improvement threshold

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_status "Starting training with early stopping:"
echo "  Base model: $BASE_MODEL"
echo "  Output directory: $MODEL_DIR"
echo "  Max epochs: $NUM_EPOCHS"
echo "  Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "  Early stopping threshold: $EARLY_STOPPING_THRESHOLD"
echo ""

# Train with early stopping
print_status "Training model with early stopping..."
python train_translation_model.py \
    --dataset_path "$DATASET_DIR" \
    --output_dir "$MODEL_DIR" \
    --model_name "$BASE_MODEL" \
    --direction "$DIRECTION" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
    --early_stopping_threshold "$EARLY_STOPPING_THRESHOLD" \
    --use_fp16

if [ $? -ne 0 ]; then
    print_warning "Training failed!"
    exit 1
fi

print_status "Training with early stopping completed!"
print_status "Model saved to: $MODEL_DIR"
echo ""
print_status "You can now test the model:"
echo "  python inference.py --model_path $MODEL_DIR" 