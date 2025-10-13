#!/bin/bash
# Evaluation script for CHQ-Summ
# Compatible with Transformers 4.35+

set -e  # Exit on error

echo "======================================================================"
echo "🚀 CHQ-SUMM MODEL EVALUATION"
echo "======================================================================"

# Configuration
export CUDA_VISIBLE_DEVICES=0

# Directories (UPDATE THESE PATHS)
PROJECT_DIR="$HOME/Yahoo-CHQ-Summ"
DATA_DIR="${PROJECT_DIR}/data"
MODELS_DIR="${PROJECT_DIR}/models"
RESULTS_DIR="${PROJECT_DIR}/results"

# Create directories
mkdir -p $MODELS_DIR
mkdir -p $RESULTS_DIR

# Check if data exists
if [ ! -f "${DATA_DIR}/test.json" ]; then
    echo "❌ Error: Dataset not found in ${DATA_DIR}"
    echo "   Please download CHQ-Summ dataset first"
    exit 1
fi

# Models to evaluate
MODELS=("bart" "pegasus" "t5" "prophetnet")

echo ""
echo "📊 Configuration:"
echo "   Project Dir: ${PROJECT_DIR}"
echo "   Data Dir:    ${DATA_DIR}"
echo "   Results Dir: ${RESULTS_DIR}"
echo "   Models:      ${MODELS[@]}"
echo ""

# Evaluate each model
for model in "${MODELS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "🔬 Evaluating ${model^^}"
    echo "======================================================================"
    
    # Check if fine-tuned model exists
    MODEL_PATH="${MODELS_DIR}/${model}/final"
    
    if [ -d "$MODEL_PATH" ]; then
        echo "✓ Using fine-tuned model: ${MODEL_PATH}"
        
        python -m evaluation.evaluate \
            --model $model \
            --model_path $MODEL_PATH \
            --data_dir $DATA_DIR \
            --samples 100
    else
        echo "⚠️  No fine-tuned model found, using pre-trained"
        
        python -m evaluation.evaluate \
            --model $model \
            --data_dir $DATA_DIR \
            --samples 100
    fi
    
    echo ""
    echo "✅ ${model^^} evaluation complete"
done

echo ""
echo "======================================================================"
echo "✅ ALL EVALUATIONS COMPLETE"
echo "======================================================================"
echo "📁 Results saved to: ${RESULTS_DIR}/"
echo ""

# Display summary
if [ -f "${RESULTS_DIR}/ALL_MODELS_eval.csv" ]; then
    echo "📊 Summary Results:"
    echo ""
    python -c "
import pandas as pd
try:
    df = pd.read_csv('${RESULTS_DIR}/ALL_MODELS_eval.csv')
    print(df.groupby('model')[['rouge_l', 'bertscore']].mean().round(4))
except Exception as e:
    print(f'Could not display summary: {e}')
"
fi
