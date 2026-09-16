#!/bin/bash

# Fine-tuned MiniLM+MLP CVE-to-CAPEC model with Qwen keywords.

#SBATCH --job-name=stage04_qwen_nnfinetuned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/qwen_nnfinetuned_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/qwen_nnfinetuned_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
OUTPUT_ROOT="${EVALUATION_OUTPUT_ROOT:-$EVALUATION_DIR/batch_run/outputs}"
QWEN_DIR=/home/gurleen/paper/methodology/03_keyword_making/outputs/qwen_keywords

for split in training validation testing; do
    file="$QWEN_DIR/${split}_final.csv"
    if [[ ! -s "$file" ]]; then
        echo "Missing Qwen split: $file" >&2
        exit 1
    fi
done

cd "$EVALUATION_DIR"
source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

python -u qwen_nnfinetuned.py
