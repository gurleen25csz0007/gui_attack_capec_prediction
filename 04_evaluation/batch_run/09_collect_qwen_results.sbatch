#!/bin/bash

# Refresh the complete model summary after both Qwen models finish.

#SBATCH --job-name=collect_qwen_results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/collect_qwen_results_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/collect_qwen_results_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
OUTPUT_ROOT="$EVALUATION_DIR/batch_run/outputs"

for metrics in \
    "$OUTPUT_ROOT/qwen_nn/metrics.json" \
    "$OUTPUT_ROOT/qwen_nnfinetuned/metrics.json"; do
    if [[ ! -s "$metrics" ]]; then
        echo "Missing completed Qwen metrics: $metrics" >&2
        exit 1
    fi
done

cd "$EVALUATION_DIR"
source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

python -u collect_results.py

SUMMARY="$OUTPUT_ROOT/model_results_summary.csv"
if [[ ! -s "$SUMMARY" ]]; then
    echo "Result collector did not create: $SUMMARY" >&2
    exit 1
fi
echo "Updated results table: $SUMMARY"
