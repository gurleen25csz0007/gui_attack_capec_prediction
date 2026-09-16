#!/bin/bash

# After Qwen training, rerun both updated Our Approach cases and collect all results.

#SBATCH --job-name=updated_approaches_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/updated_approaches_summary_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/updated_approaches_summary_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
BATCH_DIR="$EVALUATION_DIR/batch_run"
OUTPUT_ROOT="$BATCH_DIR/outputs"
COMBINED_DIR="$OUTPUT_ROOT/combined"

cd "$EVALUATION_DIR"
source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

for metrics in \
    "$OUTPUT_ROOT/qwen_nn/metrics.json" \
    "$OUTPUT_ROOT/qwen_nnfinetuned/metrics.json"; do
    if [[ ! -s "$metrics" ]]; then
        echo "Missing completed Qwen metrics: $metrics" >&2
        exit 1
    fi
done

DESCRIPTION_CGDPF_POINTER="$COMBINED_DIR/latest_CGDPF_run.txt"
KEYWORD_CGDPF_POINTER="$COMBINED_DIR/latest_CGDPF_keywords_run.txt"
OLD_RAE_CACHE_POINTER="$COMBINED_DIR/latest_our_approach_run.txt"
for pointer in \
    "$DESCRIPTION_CGDPF_POINTER" \
    "$KEYWORD_CGDPF_POINTER" \
    "$OLD_RAE_CACHE_POINTER"; do
    if [[ ! -s "$pointer" ]]; then
        echo "Missing completed-run pointer: $pointer" >&2
        exit 1
    fi
done

export SCORE_CACHE_RUN="$(<"$DESCRIPTION_CGDPF_POINTER")"
export RAE_SCORE_CACHE_RUN="$(<"$OLD_RAE_CACHE_POINTER")"
echo "[$(date '+%F %T')] Starting updated description Our Approach"
python -u our_approach.py
echo "[$(date '+%F %T')] Finished updated description Our Approach"

# The description run above now supplies an aligned RAE cache in the new format.
export SCORE_CACHE_RUN="$(<"$KEYWORD_CGDPF_POINTER")"
export RAE_SCORE_CACHE_RUN="$(<"$COMBINED_DIR/latest_our_approach_run.txt")"
echo "[$(date '+%F %T')] Starting updated keyword Our Approach"
python -u our_approach_keywords.py
echo "[$(date '+%F %T')] Finished updated keyword Our Approach"

echo "[$(date '+%F %T')] Regenerating complete model summary"
python -u collect_results.py

SUMMARY="$OUTPUT_ROOT/model_results_summary.csv"
if [[ ! -s "$SUMMARY" ]]; then
    echo "Result collector did not create: $SUMMARY" >&2
    exit 1
fi
for model in RAE_XMC RAE_XMC_calibrated our_approach; do
    if ! awk -F, -v expected="$model" 'NR > 1 && $1 == expected { found=1 } END { exit !found }' "$SUMMARY"; then
        echo "Summary is missing required model row: $model" >&2
        exit 1
    fi
done

echo "Updated evaluation and summary completed: $(date)"
echo "Results table: $SUMMARY"
