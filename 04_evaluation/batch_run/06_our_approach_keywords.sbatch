#!/bin/bash

# Rerun only keyword-aware Our Approach, then refresh the complete summary.

#SBATCH --job-name=our_approach_keywords
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/our_approach_keywords_rerun_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/our_approach_keywords_rerun_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
BATCH_DIR="$EVALUATION_DIR/batch_run"
OUTPUT_ROOT="$BATCH_DIR/outputs"
CGDPF_POINTER="$OUTPUT_ROOT/combined/latest_CGDPF_keywords_run.txt"
RAE_CACHE_POINTER="$OUTPUT_ROOT/combined/latest_our_approach_run.txt"
JOB_TAG="${SLURM_JOB_ID:-local}"

cd "$EVALUATION_DIR"
source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

for pointer in "$CGDPF_POINTER" "$RAE_CACHE_POINTER"; do
    if [[ ! -s "$pointer" ]]; then
        echo "Missing completed-run pointer: $pointer" >&2
        exit 1
    fi
done

export SCORE_CACHE_RUN="$(<"$CGDPF_POINTER")"
export RAE_SCORE_CACHE_RUN="$(<"$RAE_CACHE_POINTER")"

echo "[$(date '+%F %T')] Starting keyword Our Approach rerun"
python -u our_approach_keywords.py
echo "[$(date '+%F %T')] Finished keyword Our Approach rerun"

echo "[$(date '+%F %T')] Refreshing complete model summary"
python -u collect_results.py \
    >"$BATCH_DIR/logs/collect_results_our_approaches_${JOB_TAG}.out" \
    2>"$BATCH_DIR/logs/collect_results_our_approaches_${JOB_TAG}.err"

SUMMARY="$OUTPUT_ROOT/model_results_summary.csv"
if [[ ! -s "$SUMMARY" ]]; then
    echo "Result collector did not create: $SUMMARY" >&2
    exit 1
fi
echo "Updated results table: $SUMMARY"

