#!/bin/bash

# Resume only the failed keyword-aware Our Approach and final result collector.

#SBATCH --job-name=stage04_resume_pending
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/resume_pending_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/resume_pending_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
BATCH_DIR="$EVALUATION_DIR/batch_run"
OUTPUT_ROOT="$BATCH_DIR/outputs"
LOG_DIR="$BATCH_DIR/logs"
KEYWORD_CACHE_POINTER="$OUTPUT_ROOT/combined/latest_CGDPF_keywords_run.txt"
RAE_CACHE_POINTER="$OUTPUT_ROOT/combined/latest_our_approach_run.txt"
JOB_TAG="${SLURM_JOB_ID:-local}"

mkdir -p "$OUTPUT_ROOT/combined" "$LOG_DIR"
cd "$EVALUATION_DIR"

source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

for pointer in "$KEYWORD_CACHE_POINTER" "$RAE_CACHE_POINTER"; do
    if [[ ! -s "$pointer" ]]; then
        echo "Missing completed-run pointer: $pointer" >&2
        exit 1
    fi
done

export SCORE_CACHE_RUN="$(<"$KEYWORD_CACHE_POINTER")"
export RAE_SCORE_CACHE_RUN="$(<"$RAE_CACHE_POINTER")"

for file in \
    "$SCORE_CACHE_RUN/validation_cwe_scores.npy" \
    "$SCORE_CACHE_RUN/test_cwe_scores.npy" \
    "$SCORE_CACHE_RUN/validation_direct_scores.npy" \
    "$SCORE_CACHE_RUN/test_direct_scores.npy" \
    "$RAE_SCORE_CACHE_RUN/validation_rae_scores.npy" \
    "$RAE_SCORE_CACHE_RUN/test_rae_scores.npy" \
    "$RAE_SCORE_CACHE_RUN/idx_to_label.json"; do
    if [[ ! -s "$file" ]]; then
        echo "Missing cached prerequisite: $file" >&2
        exit 1
    fi
done

echo "[$(date '+%F %T')] Starting pending our_approach_keywords"
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gres=gpu:1 --kill-on-bad-exit=1 \
    --output="$LOG_DIR/our_approach_keywords_resume_${JOB_TAG}.out" \
    --error="$LOG_DIR/our_approach_keywords_resume_${JOB_TAG}.err" \
    python -u our_approach_keywords.py
echo "[$(date '+%F %T')] Finished pending our_approach_keywords"

echo "[$(date '+%F %T')] Creating complete comparison table"
python -u collect_results.py \
    >"$LOG_DIR/collect_results_resume_${JOB_TAG}.out" \
    2>"$LOG_DIR/collect_results_resume_${JOB_TAG}.err"

SUMMARY="$OUTPUT_ROOT/model_results_summary.csv"
if [[ ! -s "$SUMMARY" ]]; then
    echo "Result collector did not create: $SUMMARY" >&2
    exit 1
fi

echo "Resume completed: $(date)"
echo "Results table: $SUMMARY"
