#!/bin/bash

# RAE-XMC, both CGDPF cases, and both Our Approach cases.

#SBATCH --job-name=stage04_fusions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/fusions_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/fusions_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
BATCH_DIR="$EVALUATION_DIR/batch_run"
OUTPUT_ROOT="$BATCH_DIR/outputs"
LOG_DIR="$BATCH_DIR/logs"
STAGE01_DIR=/home/gurleen/paper/methodology/01_cwe_capec_ground_truth
JOB_TAG="${SLURM_JOB_ID:-local}"

mkdir -p "$OUTPUT_ROOT/combined" "$LOG_DIR"
cd "$EVALUATION_DIR"

source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

for file in \
    "$STAGE01_DIR/CWE_MAPPED.csv" \
    "$STAGE01_DIR/capec_relationships.csv" \
    "$OUTPUT_ROOT/CVE_to_CWE/NN_1way/best_cwe_multilabel_mlp.pt" \
    "$OUTPUT_ROOT/CVE_to_CWE/NN_keywords/best_cwe_multilabel_mlp.pt" \
    "$OUTPUT_ROOT/CVE_to_CWE/NN_fine_tune/best_cwe_multilabel_mlp.pt" \
    "$OUTPUT_ROOT/CVE_to_CWE/NN_fine_tune_keywords/best_cwe_multilabel_mlp.pt" \
    "$OUTPUT_ROOT/NN/best_capec_multilabel_mlp.pt" \
    "$OUTPUT_ROOT/NN_keywords/best_capec_multilabel_mlp.pt" \
    "$OUTPUT_ROOT/NN_fine_tune/best_capec_multilabel_mlp.pt" \
    "$OUTPUT_ROOT/NN_fine_tune_keywords/best_capec_multilabel_mlp.pt"; do
    if [[ ! -s "$file" ]]; then
        echo "Missing prerequisite: $file" >&2
        echo "Run the CWE and CAPEC batch files first." >&2
        exit 1
    fi
done

run_model() {
    local name="$1"
    local script="$2"
    echo "[$(date '+%F %T')] Starting $name"
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --gres=gpu:1 --kill-on-bad-exit=1 \
        --output="$LOG_DIR/${name}_${JOB_TAG}.out" \
        --error="$LOG_DIR/${name}_${JOB_TAG}.err" \
        python -u "$script"
    echo "[$(date '+%F %T')] Finished $name"
}

run_model rae_xmc rae_xmc.py
run_model cgdpf_description CGDPF.py
run_model cgdpf_keywords CGDPF_keywords.py

DESCRIPTION_CACHE="$OUTPUT_ROOT/combined/latest_CGDPF_run.txt"
KEYWORD_CACHE="$OUTPUT_ROOT/combined/latest_CGDPF_keywords_run.txt"
if [[ ! -s "$DESCRIPTION_CACHE" || ! -s "$KEYWORD_CACHE" ]]; then
    echo "One or both CGDPF cache pointers are missing." >&2
    exit 1
fi

export SCORE_CACHE_RUN="$(<"$DESCRIPTION_CACHE")"
run_model our_approach our_approach.py

DESCRIPTION_APPROACH_POINTER="$OUTPUT_ROOT/combined/latest_our_approach_run.txt"
if [[ ! -s "$DESCRIPTION_APPROACH_POINTER" ]]; then
    echo "Description-only Our Approach pointer is missing." >&2
    exit 1
fi
export RAE_SCORE_CACHE_RUN="$(<"$DESCRIPTION_APPROACH_POINTER")"
export SCORE_CACHE_RUN="$(<"$KEYWORD_CACHE")"
run_model our_approach_keywords our_approach_keywords.py

echo "[$(date '+%F %T')] Creating complete comparison table"
python -u collect_results.py \
    >"$LOG_DIR/collect_results_${JOB_TAG}.out" \
    2>"$LOG_DIR/collect_results_${JOB_TAG}.err"

SUMMARY="$OUTPUT_ROOT/model_results_summary.csv"
if [[ ! -s "$SUMMARY" ]]; then
    echo "Result collector did not create: $SUMMARY" >&2
    exit 1
fi

echo "Fusion group completed: $(date)"
echo "Results table: $SUMMARY"
