#!/bin/bash

# Four direct CVE-to-CAPEC models on one exclusive GPU, run sequentially.

#SBATCH --job-name=stage04_cve_to_capec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/cve_to_capec_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/cve_to_capec_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
BATCH_DIR="$EVALUATION_DIR/batch_run"
OUTPUT_ROOT="$BATCH_DIR/outputs"
LOG_DIR="$BATCH_DIR/logs"
KEYBERT_DIR=/home/gurleen/paper/methodology/03_keyword_making/outputs/keybert_modified
STAGE01_DIR=/home/gurleen/paper/methodology/01_cwe_capec_ground_truth
JOB_TAG="${SLURM_JOB_ID:-local}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
cd "$EVALUATION_DIR"

source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

for file in \
    "$STAGE01_DIR/3000_capec.csv" \
    "$KEYBERT_DIR/training_final.csv" \
    "$KEYBERT_DIR/validation_final.csv" \
    "$KEYBERT_DIR/testing_final.csv"; do
    if [[ ! -s "$file" ]]; then
        echo "Missing required input: $file" >&2
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

run_model capec_nn NN.py
run_model capec_nn_keywords NN_keywords.py
run_model capec_fine_tune NN_fine_tune.py
run_model capec_fine_tune_keywords NN_fine_tune_keywords.py

echo "CVE-to-CAPEC group completed: $(date)"
echo "Outputs: $OUTPUT_ROOT"
