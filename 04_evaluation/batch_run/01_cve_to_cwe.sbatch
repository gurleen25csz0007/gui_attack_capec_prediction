#!/bin/bash

# Four CVE-to-CWE models on one exclusive GPU, run sequentially.

#SBATCH --job-name=stage04_cve_to_cwe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/cve_to_cwe_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/cve_to_cwe_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
BATCH_DIR="$EVALUATION_DIR/batch_run"
OUTPUT_ROOT="$BATCH_DIR/outputs"
LOG_DIR="$BATCH_DIR/logs"
KEYBERT_DIR=/home/gurleen/paper/methodology/03_keyword_making/outputs/keybert_modified
JOB_TAG="${SLURM_JOB_ID:-local}"

mkdir -p "$OUTPUT_ROOT/CVE_to_CWE" "$LOG_DIR"
cd "$EVALUATION_DIR"

source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

for split in training validation testing; do
    file="$KEYBERT_DIR/${split}_final.csv"
    if [[ ! -s "$file" ]]; then
        echo "Missing KeyBERT split: $file" >&2
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

run_model cwe_nn CVE_to_CWE/NN_1way.py
run_model cwe_nn_keywords CVE_to_CWE/NN_keywords.py
run_model cwe_fine_tune CVE_to_CWE/NN_fine_tune.py
run_model cwe_fine_tune_keywords CVE_to_CWE/NN_fine_tune_keywords.py

echo "CVE-to-CWE group completed: $(date)"
echo "Outputs: $OUTPUT_ROOT/CVE_to_CWE"
