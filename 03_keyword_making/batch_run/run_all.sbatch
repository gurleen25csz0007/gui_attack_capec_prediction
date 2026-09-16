#!/bin/bash

# One 48-hour allocation for keyword generation and MiniLM cosine evaluation.
# Each GPU method receives exactly one A100. Slurm runs at most two method
# steps at once because the allocation contains two GPUs.

#SBATCH --job-name=keyword_full_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/03_keyword_making/batch_run/logs/pipeline_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/03_keyword_making/batch_run/logs/pipeline_%j.err

# Add site-specific --partition or --account directives above if required.

set -euo pipefail

PROJECT_DIR=/home/gurleen/paper/methodology/03_keyword_making
BATCH_DIR="$PROJECT_DIR/batch_run"
METHODS_DIR="$PROJECT_DIR/methods"
EVALUATION_SCRIPT="$PROJECT_DIR/evaluation/evaluate_all_methods.py"
OUTPUT_ROOT="$BATCH_DIR/outputs"
LOG_DIR="$BATCH_DIR/logs"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu

export PYTHONUNBUFFERED=1
export KEYWORD_MAKER_OUTPUT_ROOT="$OUTPUT_ROOT"

echo "Pipeline started: $(date)"
echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Output root: $OUTPUT_ROOT"

# Submit the four one-GPU methods as exclusive job steps. Two start
# immediately; pending steps start whenever either A100 becomes free.
GPU_METHODS=(
    occlusion
    phrase_similarity
    keybert_modified
    qwen_keywords
)
CPU_METHODS=(textrank)
ALL_METHODS=("${GPU_METHODS[@]}" "${CPU_METHODS[@]}")
GPU_PIDS=()

for METHOD_NAME in "${GPU_METHODS[@]}"; do
    echo "Queued $METHOD_NAME: $(date)"
    srun \
        --exclusive \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --gres=gpu:1 \
        --kill-on-bad-exit=1 \
        --output="$LOG_DIR/${METHOD_NAME}_${SLURM_JOB_ID}.out" \
        --error="$LOG_DIR/${METHOD_NAME}_${SLURM_JOB_ID}.err" \
        python "$METHODS_DIR/${METHOD_NAME}.py" &
    GPU_PIDS+=("$!")
done

GPU_FAILURE=0
for INDEX in "${!GPU_PIDS[@]}"; do
    METHOD_NAME="${GPU_METHODS[$INDEX]}"
    if wait "${GPU_PIDS[$INDEX]}"; then
        echo "Finished $METHOD_NAME: $(date)"
    else
        STATUS=$?
        echo "$METHOD_NAME failed with status $STATUS"
        GPU_FAILURE=1
    fi
done

if [ "$GPU_FAILURE" -ne 0 ]; then
    echo "At least one GPU keyword method failed; evaluation was not started."
    exit 1
fi

# TextRank is CPU-only. Run it with all 16 allocated CPU cores after the GPU
# generators so the final comparison contains all five keyword methods.
for METHOD_NAME in "${CPU_METHODS[@]}"; do
    echo "Starting $METHOD_NAME: $(date)"
    srun \
        --exclusive \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --kill-on-bad-exit=1 \
        --output="$LOG_DIR/${METHOD_NAME}_${SLURM_JOB_ID}.out" \
        --error="$LOG_DIR/${METHOD_NAME}_${SLURM_JOB_ID}.err" \
        python "$METHODS_DIR/${METHOD_NAME}.py"
    echo "Finished $METHOD_NAME: $(date)"
done

# Refuse to evaluate partial method output. Only the final split files are
# required; worker/temp files are disabled by the method configurations.
for METHOD_NAME in "${ALL_METHODS[@]}"; do
    for SPLIT_NAME in training validation testing; do
        FINAL_FILE="$OUTPUT_ROOT/$METHOD_NAME/${SPLIT_NAME}_final.csv"
        if [ ! -s "$FINAL_FILE" ]; then
            echo "Missing required final file: $FINAL_FILE"
            exit 1
        fi
    done
done

# Evaluation is deliberately last. Fixed MiniLM embeddings use one GPU, while
# cosine retrieval and metric calculation use all 16 allocated CPU cores.
echo "Starting keyword-only MiniLM cosine Top-5 comparison: $(date)"
srun \
    --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gres=gpu:1 \
    --kill-on-bad-exit=1 \
    --output="$LOG_DIR/evaluation_${SLURM_JOB_ID}.out" \
    --error="$LOG_DIR/evaluation_${SLURM_JOB_ID}.err" \
    env KEYWORD_EVAL_CPU_THREADS=16 \
        OMP_NUM_THREADS=16 \
        MKL_NUM_THREADS=16 \
        OPENBLAS_NUM_THREADS=16 \
        NUMEXPR_NUM_THREADS=16 \
        RAYON_NUM_THREADS=16 \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        python "$EVALUATION_SCRIPT"
EVALUATION_STATUS=$?

if [ "$EVALUATION_STATUS" -ne 0 ]; then
    echo "MiniLM cosine Top-5 evaluation failed with status $EVALUATION_STATUS"
    exit "$EVALUATION_STATUS"
fi

COMPARISON_PATH="$OUTPUT_ROOT/evaluation/minilm_cosine_top5/comparison_summary.csv"
if [ ! -s "$COMPARISON_PATH" ]; then
    echo "MiniLM cosine evaluation did not create: $COMPARISON_PATH"
    exit 1
fi

for METHOD_NAME in "${ALL_METHODS[@]}"; do
    if ! awk -F, -v method="$METHOD_NAME" \
        'NR == 1 {
            for (column = 1; column <= NF; column++) {
                if ($column == "method") method_column = column
            }
            next
        }
        method_column && $method_column == method { found = 1 }
        END { exit !found }' \
        "$COMPARISON_PATH"; then
        echo "MiniLM cosine comparison is missing method: $METHOD_NAME"
        exit 1
    fi
done

echo "Pipeline completed: $(date)"
echo "Comparison: $COMPARISON_PATH"
