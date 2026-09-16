#!/bin/bash

# Shared by the evaluation Slurm jobs. The caller must define LOG_DIR,
# JOB_TAG, EXPECTED_GPU_COUNT, and CUDA_VISIBLE_DEVICES before sourcing.

log_message() {
    echo "[$(date '+%F %T')] $*"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        log_message "ERROR: required file is missing: $1"
        exit 1
    fi
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    log_message "ERROR: Slurm did not provide CUDA_VISIBLE_DEVICES."
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
if (( ${#GPU_IDS[@]} != EXPECTED_GPU_COUNT )); then
    log_message "ERROR: expected $EXPECTED_GPU_COUNT GPUs, received ${#GPU_IDS[@]}: $CUDA_VISIBLE_DEVICES"
    exit 1
fi

TASK_PIDS=()
TASK_NAMES=()
TASK_STDOUT=()
TASK_STDERR=()

launch_task() {
    local process_name="$1"
    local script_name="$2"
    local gpu_slot="$3"
    local gpu_id="${GPU_IDS[$gpu_slot]}"
    local stdout_path="$LOG_DIR/${JOB_TAG}_${process_name}.out"
    local stderr_path="$LOG_DIR/${JOB_TAG}_${process_name}.err"

    log_message "START $process_name -> $script_name on allocated GPU $gpu_id"
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export OMP_NUM_THREADS=4
        python -u "$script_name"
    ) >"$stdout_path" 2>"$stderr_path" &

    TASK_PIDS+=("$!")
    TASK_NAMES+=("$process_name")
    TASK_STDOUT+=("$stdout_path")
    TASK_STDERR+=("$stderr_path")
}

wait_for_wave() {
    local failed=0
    local status=0
    local index

    for index in "${!TASK_PIDS[@]}"; do
        if wait "${TASK_PIDS[$index]}"; then
            log_message "DONE  ${TASK_NAMES[$index]}"
        else
            status=$?
            failed=1
            log_message "FAILED ${TASK_NAMES[$index]} with exit code $status"
            log_message "See ${TASK_STDOUT[$index]} and ${TASK_STDERR[$index]}"
        fi
    done

    TASK_PIDS=()
    TASK_NAMES=()
    TASK_STDOUT=()
    TASK_STDERR=()

    if (( failed != 0 )); then
        log_message "Stopping because at least one process in the wave failed."
        exit 1
    fi
}
