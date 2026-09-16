#!/bin/bash

# Use two GPUs as three dependency-ordered pairs, then collect the summary.
# Existing job IDs can be supplied to resume safely at the CGDPF, RAE-XMC, or
# Our Approach phase without resubmitting work that is already running.

set -euo pipefail

BATCH_DIR=/home/gurleen/paper/methodology/04_evaluation/batch_run

submit() {
    local result
    if ! result="$(sbatch --parsable "$@")"; then
        return 1
    fi
    if [[ -z "$result" ]]; then
        echo "Slurm returned an empty job ID" >&2
        return 1
    fi
    printf '%s\n' "${result%%;*}"
}

wait_for_jobs() {
    local phase="$1"
    shift
    local job state all_complete
    while true; do
        all_complete=true
        for job in "$@"; do
            state="$(
                sacct -X -n -P -j "$job" --format=State \
                    | awk -F'|' 'NF {print $1; exit}'
            )"
            case "$state" in
                COMPLETED)
                    ;;
                FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|BOOT_FAIL*|DEADLINE*)
                    echo "$phase failed: job $job ended in state $state" >&2
                    return 1
                    ;;
                *)
                    all_complete=false
                    ;;
            esac
        done
        if [[ "$all_complete" == true ]]; then
            echo "$phase completed successfully: $*"
            return 0
        fi
        sleep 30
    done
}

RESUME_PHASE="${1:-}"
if [[ "$RESUME_PHASE" == "--resume-rae" ]]; then
    if [[ $# -ne 3 ]]; then
        echo "Usage: $0 --resume-rae RAE_DESCRIPTION_JOB RAE_KEYWORD_JOB" >&2
        exit 2
    fi
    RAE_DESC_JOB="$2"
    RAE_KEYWORDS_JOB="$3"
    RAE_DEPENDENCY="afterok:${RAE_DESC_JOB}:${RAE_KEYWORDS_JOB}"
    echo "Resuming phase 2 (parallel RAE-XMC): $RAE_DESC_JOB  $RAE_KEYWORDS_JOB"
    wait_for_jobs "RAE-XMC phase" "$RAE_DESC_JOB" "$RAE_KEYWORDS_JOB"
elif [[ "$RESUME_PHASE" == "--resume-our" ]]; then
    if [[ $# -ne 3 ]]; then
        echo "Usage: $0 --resume-our OUR_DESCRIPTION_JOB OUR_KEYWORD_JOB" >&2
        exit 2
    fi
    OUR_DESC_JOB="$2"
    OUR_KEYWORDS_JOB="$3"
    OUR_DEPENDENCY="afterok:${OUR_DESC_JOB}:${OUR_KEYWORDS_JOB}"
    echo "Resuming phase 3 (parallel Our Approach): $OUR_DESC_JOB  $OUR_KEYWORDS_JOB"
    wait_for_jobs "Our Approach phase" "$OUR_DESC_JOB" "$OUR_KEYWORDS_JOB"
else
    # Phase 1: the two CGDPF variants may occupy the two GPUs concurrently.
    if [[ $# -eq 2 ]]; then
        CGDPF_DESC_JOB="$1"
        CGDPF_KEYWORDS_JOB="$2"
    elif [[ $# -eq 0 ]]; then
        CGDPF_DESC_JOB="$(submit "$BATCH_DIR/11_cgdpf_description_grid.sbatch")"
        CGDPF_KEYWORDS_JOB="$(submit "$BATCH_DIR/12_cgdpf_keywords_grid.sbatch")"
    else
        echo "Usage: $0 [CGDPF_DESCRIPTION_JOB CGDPF_KEYWORD_JOB]" >&2
        echo "       $0 --resume-rae RAE_DESCRIPTION_JOB RAE_KEYWORD_JOB" >&2
        echo "       $0 --resume-our OUR_DESCRIPTION_JOB OUR_KEYWORD_JOB" >&2
        exit 2
    fi
    CGDPF_DEPENDENCY="afterok:${CGDPF_DESC_JOB}:${CGDPF_KEYWORDS_JOB}"
    echo "Phase 1 (parallel CGDPF):          $CGDPF_DESC_JOB  $CGDPF_KEYWORDS_JOB"
    wait_for_jobs "CGDPF phase" "$CGDPF_DESC_JOB" "$CGDPF_KEYWORDS_JOB"

    # Phase 2: both RAE-XMC variants wait for phase 1, then run concurrently.
    RAE_DESC_JOB="$(submit --dependency="$CGDPF_DEPENDENCY" "$BATCH_DIR/13_rae_xmc_description_grid.sbatch")"
    RAE_KEYWORDS_JOB="$(submit --dependency="$CGDPF_DEPENDENCY" "$BATCH_DIR/14_rae_xmc_keywords_grid.sbatch")"
    RAE_DEPENDENCY="afterok:${RAE_DESC_JOB}:${RAE_KEYWORDS_JOB}"
    echo "Phase 2 (parallel RAE-XMC):        $RAE_DESC_JOB  $RAE_KEYWORDS_JOB"
    wait_for_jobs "RAE-XMC phase" "$RAE_DESC_JOB" "$RAE_KEYWORDS_JOB"
fi

if [[ "$RESUME_PHASE" != "--resume-our" ]]; then
    # Phase 3: both final fusions wait for both complete RAE-XMC runs.
    OUR_DESC_JOB="$(submit --dependency="$RAE_DEPENDENCY" "$BATCH_DIR/15_our_approach_description_grid.sbatch")"
    OUR_KEYWORDS_JOB="$(submit --dependency="$RAE_DEPENDENCY" "$BATCH_DIR/16_our_approach_keywords_grid.sbatch")"
    OUR_DEPENDENCY="afterok:${OUR_DESC_JOB}:${OUR_KEYWORDS_JOB}"
    echo "Phase 3 (parallel Our Approach):   $OUR_DESC_JOB  $OUR_KEYWORDS_JOB"
    wait_for_jobs "Our Approach phase" "$OUR_DESC_JOB" "$OUR_KEYWORDS_JOB"
fi

COLLECT_JOB="$(submit --dependency="$OUR_DEPENDENCY" "$BATCH_DIR/17_collect_grid_results.sbatch")"
echo "Final result collector:            $COLLECT_JOB"
wait_for_jobs "Result collection" "$COLLECT_JOB"

if [[ -n "${CGDPF_DEPENDENCY:-}" ]]; then
    echo "RAE dependency:                    $CGDPF_DEPENDENCY"
fi
if [[ -n "${RAE_DEPENDENCY:-}" ]]; then
    echo "Our Approach dependency:           $RAE_DEPENDENCY"
fi
echo "Collector dependency:              $OUR_DEPENDENCY"
