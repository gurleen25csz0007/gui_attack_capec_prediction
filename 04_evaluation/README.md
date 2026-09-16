# Stage 04: complete CVE classification comparison

Every model reads the same fixed files produced by the selected Stage-03
`keybert_modified` method:

- `../03_keyword_making/outputs/keybert_modified/training_final.csv`
- `../03_keyword_making/outputs/keybert_modified/validation_final.csv`
- `../03_keyword_making/outputs/keybert_modified/testing_final.csv`

The comparison matrix is:

| Target | Cleaned description | Cleaned description + KeyBERT keywords |
|---|---|---|
| CAPEC, frozen MiniLM | `NN.py` | `NN_keywords.py` |
| CAPEC, fine-tuned MiniLM | `NN_fine_tune.py` | `NN_fine_tune_keywords.py` |
| CWE, frozen MiniLM | `CVE_to_CWE/NN_1way.py` | `CVE_to_CWE/NN_keywords.py` |
| CWE, fine-tuned MiniLM | `CVE_to_CWE/NN_fine_tune.py` | `CVE_to_CWE/NN_fine_tune_keywords.py` |
| CGDPF | `CGDPF.py` | `CGDPF_keywords.py` |

`rae_xmc.py` remains the standalone retrieval-augmented model and
`our_approach.py` is the final description-only, validation-calibrated fusion
of CGDPF and RAE-XMC. `rae_xmc_keywords.py` trains the corresponding
description-plus-KeyBERT RAE-XMC model. `our_approach_keywords.py` combines the
KeyBERT-aware CWE, direct-CAPEC, and RAE-XMC branches. The two cases are
calibrated and saved independently.

All eight neural baselines use seed 42, a maximum of 50 epochs, patience 5,
dropout 0.30, validation Micro-F1 checkpoint selection, and the same validation
threshold grid. These classifier decision thresholds are separate from the
Stage-03 keyword-extraction hyperparameters. Each model selects its decision
threshold using validation predictions only, then evaluates the test split
exactly once with that fixed threshold. Frozen models use batch size 256.
Fine-tuned models use physical batch size 32 with eight-step gradient
accumulation, giving the same effective batch size of 256; their encoder
learning rate is necessarily lower than the MLP learning rate.

The two CVE-to-CWE fine-tuning files are standalone programs: they contain
their own data loading, model, training, threshold selection, evaluation, and
artifact-writing code and do not import `fine_tune_runner.py`. That runner is
used only by the two direct CVE-to-CAPEC fine-tuning programs.

The shared validation threshold grid is intentionally restricted to:

```python
thresholds = np.array([
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.96,
    0.97,
    0.98,
    0.99,
])
```

The four direct CVE-to-CAPEC NN variants use these same nine candidates. Other
Stage-04 model families retain their own documented threshold configuration.

CGDPF selects its direct-versus-mapping alpha on the inclusive grid from 0.00
to 1.00 in steps of 0.02. The combined CGDPF/RAE-XMC models use the same grid
for both the CGDPF alpha and the final RAE-XMC fusion beta. RAE-XMC selects
lambda from this same inclusive grid. Grid metrics and selected rows are saved
as `fusion_calibration_results.csv` or `lambda_calibration_results.csv` in the
corresponding run directory.

All CAPEC evaluators use the union of the 559 catalog IDs from
`../01_cwe_capec_ground_truth/3000_capec.csv` and every real CAPEC in the fixed
splits, followed by `CAPEC-noID`. The current fixed splits add no out-of-catalog
IDs, giving a consistent 560-column evaluation space.

Run the complete three-job comparison with:

```bash
bash /home/gurleen/paper/methodology/04_evaluation/batch_run/submit_all.sh
```

The helper submits three jobs: CWE and direct CAPEC may run independently, and
the fusion job starts only after both succeed. Each job requests one GPU, four
CPU cores, 32 GB RAM, and at most 48 hours. Models within each job run
sequentially. Batch artifacts are isolated under `batch_run/outputs/`, and
every model has separate stdout/stderr under `batch_run/logs/`. At completion,
`collect_results.py` writes `batch_run/outputs/model_results_summary.csv`.
Its winner flag and rank are based on validation Micro-F1; test metrics remain
the final unbiased report. The summary publishes exactly two final Our
Approach rows (`our_approach` and `our_approach_keywords`); calibrated RAE and
original-hybrid branches remain available inside each run as diagnostics but
are not counted as additional models. The collector also rejects an RAE
directory while an in-place training run is incomplete, preventing validation
metrics from one execution being combined with test metrics from another.

Interactive execution still defaults to `outputs/`. All Stage-04 programs
honor `EVALUATION_OUTPUT_ROOT` when a different result location is needed.
