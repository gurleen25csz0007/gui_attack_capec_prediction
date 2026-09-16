# CVE/description-to-CAPEC UI

The UI deploys the final Stage-04 **Our Approach** model selected by validation:
the keyword-aware, validation-calibrated fusion of CGDPF and RAE-XMC.

## Bundled trained model

`models/our_approach/` retains local inference assets, including the MiniLM
sentence encoder. The active fusion run is resolved from Stage 04's completed
keyword-aware `our_approach_keywords` pointer.

The selected run supplies:

- the keyword-aware CWE MLP checkpoint (`NN_keywords`);
- the keyword-aware direct-CAPEC MLP checkpoint (`NN_keywords`);
- the keyword-trained RAE-XMC encoder (`RAE_XMC_keywords`) and training-only retrieval memory;
- the local MiniLM sentence encoder, so startup does not require Hugging Face;
- the CWE-to-CAPEC mapping; and
- the validation-fitted calibration, alpha, beta, threshold, and label files.

The selected fusion references the trained Stage-04 component directories;
keep these artifacts in place. The UI reuses the bundled MiniLM sentence
encoder when available. You can select another completed keyword-aware run
by setting `CAPEC_UI_MODEL_PATH` to a directory such as:

```text
/home/gurleen/paper/methodology/04_evaluation/outputs/combined/our_approach_keywords_YYYYmmdd_HHMMSS
```

By default, the UI reads
`04_evaluation/batch_run/outputs/combined/latest_our_approach_keywords_run.txt`,
with the equivalent interactive-output pointer as a fallback. It reads alpha,
beta, and the final threshold directly from that run's
`fusion_parameters.json`. The UI calculates fresh CWE, direct-CAPEC, mapping,
RAE-XMC, and final fusion scores for every request; it does not look up stored
test predictions or hard-code weights.

## Input processing

- Complete CVE IDs use exact ID lookup (case-insensitive, with surrounding
  whitespace ignored). Partial IDs and CVE IDs followed by extra text are
  rejected. The Stage-02 dataset is used only to resolve a known CVE ID to its raw
  `uncleaned_description` and to show ground-truth labels for an optional
  accuracy check. Stored `cleaned_description` and `keyphrases` values are never
  used for prediction. The exact original description is preserved for display.
- Pasted descriptions also resolve to a CVE when they exactly match its
  `uncleaned_description` (ignoring surrounding whitespace). Matching is
  case-sensitive and preserves internal whitespace. Descriptions shared by
  multiple CVEs remain free text because they cannot identify a unique CVE.
- Every request, including a known CVE ID, is freshly processed by the Stage-02
  cleaning implementation copied into `text_preprocessing.py`. The final
  Stage-02 CSV does not retain CPE vendor/product/version fields, so the online
  path applies the same generic cleaning, stop-word handling, and lemmatization
  but cannot reconstruct the earlier CPE-specific masks.
- The UI displays both the original/resolved description and the exact cleaned
  description sent to every trained model branch.
- The UI reports train, validation, or test membership by reading exact CVE IDs
  from `training_final.csv`, `validation_final.csv`, and `testing_final.csv` in
  the selected model's saved `data.split_directory`. This applies to both exact
  ID and exact description matches. Source paths and split filenames are hidden
  in the results. A CVE absent from these files is
  labeled "Dataset only"; unreadable split files are labeled "Unavailable".
  Unmatched free-text input has no dataset membership. Split membership never affects
  model scores.
- Every request extracts fresh modified-KeyBERT phrases from the freshly cleaned
  description using the published Stage-03 configuration. The extracted phrases
  are displayed and passed into Our Approach for scoring. Stored dataset keywords
  are never reused.

The predictor accepts only a completed keyword-aware `our_approach_keywords_*` run,
whether resolved by the pointer or provided through `CAPEC_UI_MODEL_PATH`.
Startup validates the input case and all three source directories, and confirms
that the RAE-XMC artifact was trained with keyphrases. Description-only or mixed
artifacts are rejected. Online keyword extraction settings are checked against
the published Stage-03 configuration.

Ground-truth CAPEC/CWE labels are displayed only for dataset matches. They are
never provided to the model or used to calculate prediction scores.

## Run

Start the UI:

```bash
cd /home/gurleen/paper/methodology/05_UI
/home/gurleen/miniconda3/envs/torchgpu/bin/python server.py
```

Open <http://127.0.0.1:8000>. Use `CAPEC_UI_DEVICE` to select `cpu`, `cuda`, or
another PyTorch device. The Stage-02 cleaner also requires `en_core_web_sm`.

The page shows the CVE dataset connection and the number of indexed IDs, even
when the model is unavailable. Open the server URL rather than `index.html`
directly, because CVE lookup runs in Python. The default dataset is
`02_preprocessing/dataset.csv`; set `CAPEC_UI_DATASET_PATH` before starting the
server to use another CSV with `cve_id` and `uncleaned_description` columns.
Dataset loading errors are displayed explicitly; description input remains
available when the model is ready.
