 # CVE to CAPEC UI

This local web app runs only the trained hybrid CAPEC predictor:

- 48% direct CVE-to-CAPEC score
- 52% CVE-to-CWE-to-CAPEC mapping score
- Hybrid threshold: 0.87
- CWE mapping uses the top 10 predicted CWEs

The model weights, tokenizers, and mapping file are bundled inside this folder.
The train/validation/test index is loaded from the selected Stage-03 KeyBERT
Modified outputs under `paper/methodology/03_keyword_making/outputs/keybert_modified`.
No network call is made during prediction.

## Run

The existing `torchgpu` environment already contains the required packages:

```bash
cd /home/gurleen/UI
/home/gurleen/miniconda3/envs/torchgpu/bin/python server.py
```

Open <http://127.0.0.1:8000> in a browser. Stop the server with `Ctrl+C`.

To use another environment:

```bash
cd /home/gurleen/UI
python -m pip install -r requirements.txt
python server.py
```

The first startup loads two MiniLM checkpoints and indexes the three dataset splits, so it can take a few seconds.

## Input and evaluation behavior

- A CVE ID is resolved from the bundled datasets and its cleaned description is sent to the model.
- Free-form vulnerability text is sent directly to the model.
- If pasted text exactly matches a dataset description, its CVE and correct CAPEC labels are also resolved.
- When dataset ground truth exists, the UI shows the true CWE and CAPEC labels. It reports CAPEC Top-1 correctness and Top-5/Top-10 hits, plus whether the threshold-selected multi-label output has any hit and whether it exactly matches all CAPEC labels.
- An unknown CVE ID cannot be predicted from the identifier alone; paste its description instead.

Hybrid scores are model ranking scores, not calibrated probabilities.
