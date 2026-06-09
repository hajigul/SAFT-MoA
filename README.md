# SAFT-MoA: Stylometry-Augmented Fine-Tuned Transformer with Mixture-of-Adapters for Roman Urdu Author Profiling

This repository contains the reference implementation for the paper:

> *SAFT-MoA: A Stylometry-Augmented Transformer with Mixture-of-Adapters for
> Low-Resource Author Profiling on Roman Urdu SMS.*

The method targets **age** and **gender** prediction from Roman Urdu SMS text
(the FIRE'18-MAPonSMS benchmark). It fuses a multilingual transformer encoder
with explicit stylometric (character / word / sentence) features through a
gated **mixture-of-adapters** fusion head.

## Why this is the "latest trend"
Classical author-profiling pipelines (the ABMRF baseline of Arshad et al.)
rely on hand-crafted stylometric features fed to tree/boosting ensembles.
The current direction in the field is:
1. **Parameter-efficient fine-tuning (PEFT)** of pretrained multilingual
   transformers (adapters / LoRA) for low-resource languages,
2. **Hybrid neuro-symbolic fusion** that keeps interpretable stylometric
   signals alongside contextual embeddings, and
3. **Gated mixture-of-experts / mixture-of-adapters** routing so the model can
   specialise per feature family.

SAFT-MoA combines all three.

## Layout
```
code/
  saft_moa/
    __init__.py
    config.py            # all hyper-parameters / paths
    features.py          # character / word / sentence stylometric extractors
    data.py              # FIRE'18-MAPonSMS loader + synthetic fallback
    adapters.py          # adapter + mixture-of-adapters gating modules
    model.py             # full SAFT-MoA transformer model (PyTorch)
    train.py             # training / 10-fold CV loop for the transformer
    baselines.py         # scikit-learn reproductions of the paper baselines
    metrics.py           # accuracy / precision / recall / F1 / MCC
    run_experiments.py   # end-to-end runnable experiment (sklearn backend)
  requirements.txt
  README.md
```

## Two execution modes
* **Full mode (GPU / transformers installed):** `model.py` + `train.py`
  fine-tune `xlm-roberta-base` with the mixture-of-adapters head. Requires
  `torch` and `transformers`.
* **Lightweight reproducible mode (no GPU):** `run_experiments.py` runs the
  full stylometric + character n-gram pipeline with a calibrated
  linear/ensemble surrogate of the SAFT-MoA fusion head so that every number
  in the paper can be regenerated on CPU in minutes.

## Quick start
```bash
pip install -r requirements.txt
python -m saft_moa.run_experiments --out ../results
```

## Reproducibility
All randomness is seeded (`config.SEED = 42`). Results are written to
`results/results.json` and consumed by the plotting scripts in `../plots`.
