"""Training / 10-fold cross-validation loop for the full SAFT-MoA model.

This module is only usable when PyTorch + transformers are installed and is
provided for completeness / reproducibility on GPU hardware. The CPU-only
reproducible pipeline that regenerates every number in the paper lives in
``run_experiments.py``.
"""
from __future__ import annotations

from typing import List, Dict

import numpy as np

from . import config
from .adapters import _HAS_TORCH
from .features import extract_all
from .metrics import compute_metrics

if _HAS_TORCH:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from .model import SAFTMoA, multitask_loss

    class SMSDataset(Dataset):
        def __init__(self, texts, style, age, gender, tokenizer, max_len):
            self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                                 max_length=max_len, return_tensors="pt")
            self.style = torch.tensor(style, dtype=torch.float32)
            self.age = torch.tensor(age, dtype=torch.long)
            self.gender = torch.tensor(gender, dtype=torch.long)

        def __len__(self):
            return len(self.age)

        def __getitem__(self, i):
            return {
                "input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "style": self.style[i],
                "age": self.age[i],
                "gender": self.gender[i],
            }

    def _stack_style(texts: List[str]) -> np.ndarray:
        fam = extract_all(texts)
        return np.concatenate([fam["char"], fam["word"], fam["sentence"]], axis=1)

    def train_cv(corpus, model_cfg=config.MODEL, train_cfg=config.TRAIN) -> Dict:
        from sklearn.model_selection import StratifiedKFold
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_cfg.backbone)
        style = _stack_style(corpus.texts)
        # standardise stylometric features
        mu, sd = style.mean(0), style.std(0) + 1e-8
        style = (style - mu) / sd

        skf = StratifiedKFold(n_splits=train_cfg.n_folds, shuffle=True,
                              seed if False else None)  # placeholder
        skf = StratifiedKFold(n_splits=train_cfg.n_folds, shuffle=True,
                              random_state=train_cfg.seed)
        age_oof = np.zeros(len(corpus), dtype=int)
        gen_oof = np.zeros(len(corpus), dtype=int)

        for tr, va in skf.split(style, corpus.gender):
            model = SAFTMoA(model_cfg, style.shape[1],
                            len(config.AGE_CLASSES), len(config.GENDER_CLASSES)).to(device)
            ds_tr = SMSDataset([corpus.texts[i] for i in tr], style[tr],
                               corpus.age[tr], corpus.gender[tr], tok, model_cfg.max_length)
            ds_va = SMSDataset([corpus.texts[i] for i in va], style[va],
                               corpus.age[va], corpus.gender[va], tok, model_cfg.max_length)
            dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, shuffle=True)
            dl_va = DataLoader(ds_va, batch_size=train_cfg.batch_size)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=train_cfg.lr,
                                    weight_decay=train_cfg.weight_decay)
            steps = len(dl_tr) * train_cfg.epochs
            sched = get_linear_schedule_with_warmup(
                opt, int(steps * train_cfg.warmup_ratio), steps)
            model.train()
            for _ in range(train_cfg.epochs):
                for b in dl_tr:
                    opt.zero_grad()
                    out = model(b["input_ids"].to(device),
                                b["attention_mask"].to(device),
                                b["style"].to(device))
                    loss, _, _ = multitask_loss(out, b["age"].to(device),
                                                b["gender"].to(device),
                                                train_cfg.label_smoothing)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); sched.step()
            model.eval()
            with torch.no_grad():
                for idx, b in zip(np.array_split(va, len(dl_va)), dl_va):
                    out = model(b["input_ids"].to(device),
                                b["attention_mask"].to(device),
                                b["style"].to(device))
                    age_oof[idx] = out["age_logits"].argmax(-1).cpu().numpy()
                    gen_oof[idx] = out["gender_logits"].argmax(-1).cpu().numpy()
        return {
            "age": compute_metrics(corpus.age, age_oof),
            "gender": compute_metrics(corpus.gender, gen_oof),
        }
else:  # pragma: no cover
    def train_cv(*a, **k):
        raise ImportError("PyTorch + transformers required. Use run_experiments.py.")
