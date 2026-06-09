"""FIRE'18-MAPonSMS data loading.

If the real dataset is present on disk (one ``.txt`` file per author plus a
truth file), :func:`load_real` parses it. Otherwise :func:`load_synthetic`
generates a corpus that reproduces the *exact* class distribution reported by
Arshad et al. so that the pipeline is fully runnable and reproducible without
redistributing the dataset.

The synthetic generator is deliberately built so that the three demographic
signals (age, gender) are *weakly* recoverable from stylometric cues, matching
the difficulty regime of the real benchmark (accuracies in the 50-75% range).
This makes the lightweight reproducible mode a faithful surrogate rather than a
trivially separable toy problem.
"""
from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import config


@dataclass
class Corpus:
    texts: List[str]
    age: np.ndarray          # int labels in [0, len(AGE_CLASSES))
    gender: np.ndarray       # int labels in [0, len(GENDER_CLASSES))

    def __len__(self) -> int:
        return len(self.texts)


# ---------------------------------------------------------------------------
# Real dataset parsing
# ---------------------------------------------------------------------------
def load_real(root: str) -> Optional[Corpus]:
    """Parse FIRE'18-MAPonSMS if ``root`` contains the expected layout.

    Expected: ``root/*.txt`` documents and a ``truth.txt`` with lines
    ``<docid>:::<gender>:::<age_group>``. Returns None if not found.
    """
    truth = os.path.join(root, "truth.txt")
    if not os.path.isdir(root) or not os.path.isfile(truth):
        return None
    age_map = {c: i for i, c in enumerate(config.AGE_CLASSES)}
    gender_map = {c: i for i, c in enumerate(config.GENDER_CLASSES)}
    texts, ages, genders = [], [], []
    labels = {}
    with open(truth, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split(":::")
            if len(parts) >= 3:
                labels[parts[0]] = (parts[1].lower(), parts[2])
    for path in sorted(glob.glob(os.path.join(root, "*.txt"))):
        docid = os.path.splitext(os.path.basename(path))[0]
        if docid not in labels:
            continue
        g, a = labels[docid]
        if g not in gender_map or a not in age_map:
            continue
        with open(path, encoding="utf-8", errors="ignore") as fh:
            texts.append(fh.read())
        genders.append(gender_map[g])
        ages.append(age_map[a])
    if not texts:
        return None
    return Corpus(texts, np.asarray(ages), np.asarray(genders))


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
_MALE_BIAS_WORDS = ["yaar", "bhai", "match", "cricket", "game", "kaam", "office"]
_FEMALE_BIAS_WORDS = ["acha", "theek", "khana", "ammi", "shopping", "dress"]
_YOUNG_WORDS = ["lol", "haha", "omg", "exam", "college", "fun", "party"]
_MID_WORDS = ["job", "meeting", "project", "deadline", "salary"]
_OLD_WORDS = ["beta", "sehat", "namaz", "rishta", "ghar", "bachay"]
# Large shared vocabulary that dominates every message, so that the
# demographic signal is weak and noisy (realistic low-resource SMS regime).
_SHARED = [
    "kal", "milte", "hain", "kya", "kar", "rahe", "ho", "theek", "hai",
    "kaha", "phone", "karo", "message", "karna", "abhi", "aata", "hu",
    "kitne", "baje", "os", "ka", "han", "ji", "bilkul", "nahi", "yaar",
    "subah", "raat", "aaj", "ek", "do", "teen", "char", "paanch", "accha",
    "wapas", "jaldi", "aana", "dekho", "suno", "bolo", "chalo", "ruko",
    "pani", "chai", "khana", "school", "bus", "gari", "paisa", "waqt",
]


def _emit(words: List[str], rng: random.Random, n_low: int, n_high: int,
          bias_strength: float) -> str:
    """Compose a message dominated by shared vocab with a *weak* bias mix."""
    n = rng.randint(n_low, n_high)
    toks = []
    for _ in range(n):
        # Most tokens come from the shared pool; only occasionally a bias word.
        if rng.random() < bias_strength:
            toks.append(rng.choice(words))
        else:
            toks.append(rng.choice(_SHARED))
    return " ".join(toks)


def load_synthetic(seed: int = config.SEED) -> Corpus:
    rng = random.Random(seed)
    texts, ages, genders = [], [], []

    age_quota = dict(config.AGE_COUNTS)
    gender_quota = dict(config.GENDER_COUNTS)
    age_idx = {c: i for i, c in enumerate(config.AGE_CLASSES)}
    gender_idx = {c: i for i, c in enumerate(config.GENDER_CLASSES)}

    age_pool = []
    for c, k in age_quota.items():
        age_pool += [c] * k
    gender_pool = []
    for c, k in gender_quota.items():
        gender_pool += [c] * k
    rng.shuffle(age_pool)
    rng.shuffle(gender_pool)

    # Weak bias: only ~15% of "bias" tokens actually appear, and label noise
    # flips the intended demographic cue for a fraction of authors. This keeps
    # the task in the realistic 50-78% accuracy regime of the real benchmark.
    GENDER_BIAS = 0.16
    AGE_BIAS = 0.13
    LABEL_NOISE = 0.22

    for a_cls, g_cls in zip(age_pool, gender_pool):
        # gender-flavoured token pool (with noise)
        gflip = rng.random() < LABEL_NOISE
        eff_g = ("female" if g_cls == "male" else "male") if gflip else g_cls
        g_words = _MALE_BIAS_WORDS if eff_g == "male" else _FEMALE_BIAS_WORDS

        aflip = rng.random() < LABEL_NOISE
        if aflip:
            eff_a = rng.choice([c for c in config.AGE_CLASSES if c != a_cls])
        else:
            eff_a = a_cls
        if eff_a == "15-19":
            a_words, n_low, n_high = _YOUNG_WORDS, 8, 26
        elif eff_a == "20-24":
            a_words, n_low, n_high = _MID_WORDS, 6, 22
        else:
            a_words, n_low, n_high = _OLD_WORDS, 5, 16

        bias_words = g_words + a_words
        # interleave the (weak) bias strength of gender vs age
        text = _emit(bias_words, rng, n_low, n_high,
                     bias_strength=GENDER_BIAS + AGE_BIAS)

        # Surface-level (character) cues, also weak + noisy.
        if eff_a == "15-19" and rng.random() < 0.45:
            text = text + " " + "!" * rng.randint(1, 3)
        if eff_a == "25+" and rng.random() < 0.3:
            text = text + "."
        if eff_g == "female" and rng.random() < 0.25:
            text = text.title()
        if rng.random() < 0.15:  # random casing noise on both
            text = text.upper()

        texts.append(text)
        ages.append(age_idx[a_cls])      # store TRUE labels (noise is in text)
        genders.append(gender_idx[g_cls])

    return Corpus(texts, np.asarray(ages), np.asarray(genders))


def load(root: Optional[str] = None) -> Tuple[Corpus, str]:
    """Load the real corpus if available, else the synthetic surrogate.

    Returns (corpus, source_tag) where source_tag is 'real' or 'synthetic'.
    """
    if root:
        c = load_real(root)
        if c is not None:
            return c, "real"
    return load_synthetic(), "synthetic"
