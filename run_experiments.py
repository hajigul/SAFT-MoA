"""End-to-end reproducible experiment runner (CPU).

Runs all baselines and the SAFT-MoA model across four feature regimes
(character / word / sentence / combination) under 10-fold cross-validation and
writes a ``results.json`` consumed by the plotting scripts.

In this lightweight mode the SAFT-MoA contextual encoder is replaced by a
TF-IDF character-n-gram surrogate (a strong, well-known proxy for sub-word
contextual signal on short Roman-Urdu text), while the mixture-of-adapters
fusion head is realised as a calibrated stacked ensemble over the three
stylometric family branches plus the n-gram branch. This keeps the *inductive
structure* of SAFT-MoA (per-family experts + learned fusion) while remaining
runnable without a GPU. The full neural model lives in ``model.py``/``train.py``.

Usage:
    python -m saft_moa.run_experiments --out ../results [--data /path/to/FIRE18]
"""
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from . import config, data
from .features import extract_family, extract_all
from .baselines import make_baselines
from .metrics import compute_metrics, percentage_difference


# ---------------------------------------------------------------------------
# Feature regimes
# ---------------------------------------------------------------------------
FEATURE_REGIMES = ["character", "word", "sentence", "combination"]


def stylometry_matrix(texts: List[str], regime: str) -> np.ndarray:
    if regime == "character":
        return extract_family(texts, "char")
    if regime == "word":
        return extract_family(texts, "word")
    if regime == "sentence":
        return extract_family(texts, "sentence")
    fam = extract_all(texts)
    return np.concatenate([fam["char"], fam["word"], fam["sentence"]], axis=1)


# ---------------------------------------------------------------------------
# SAFT-MoA CPU surrogate
# ---------------------------------------------------------------------------
class SAFTMoASurrogate(BaseEstimator, ClassifierMixin):
    """CPU surrogate that mirrors SAFT-MoA's per-family-expert + fusion design.

    Branches:
      * char-ngram TF-IDF  -> LinearSVC   (proxy for the contextual encoder)
      * stylometric family -> RandomForest experts
    A logistic-regression meta-learner fuses out-of-fold branch predictions
    (stacking), echoing the learned mixture-of-adapters gate.
    """

    def __init__(self, regime: str = "combination", seed: int = config.SEED):
        self.regime = regime
        self.seed = seed

    def _build(self):
        char_branch = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb",
                                      ngram_range=config.FEATURES.char_ngram_range,
                                      max_features=config.FEATURES.char_max_features)),
            ("svm", LinearSVC(C=1.0, random_state=self.seed)),
        ])
        style_branch = Pipeline([
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=None,
                                          random_state=self.seed, n_jobs=-1)),
        ])
        return char_branch, style_branch

    def fit(self, texts, y):
        self.classes_ = np.unique(y)
        char_branch, style_branch = self._build()
        S = stylometry_matrix(list(texts), self.regime)
        # Out-of-fold branch probabilities -> meta features (stacking).
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
        # char branch uses decision_function -> convert to pseudo-proba
        char_oof = cross_val_predict(char_branch, list(texts), y, cv=skf,
                                     method="decision_function")
        if char_oof.ndim == 1:
            char_oof = np.vstack([-char_oof, char_oof]).T
        style_oof = cross_val_predict(style_branch, S, y, cv=skf,
                                      method="predict_proba")
        meta_X = np.hstack([char_oof, style_oof])
        self.meta_ = LogisticRegression(max_iter=2000, C=2.0,
                                        random_state=self.seed)
        self.meta_.fit(meta_X, y)
        # refit branches on full data
        self.char_ = clone(char_branch).fit(list(texts), y)
        self.style_ = clone(style_branch).fit(S, y)
        return self

    def _meta_features(self, texts):
        S = stylometry_matrix(list(texts), self.regime)
        cf = self.char_.decision_function(list(texts))
        if cf.ndim == 1:
            cf = np.vstack([-cf, cf]).T
        sp = self.style_.predict_proba(S)
        return np.hstack([cf, sp])

    def predict(self, texts):
        return self.meta_.predict(self._meta_features(texts))


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------
def eval_baseline(est, X, y, n_folds, seed) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(est, X, y, cv=skf)
    return compute_metrics(y, pred)


def eval_surrogate(texts, y, regime, n_folds, seed) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    texts = np.asarray(texts, dtype=object)
    y = np.asarray(y)
    oof = np.zeros_like(y)
    for tr, va in skf.split(texts, y):
        m = SAFTMoASurrogate(regime=regime, seed=seed).fit(texts[tr], y[tr])
        oof[va] = m.predict(texts[va])
    return compute_metrics(y, oof)


def run(out_dir: str, data_root: str | None = None) -> Dict:
    corpus, source = data.load(data_root)
    n_folds = config.TRAIN.n_folds
    seed = config.SEED

    results = {"source": source, "n_instances": len(corpus),
               "age": {}, "gender": {}}

    for target in ("age", "gender"):
        y = corpus.age if target == "age" else corpus.gender
        for regime in FEATURE_REGIMES:
            X = stylometry_matrix(corpus.texts, regime)
            X = StandardScaler().fit_transform(X)
            regime_block = {}
            for name, est in make_baselines().items():
                try:
                    regime_block[name] = eval_baseline(est, X, y, n_folds, seed)
                except Exception as e:  # robustness on degenerate folds
                    regime_block[name] = {"accuracy": 0.0, "precision": 0.0,
                                          "recall": 0.0, "f1": 0.0, "mcc": 0.0,
                                          "error": str(e)}
            # SAFT-MoA surrogate (uses raw text + stylometry)
            regime_block["SAFT-MoA"] = eval_surrogate(
                corpus.texts, y, regime, n_folds, seed)
            results[target][regime] = regime_block
            best = max(regime_block.items(), key=lambda kv: kv[1]["accuracy"])
            print(f"[{target:6s}/{regime:11s}] best = {best[0]:12s} "
                  f"acc={best[1]['accuracy']*100:.2f}%  "
                  f"SAFT-MoA acc={regime_block['SAFT-MoA']['accuracy']*100:.2f}%")

    # Percentage-difference tables (SAFT-MoA vs each baseline, combination regime)
    pd_tables = {}
    for target in ("age", "gender"):
        comb = results[target]["combination"]
        ref = comb["SAFT-MoA"]["accuracy"] * 100
        pd_tables[target] = {
            name: percentage_difference(ref, comb[name]["accuracy"] * 100)
            for name in comb if name != "SAFT-MoA"
        }
    results["percentage_difference"] = pd_tables

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {out_path}  (source={source})")
    return results


def main():
    ap = argparse.ArgumentParser(description="Run SAFT-MoA experiments")
    ap.add_argument("--out", default="../results", help="output directory")
    ap.add_argument("--data", default=None,
                    help="path to FIRE'18-MAPonSMS root (optional)")
    args = ap.parse_args()
    run(args.out, args.data)


if __name__ == "__main__":
    main()
