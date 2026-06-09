"""scikit-learn reproductions of the baseline classifiers from the paper.

The original paper benchmarks: NB, NB-Updatable, J48, KNN (IBK), RF, CHIRP,
AdaBoostM1, and the ABMRF ensemble. We map each to its closest scikit-learn
equivalent so the comparison is reproducible end-to-end:

    NB            -> GaussianNB
    NB-Updatable  -> BernoulliNB        (online-updatable NB variant)
    J48           -> DecisionTreeClassifier (C4.5-style)
    KNN / IBK     -> KNeighborsClassifier
    RF            -> RandomForestClassifier
    CHIRP         -> NearestCentroid     (composite hypercube proxy)
    AdaBoostM1    -> AdaBoostClassifier(SAMME)
    ABMRF         -> AdaBoost over RandomForest (the prior-art ensemble)
"""
from __future__ import annotations

from typing import Dict

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)

from . import config


def make_baselines() -> Dict[str, object]:
    """Return a dict name -> fresh estimator (deterministic where possible)."""
    seed = config.SEED
    return {
        "NB": GaussianNB(),
        "NB Updatable": BernoulliNB(),
        "J48": DecisionTreeClassifier(criterion="entropy", random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF": RandomForestClassifier(n_estimators=300, random_state=seed,
                                     n_jobs=-1),
        "CHIRP": NearestCentroid(),
        "AdaBoostM1": AdaBoostClassifier(n_estimators=100, random_state=seed),
        "ABMRF": AdaBoostClassifier(
            estimator=RandomForestClassifier(n_estimators=40, max_depth=8,
                                             random_state=seed, n_jobs=-1),
            n_estimators=12, random_state=seed),
    }
