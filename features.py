"""Stylometric feature extractors.

Three interpretable feature families are computed, mirroring the taxonomy used
in the ABMRF baseline (Arshad et al.) but extended with normalised ratios that
behave well for short SMS text:

  * character-based : counts / ratios of symbols, digits, punctuation, casing
  * word-based      : length statistics, type-token ratio, lexical richness
  * sentence-based  : segmentation statistics, words-per-sentence, etc.

Each extractor returns a fixed-length numeric vector so the families can be
routed to separate adapters in the mixture-of-adapters head.
"""
from __future__ import annotations

import re
import math
from typing import Dict, List

import numpy as np

PUNCT = set(".,;:!?\"'()-[]{}<>/\\|@#$%^&*_+=~`")
VOWELS = set("aeiou")
# Frequent Roman Urdu function words (used as light stop-word style markers).
ROMAN_URDU_FUNCTION_WORDS = {
    "hai", "ka", "ki", "ke", "ko", "se", "me", "mein", "or", "aur", "ya",
    "na", "nahi", "han", "ji", "to", "tu", "tum", "ap", "main", "hum", "wo",
    "ye", "is", "us", "kya", "kyun", "kaise", "acha", "theek", "os",
}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def character_features(text: str) -> Dict[str, float]:
    """Character-level stylometry (length-normalised)."""
    n = len(text)
    if n == 0:
        return {k: 0.0 for k in _CHAR_KEYS}
    digits = sum(c.isdigit() for c in text)
    uppers = sum(c.isupper() for c in text)
    spaces = sum(c.isspace() for c in text)
    puncts = sum(c in PUNCT for c in text)
    vowels = sum(c.lower() in VOWELS for c in text)
    alpha = sum(c.isalpha() for c in text)
    return {
        "char_count": float(n),
        "digit_ratio": _safe_div(digits, n),
        "upper_ratio": _safe_div(uppers, n),
        "space_ratio": _safe_div(spaces, n),
        "punct_ratio": _safe_div(puncts, n),
        "vowel_ratio": _safe_div(vowels, max(alpha, 1)),
        "alpha_ratio": _safe_div(alpha, n),
        "exclaim_ratio": _safe_div(text.count("!"), n),
        "question_ratio": _safe_div(text.count("?"), n),
        "unique_char_ratio": _safe_div(len(set(text)), n),
    }


_CHAR_KEYS = [
    "char_count", "digit_ratio", "upper_ratio", "space_ratio", "punct_ratio",
    "vowel_ratio", "alpha_ratio", "exclaim_ratio", "question_ratio",
    "unique_char_ratio",
]


def word_features(text: str) -> Dict[str, float]:
    """Word-level stylometry."""
    words = re.findall(r"\w+", text.lower())
    nw = len(words)
    if nw == 0:
        return {k: 0.0 for k in _WORD_KEYS}
    lengths = [len(w) for w in words]
    types = set(words)
    fw = sum(w in ROMAN_URDU_FUNCTION_WORDS for w in words)
    short = sum(1 for L in lengths if L <= 3)
    long_ = sum(1 for L in lengths if L >= 7)
    return {
        "word_count": float(nw),
        "avg_word_len": float(np.mean(lengths)),
        "std_word_len": float(np.std(lengths)),
        "type_token_ratio": _safe_div(len(types), nw),
        "hapax_ratio": _safe_div(sum(1 for w in types if words.count(w) == 1), nw),
        "function_word_ratio": _safe_div(fw, nw),
        "short_word_ratio": _safe_div(short, nw),
        "long_word_ratio": _safe_div(long_, nw),
        "lexical_richness": _safe_div(len(types), math.sqrt(nw)),  # Guiraud's R
    }


_WORD_KEYS = [
    "word_count", "avg_word_len", "std_word_len", "type_token_ratio",
    "hapax_ratio", "function_word_ratio", "short_word_ratio",
    "long_word_ratio", "lexical_richness",
]


def sentence_features(text: str) -> Dict[str, float]:
    """Sentence-level stylometry."""
    sents = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    ns = len(sents)
    words = re.findall(r"\w+", text)
    nw = len(words)
    if ns == 0:
        return {k: 0.0 for k in _SENT_KEYS}
    wps = [len(re.findall(r"\w+", s)) for s in sents]
    return {
        "sentence_count": float(ns),
        "avg_words_per_sentence": float(np.mean(wps)),
        "std_words_per_sentence": float(np.std(wps)),
        "max_words_per_sentence": float(np.max(wps)),
        "avg_sentence_len_chars": float(np.mean([len(s) for s in sents])),
        "words_per_sentence_overall": _safe_div(nw, ns),
        "line_count": float(text.count("\n") + 1),
    }


_SENT_KEYS = [
    "sentence_count", "avg_words_per_sentence", "std_words_per_sentence",
    "max_words_per_sentence", "avg_sentence_len_chars",
    "words_per_sentence_overall", "line_count",
]


FEATURE_FAMILIES = {
    "char": (_CHAR_KEYS, character_features),
    "word": (_WORD_KEYS, word_features),
    "sentence": (_SENT_KEYS, sentence_features),
}


def extract_family(texts: List[str], family: str) -> np.ndarray:
    """Return an (N, d_family) matrix for one feature family."""
    keys, fn = FEATURE_FAMILIES[family]
    rows = []
    for t in texts:
        d = fn(t)
        rows.append([d[k] for k in keys])
    return np.asarray(rows, dtype=np.float64)


def extract_all(texts: List[str]) -> Dict[str, np.ndarray]:
    """Return a dict family -> matrix for all three families."""
    return {fam: extract_family(texts, fam) for fam in FEATURE_FAMILIES}


def family_dims() -> Dict[str, int]:
    return {fam: len(keys) for fam, (keys, _) in FEATURE_FAMILIES.items()}
