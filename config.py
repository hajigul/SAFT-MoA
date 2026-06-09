"""Central configuration for SAFT-MoA experiments."""
from dataclasses import dataclass, field
from typing import List


SEED = 42

# ----- Dataset --------------------------------------------------------------
# FIRE'18-MAPonSMS: 350 training instances, 150 test instances.
# Age groups: 15-19, 20-24, 25+ ; Gender: male / female.
AGE_CLASSES: List[str] = ["15-19", "20-24", "25+"]
GENDER_CLASSES: List[str] = ["male", "female"]

# Class counts reported by Arshad et al. (used for the synthetic fallback so
# that the generated corpus matches the real class imbalance).
AGE_COUNTS = {"15-19": 108, "20-24": 176, "25+": 66}
GENDER_COUNTS = {"male": 210, "female": 140}
N_TRAIN = 350
N_TEST = 150


# ----- Transformer backbone (full mode) ------------------------------------
@dataclass
class ModelConfig:
    backbone: str = "xlm-roberta-base"   # multilingual, covers Roman Urdu
    max_length: int = 64                 # SMS are short
    hidden_size: int = 768
    adapter_bottleneck: int = 64         # PEFT bottleneck dimension
    n_adapters: int = 3                  # char / word / sentence experts
    style_dim: int = 64                  # projected stylometric vector size
    dropout: float = 0.1
    gate_temperature: float = 1.0
    freeze_backbone: bool = True         # parameter-efficient fine-tuning


# ----- Training -------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    n_folds: int = 10                    # 10-fold CV, as in the baseline paper
    label_smoothing: float = 0.05
    seed: int = SEED


# ----- Feature extraction ---------------------------------------------------
@dataclass
class FeatureConfig:
    char_ngram_range: tuple = (1, 4)
    char_max_features: int = 4000
    word_ngram_range: tuple = (1, 2)
    word_max_features: int = 4000
    use_stylometry: bool = True


MODEL = ModelConfig()
TRAIN = TrainConfig()
FEATURES = FeatureConfig()
