"""SAFT-MoA full model (PyTorch / HuggingFace transformers).

Architecture
------------
1.  A frozen multilingual transformer encoder (xlm-roberta-base) produces a
    contextual sentence embedding ``h_cls`` for each SMS document.
2.  Three stylometric family vectors (char / word / sentence) are concatenated
    and projected to a dense stylometric code ``s`` by a small MLP.
3.  A Mixture-of-Adapters head adapts ``h_cls`` using ``s`` to route between
    family-specialised adapters, yielding ``h_moa``.
4.  ``h_moa`` and ``s`` are fused (gated residual) and passed to two linear
    classification heads (age, gender) trained jointly (multi-task).

Only the adapters, projection, gate, fusion and heads are trained, which keeps
the trainable parameter count under ~3% of the backbone (PEFT).

Requires ``torch`` and ``transformers``.
"""
from __future__ import annotations

from typing import Dict, Optional

from .config import ModelConfig
from .adapters import _HAS_TORCH

if _HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .adapters import MixtureOfAdapters

    try:
        from transformers import AutoModel, AutoTokenizer
        _HAS_HF = True
    except Exception:  # pragma: no cover
        _HAS_HF = False

    class StyleProjector(nn.Module):
        """Project concatenated stylometric families to a dense code."""

        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 2, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    class SAFTMoA(nn.Module):
        def __init__(self, cfg: ModelConfig, style_in_dim: int,
                     n_age: int, n_gender: int):
            super().__init__()
            if not _HAS_HF:
                raise ImportError("transformers is required for SAFT-MoA.")
            self.cfg = cfg
            self.encoder = AutoModel.from_pretrained(cfg.backbone)
            if cfg.freeze_backbone:
                for p in self.encoder.parameters():
                    p.requires_grad = False
            H = cfg.hidden_size
            self.style_proj = StyleProjector(style_in_dim, cfg.style_dim, cfg.dropout)
            self.moa = MixtureOfAdapters(
                hidden_size=H,
                bottleneck=cfg.adapter_bottleneck,
                n_adapters=cfg.n_adapters,
                style_dim=cfg.style_dim,
                dropout=cfg.dropout,
                tau=cfg.gate_temperature,
            )
            # Gated residual fusion of contextual + stylometric signals.
            self.fuse_gate = nn.Linear(H + cfg.style_dim, H)
            self.fuse_proj = nn.Linear(cfg.style_dim, H)
            self.norm = nn.LayerNorm(H)
            self.dropout = nn.Dropout(cfg.dropout)
            self.head_age = nn.Linear(H, n_age)
            self.head_gender = nn.Linear(H, n_gender)

        def encode(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # mean pooling over valid tokens
            mask = attention_mask.unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-6)
            return summed / counts

        def forward(self, input_ids, attention_mask, style_feats) -> Dict[str, "torch.Tensor"]:
            h = self.encode(input_ids, attention_mask)        # (B, H)
            s = self.style_proj(style_feats)                  # (B, style_dim)
            h_moa, gate = self.moa(h, s)                      # (B, H), (B, K)
            # gated residual fusion
            g = torch.sigmoid(self.fuse_gate(torch.cat([h_moa, s], dim=-1)))
            fused = self.norm(h_moa + g * self.fuse_proj(s))
            fused = self.dropout(fused)
            return {
                "age_logits": self.head_age(fused),
                "gender_logits": self.head_gender(fused),
                "gate": gate,
            }

    def multitask_loss(out, age_labels, gender_labels, label_smoothing=0.05):
        la = F.cross_entropy(out["age_logits"], age_labels,
                             label_smoothing=label_smoothing)
        lg = F.cross_entropy(out["gender_logits"], gender_labels,
                             label_smoothing=label_smoothing)
        return la + lg, la, lg

else:  # pragma: no cover
    class SAFTMoA:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch + transformers required for the full "
                              "SAFT-MoA model. Use run_experiments.py for the "
                              "CPU reproducible pipeline.")
