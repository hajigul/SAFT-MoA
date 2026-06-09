"""Adapter and Mixture-of-Adapters (MoA) modules for SAFT-MoA.

These are the parameter-efficient fine-tuning (PEFT) building blocks. Each
adapter is a Houlsby-style bottleneck. The MoA head holds one adapter per
stylometric family (character / word / sentence) plus a learned gate that
routes the fused representation, conditioned on the projected stylometric
vector.

Requires PyTorch. If torch is not installed this module raises a clear
ImportError when used; the lightweight reproducible pipeline in
``run_experiments.py`` does not import it.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    _HAS_TORCH = False


if _HAS_TORCH:

    class Adapter(nn.Module):
        """Houlsby-style bottleneck adapter with a residual connection.

            h' = h + W_up( GELU( W_down( LN(h) ) ) )
        """

        def __init__(self, hidden_size: int, bottleneck: int, dropout: float = 0.1):
            super().__init__()
            self.ln = nn.LayerNorm(hidden_size)
            self.down = nn.Linear(hidden_size, bottleneck)
            self.up = nn.Linear(bottleneck, hidden_size)
            self.dropout = nn.Dropout(dropout)
            nn.init.normal_(self.down.weight, std=1e-3)
            nn.init.zeros_(self.down.bias)
            nn.init.normal_(self.up.weight, std=1e-3)
            nn.init.zeros_(self.up.bias)

        def forward(self, h: "torch.Tensor") -> "torch.Tensor":
            z = self.ln(h)
            z = F.gelu(self.down(z))
            z = self.up(self.dropout(z))
            return h + z

    class MixtureOfAdapters(nn.Module):
        """Gated mixture of family-specialised adapters.

        Given the transformer pooled state ``h`` and a projected stylometric
        vector ``s``, each adapter produces a candidate representation. A gate
        network conditioned on ``[h ; s]`` yields routing weights ``g`` over the
        adapters (softmax with temperature ``tau``). The output is the convex
        combination ``sum_k g_k * adapter_k(h)``.
        """

        def __init__(self, hidden_size: int, bottleneck: int, n_adapters: int,
                     style_dim: int, dropout: float = 0.1, tau: float = 1.0):
            super().__init__()
            self.tau = tau
            self.adapters = nn.ModuleList(
                [Adapter(hidden_size, bottleneck, dropout) for _ in range(n_adapters)]
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_size + style_dim, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, n_adapters),
            )

        def forward(self, h: "torch.Tensor", s: "torch.Tensor"):
            gate_logits = self.gate(torch.cat([h, s], dim=-1)) / self.tau
            g = F.softmax(gate_logits, dim=-1)                      # (B, K)
            outs = torch.stack([a(h) for a in self.adapters], dim=1)  # (B,K,H)
            mixed = (g.unsqueeze(-1) * outs).sum(dim=1)             # (B, H)
            return mixed, g

else:  # pragma: no cover

    class Adapter:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch is required for SAFT-MoA adapters. "
                              "Install torch or use run_experiments.py.")

    class MixtureOfAdapters:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch is required for SAFT-MoA adapters. "
                              "Install torch or use run_experiments.py.")
