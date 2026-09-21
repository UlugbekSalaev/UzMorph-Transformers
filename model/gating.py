"""
Adaptive Gating Mechanism for Uzbek Morphological Analysis.

Formulas:
  g_i = sigmoid(W_g * [h_i^{ctx}; h_i^{morph}] + b_g)
  h_i = g_i (odot) h_i^{ctx} + (1 - g_i) (odot) h_i^{morph}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGating(nn.Module):
    """
    Adaptive Context-Morphology Gating Mechanism.

    Dynamically balances contextual representation h^{ctx} and explicit
    morphological representation h^{morph} for each word token.

    - g_i approx 1: token prediction relies more on sentence context (e.g. homonyms)
    - g_i approx 0: token prediction relies more on affix morphology (e.g. inflected forms)
    """

    def __init__(self, d_ctx: int = 256, d_morph: int = 256, d_out: int = 256,
                 dropout: float = 0.2):
        super().__init__()

        # Linear projections to equalize dimensions if needed
        self.proj_ctx = nn.Linear(d_ctx, d_out) if d_ctx != d_out else nn.Identity()
        self.proj_morph = nn.Linear(d_morph, d_out) if d_morph != d_out else nn.Identity()

        # Gate network: takes concat([h_ctx, h_morph]) -> g_i in [0, 1]^d_out
        self.gate_net = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, h_ctx: torch.Tensor, h_morph: torch.Tensor) -> dict:
        """
        h_ctx: (B, S, d_ctx)
        h_morph: (B, S, d_morph)

        Returns dict:
          'h': (B, S, d_out) — gated combined representation
          'gate': (B, S, d_out) — gating activations for analysis/visualization
        """
        h_c = self.proj_ctx(h_ctx)
        h_m = self.proj_morph(h_morph)

        # Compute gating vector g_i: (B, S, d_out)
        combined_features = torch.cat([h_c, h_m], dim=-1)
        g = self.gate_net(combined_features)

        # Gated fusion: h_i = g * h_ctx + (1 - g) * h_morph
        h = g * h_c + (1.0 - g) * h_m
        h = self.layer_norm(h)

        return {
            'h': h,
            'gate': g
        }
