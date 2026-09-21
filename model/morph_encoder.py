"""
Morfologik Representation Encoder.

Formula:
  h_i^{morph} = W_m * [e_{m_1}^{morph}; ...; e_{m_p}^{morph}] + b_m
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphEncoder(nn.Module):
    """
    Explicit Morfologik Representation Encoder.

    Uses character CNN representations with dedicated projection and attention pooling
    to form an explicit morphological representation vector h_i^{morph} for each word.
    """

    def __init__(self, char_dim: int, d_model: int = 256, dropout: float = 0.3):
        super().__init__()

        # Linear projection from character CNN features to morphological space
        self.proj = nn.Sequential(
            nn.Linear(char_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # Affix attention pooling layer
        self.attn_w = nn.Linear(d_model, 1)

        self.out_dim = d_model

    def forward(self, char_rep: torch.Tensor, word_mask: torch.Tensor = None) -> torch.Tensor:
        """
        char_rep: (B, S, char_dim) — output from CharCNNEncoder
        word_mask: optional boolean mask (B, S)
        Returns h_morph: (B, S, d_model)
        """
        # Project character features: (B, S, d_model)
        h_morph = self.proj(char_rep)

        return h_morph
