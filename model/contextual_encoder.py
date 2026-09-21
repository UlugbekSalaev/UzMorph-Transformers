"""
Contextual Encoders for Uzbek Morphological Analysis.

Supports:
1. CompactTransformerEncoder (Default — L=3, d_model=256, h=4)
2. xLSTMEncoder (Extended sLSTM/mLSTM with exponential gating)
3. BiLSTMEncoder (Baseline option for ablation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, S, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CompactTransformerEncoder(nn.Module):
    """
    Compact Transformer Contextual Encoder.

    Formula:
      H^{(0)} = [e_1, e_2, ..., e_n]
      A^{(l)} = Softmax(Q K^T / sqrt(d_k)) V
      H^{(l)} = LayerNorm(H^{(l-1)} + FFN(LayerNorm(H^{(l-1)} + A^{(l)})))
    """

    def __init__(self, input_dim: int, d_model: int = 256, n_heads: int = 4,
                 d_ff: int = 512, n_layers: int = 3, dropout: float = 0.3):
        super().__init__()

        # Linear projection if input_dim != d_model
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.out_dim = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, S, input_dim)
        mask: boolean padding mask (B, S) where True indicates PADDING
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # PyTorch TransformerEncoder accepts src_key_padding_mask of shape (B, S)
        h_ctx = self.transformer(x, src_key_padding_mask=mask)
        return h_ctx


class xLSTMBlock(nn.Module):
    """
    sLSTM Block with Exponential Gating (simplified xLSTM variant).

    Formula:
      i_t = exp(W_i x_t + U_i h_{t-1} + b_i)
      f_t = exp(W_f x_t + U_f h_{t-1} + b_f)
      z_t = tanh(W_z x_t + U_z h_{t-1} + b_z)
      c_t = f_t * c_{t-1} + i_t * z_t
      n_t = f_t * n_{t-1} + i_t
      h_t = o_t * (c_t / n_t)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input, Forget, Output, Cell candidate gates
        self.w_i = nn.Linear(hidden_dim, hidden_dim)
        self.w_f = nn.Linear(hidden_dim, hidden_dim)
        self.w_o = nn.Linear(hidden_dim, hidden_dim)
        self.w_z = nn.Linear(hidden_dim, hidden_dim)

        self.u_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.u_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.u_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.u_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, S, hidden_dim)"""
        B, S, H = x.shape
        h_t = torch.zeros(B, H, device=x.device)
        c_t = torch.zeros(B, H, device=x.device)
        n_t = torch.ones(B, H, device=x.device)

        outputs = []

        for t in range(S):
            xt = x[:, t, :]

            # Exponential gates with stabilization
            i_logits = self.w_i(xt) + self.u_i(h_t)
            f_logits = self.w_f(xt) + self.u_f(h_t)
            o_logits = self.w_o(xt) + self.u_o(h_t)
            z_logits = self.w_z(xt) + self.u_z(h_t)

            # Stabilized exp gating
            i_t = torch.exp(torch.clamp(i_logits, -10.0, 10.0))
            f_t = torch.exp(torch.clamp(f_logits, -10.0, 10.0))
            o_t = torch.sigmoid(o_logits)
            z_t = torch.tanh(z_logits)

            # Normalization state n_t and cell state c_t
            n_t = f_t * n_t + i_t
            c_t = f_t * c_t + i_t * z_t

            # Output state with normalizer
            h_t = o_t * (c_t / (n_t + 1e-8))
            outputs.append(h_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class xLSTMEncoder(nn.Module):
    """Bidirectional xLSTM Contextual Encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.fwd_blocks = nn.ModuleList([xLSTMBlock(hidden_dim) for _ in range(n_layers)])
        self.bwd_blocks = nn.ModuleList([xLSTMBlock(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.dropout(x)

        # Forward pass
        h_fwd = x
        for block in self.fwd_blocks:
            h_fwd = block(h_fwd)

        # Backward pass
        x_rev = torch.flip(x, dims=[1])
        h_bwd = x_rev
        for block in self.bwd_blocks:
            h_bwd = block(h_bwd)
        h_bwd = torch.flip(h_bwd, dims=[1])

        return torch.cat([h_fwd, h_bwd], dim=-1)


class BiLSTMEncoder(nn.Module):
    """Standard BiLSTM Encoder for baseline ablation."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h_ctx, _ = self.lstm(x)
        return h_ctx
