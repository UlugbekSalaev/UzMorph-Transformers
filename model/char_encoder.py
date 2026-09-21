"""
Character-level 1D-CNN Encoder for Uzbek Words.

Formula:
  c_ij = Embed(char_ij)
  h_ij^{(k)} = ReLU(W^{(k)} * [c_{i,j}; ...; c_{i,j+k-1}] + b^{(k)}),  k in {3, 5, 7}
  e_i^{char} = [max_j h_ij^{(3)}; max_j h_ij^{(5)}; max_j h_ij^{(7)}]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNNEncoder(nn.Module):
    """
    Multi-width 1D-CNN Character Encoder.

    Input: char_ids tensor of shape (B, S, W) or (N, W)
      - B: batch size
      - S: sequence length (number of words in sentence)
      - W: max word length (number of characters in word)

    Output: character representation tensor of shape (B, S, num_kernels * num_filters)
            or (N, num_kernels * num_filters)
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 num_filters: int = 128, kernels: list = [3, 5, 7],
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # 1D Convolution layers for each kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2  # same padding
            )
            for k in kernels
        ])

        self.out_dim = len(kernels) * num_filters

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        char_ids can be 2D (N, W) or 3D (B, S, W).
        """
        is_3d = (char_ids.dim() == 3)
        if is_3d:
            B, S, W = char_ids.shape
            # Flatten to (B*S, W)
            char_ids = char_ids.view(B * S, W)
        else:
            N, W = char_ids.shape

        # Embedding: (N, W) -> (N, W, embed_dim)
        x = self.embedding(char_ids)
        x = self.dropout(x)

        # Transpose for Conv1d: (N, embed_dim, W)
        x = x.transpose(1, 2)

        # Apply multi-kernel Conv1d + ReLU + Global Max Pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv output: (N, num_filters, W)
            h = F.relu(conv(x))
            # Global max pooling over character sequence: (N, num_filters)
            pooled, _ = torch.max(h, dim=2)
            conv_outputs.append(pooled)

        # Concatenate features from all kernel sizes: (N, num_kernels * num_filters)
        char_rep = torch.cat(conv_outputs, dim=-1)

        if is_3d:
            # Reshape back to (B, S, out_dim)
            char_rep = char_rep.view(B, S, self.out_dim)

        return char_rep
