"""
Multi-Task Prediction Heads for Uzbek Morphological Analysis.

Includes:
1. ClassificationHeads: POS, Case, Number, Person, Tense, Mood, VerbForm, Voice
2. LemmaDecoderHead: Character-level GRU decoder for lemmatization
3. SegmentationHead: Character-level BIES classification for morpheme boundary detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SingleClassificationHead(nn.Module):
    """Single MLP classification head with LayerNorm + Dropout."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (B, S, in_dim) -> logits: (B, S, num_classes)"""
        return self.net(x)


class MultiTaskClassificationHeads(nn.Module):
    """
    Multi-Task Classification Heads for morphological attributes.
    """

    def __init__(self, in_dim: int, label_vocabs: Dict[str, Dict], hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.task_names = []

        for task_name, vocab in label_vocabs.items():
            if task_name in ['char', 'bies']:
                continue
            self.task_names.append(task_name)
            self.heads[task_name] = SingleClassificationHead(
                in_dim=in_dim,
                num_classes=len(vocab),
                hidden_dim=hidden_dim,
                dropout=dropout
            )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        h: (B, S, in_dim)
        Returns dict of logits: {task_name: (B, S, num_classes), ...}
        """
        logits = {}
        for task_name in self.task_names:
            logits[task_name] = self.heads[task_name](h)
        return logits


class LemmaDecoderHead(nn.Module):
    """
    Character-level GRU Decoder for Lemmatization.

    Takes word representation vector h_i and character embeddings of word form,
    decodes lemma character by character: P(l_i | x_i) = prod_t P(l_it | l_{i,<t}, h_i)
    """

    def __init__(self, char_vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.proj_init = nn.Linear(hidden_dim, hidden_dim)

        self.gru = nn.GRUCell(embed_dim + hidden_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, char_vocab_size)
        )

        self.hidden_dim = hidden_dim
        self.char_vocab_size = char_vocab_size

    def forward(self, h: torch.Tensor, target_lemma_chars: torch.Tensor = None,
                max_len: int = 30) -> torch.Tensor:
        """
        h: (B, S, hidden_dim) — token representation
        target_lemma_chars: optional (B, S, L) target character IDs for teacher forcing

        Returns logits: (B, S, L, char_vocab_size)
        """
        B, S, H = h.shape

        # Flatten (B, S, H) -> (B*S, H)
        h_flat = h.view(B * S, H)

        if target_lemma_chars is not None:
            L = target_lemma_chars.size(2)
            target_flat = target_lemma_chars.view(B * S, L)
        else:
            L = max_len
            target_flat = None

        # Initial hidden state derived from token vector
        gru_h = torch.tanh(self.proj_init(h_flat))

        # Initial input character: <BOS> (index 2) or zero
        char_in = torch.full((B * S,), fill_value=2, dtype=torch.long, device=h.device)

        logits_list = []

        for t in range(L):
            char_emb = self.char_embed(char_in)  # (B*S, embed_dim)
            gru_in = torch.cat([char_emb, h_flat], dim=-1)  # (B*S, embed_dim + H)

            gru_h = self.gru(gru_in, gru_h)
            logit_t = self.out(gru_h)  # (B*S, char_vocab_size)
            logits_list.append(logit_t.unsqueeze(1))  # (B*S, 1, V)

            if target_flat is not None:
                # Teacher forcing
                char_in = target_flat[:, t]
            else:
                # Greedy decoding
                char_in = torch.argmax(logit_t, dim=-1)

        # Concatenate time steps: (B*S, L, V)
        all_logits = torch.cat(logits_list, dim=1)

        # Reshape back to (B, S, L, V)
        return all_logits.view(B, S, L, self.char_vocab_size)


class SegmentationHead(nn.Module):
    """
    Character-level Morpheme Segmentation Classifier (BIES tags).

    Label scheme:
      B: Begin morpheme
      I: Inside morpheme
      E: End morpheme
      S: Single-char morpheme
    """

    def __init__(self, char_embed_dim: int = 128, hidden_dim: int = 128,
                 num_classes: int = 5, dropout: float = 0.2):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=char_embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, char_embeddings: torch.Tensor) -> torch.Tensor:
        """
        char_embeddings: (N, W, char_embed_dim) where N is batch size (words), W is max word length
        Returns logits: (N, W, num_classes)
        """
        h_lstm, _ = self.bilstm(char_embeddings)
        logits = self.classifier(h_lstm)
        return logits
