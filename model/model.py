"""
To'liq Uzbek Morphological Analysis Neural Network Model.

Unified architecture combining:
1. Character-level 1D-CNN Encoder (multi-width kernels)
2. Compact Contextual Transformer / xLSTM / BiLSTM Encoder
3. Morfologik Representation Encoder
4. Adaptive Context-Morphology Gating
5. Multi-Task Classification Heads (POS, Case, Number, Person, Tense, Mood, VerbForm, Voice)
6. Character-level GRU Lemmatizer
7. Morpheme Segmentation Head (BIES)
8. Structured Grammatical Consistency Regularizer
9. Uncertainty-based Multi-Task Loss Weighting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from model.char_encoder import CharCNNEncoder
from model.contextual_encoder import CompactTransformerEncoder, xLSTMEncoder, BiLSTMEncoder
from model.morph_encoder import MorphEncoder
from model.gating import AdaptiveGating
from model.heads import MultiTaskClassificationHeads, LemmaDecoderHead, SegmentationHead
from model.consistency import ConsistencyRegularizer
from model.uncertainty import UncertaintyWeighting
from model.focal_loss import FocalLoss


class UzbekMorphModel(nn.Module):
    """
    Method 1: Joint Neural Morphological Analyzer for Uzbek.
    """

    def __init__(self, cfg, label_vocabs: Dict[str, Dict],
                 compat_matrix: Dict = None,
                 encoder_type: str = 'transformer',
                 use_gating: bool = True,
                 use_consistency: bool = True,
                 use_uncertainty: bool = True):
        super().__init__()

        self.cfg = cfg
        self.label_vocabs = label_vocabs
        self.encoder_type = encoder_type.lower()
        self.use_gating = use_gating
        self.use_consistency = use_consistency
        self.use_uncertainty = use_uncertainty

        # 1. Character Encoder (1D-CNN)
        char_vocab_size = len(label_vocabs['char'])
        self.char_encoder = CharCNNEncoder(
            vocab_size=char_vocab_size,
            embed_dim=cfg.char_embed_dim,
            num_filters=cfg.cnn_filters,
            kernels=cfg.cnn_kernels,
            dropout=cfg.dropout
        )
        char_out_dim = self.char_encoder.out_dim  # 384 (128 * 3)

        # Token Embedding (learnable word embedding / random / pre-trained)
        self.token_embed = nn.Embedding(50000, cfg.token_embed_dim, padding_idx=0)

        # Combined representation dimension before contextual encoder
        combined_dim = cfg.token_embed_dim + char_out_dim  # 128 + 384 = 512

        # 2. Contextual Encoder (Transformer / xLSTM / BiLSTM)
        if self.encoder_type == 'transformer':
            self.contextual_encoder = CompactTransformerEncoder(
                input_dim=combined_dim,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout
            )
        elif self.encoder_type == 'xlstm':
            self.contextual_encoder = xLSTMEncoder(
                input_dim=combined_dim,
                hidden_dim=cfg.d_model // 2,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout
            )
        elif self.encoder_type == 'bilstm':
            self.contextual_encoder = BiLSTMEncoder(
                input_dim=combined_dim,
                hidden_dim=cfg.d_model // 2,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        ctx_dim = self.contextual_encoder.out_dim  # 256

        # 3. Morfologik Representation Encoder
        self.morph_encoder = MorphEncoder(
            char_dim=char_out_dim,
            d_model=cfg.d_model,
            dropout=cfg.dropout
        )

        # 4. Adaptive Gating Mechanism
        if self.use_gating:
            self.gating = AdaptiveGating(
                d_ctx=ctx_dim,
                d_morph=self.morph_encoder.out_dim,
                d_out=cfg.d_model,
                dropout=cfg.dropout
            )
            head_in_dim = cfg.d_model
        else:
            head_in_dim = ctx_dim

        # 5. Multi-Task Classification Heads
        self.classification_heads = MultiTaskClassificationHeads(
            in_dim=head_in_dim,
            label_vocabs=label_vocabs,
            hidden_dim=128,
            dropout=cfg.dropout
        )

        # 6. Lemma Character Decoder
        self.lemma_decoder = LemmaDecoderHead(
            char_vocab_size=char_vocab_size,
            embed_dim=cfg.char_embed_dim,
            hidden_dim=head_in_dim,
            dropout=cfg.dropout
        )

        # 7. Segmentation Head (BIES)
        bies_vocab_size = len(label_vocabs.get('bies', {'<PAD>': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}))
        self.segmentation_head = SegmentationHead(
            char_embed_dim=cfg.char_embed_dim,
            hidden_dim=128,
            num_classes=bies_vocab_size,
            dropout=cfg.dropout
        )

        # 8. Consistency Regularizer
        if self.use_consistency and compat_matrix is not None:
            self.consistency_regularizer = ConsistencyRegularizer(
                compat_matrix=compat_matrix,
                label_vocabs=label_vocabs
            )
        else:
            self.consistency_regularizer = None

        # 9. Uncertainty Weighting
        task_names = list(self.classification_heads.task_names) + ['lemma']
        if self.use_uncertainty:
            self.uncertainty_weighter = UncertaintyWeighting(task_names)
        else:
            self.uncertainty_weighter = None

    def forward(self, char_ids: torch.Tensor,
                token_ids: torch.Tensor = None,
                lengths: torch.Tensor = None,
                target_lemma_chars: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        char_ids: (B, S, W) — character IDs tensor
        token_ids: optional (B, S) — token IDs tensor
        lengths: (B,) — sentence lengths
        target_lemma_chars: optional (B, S, L) — target lemma character IDs for decoder

        Returns dict containing:
          - 'logits': dict of classification logits per task {task_name: (B, S, num_classes)}
          - 'lemma_logits': (B, S, L, char_vocab_size)
          - 'gate': optional (B, S, d_model) — gating activations
          - 'h': (B, S, d_model) — final token representations
        """
        B, S, W = char_ids.shape

        # 1. Character CNN features: (B, S, char_out_dim)
        char_feats = self.char_encoder(char_ids)

        # Token embeddings: if token_ids not provided, use dummy zeros
        if token_ids is None:
            token_ids = torch.zeros(B, S, dtype=torch.long, device=char_ids.device)

        tok_feats = self.token_embed(token_ids)  # (B, S, token_embed_dim)

        # Concatenate character and token features: (B, S, combined_dim)
        combined = torch.cat([tok_feats, char_feats], dim=-1)

        # Padding mask for Transformer (True indicates PADDING)
        padding_mask = None
        if lengths is not None:
            seq_range = torch.arange(S, device=char_ids.device).unsqueeze(0)  # (1, S)
            padding_mask = seq_range >= lengths.unsqueeze(1)  # (B, S)

        # 2. Contextual Encoder: (B, S, ctx_dim)
        h_ctx = self.contextual_encoder(combined, mask=padding_mask)

        # 3. Morfologik Representation Encoder: (B, S, d_model)
        h_morph = self.morph_encoder(char_feats)

        # 4. Adaptive Gating
        if self.use_gating:
            gate_out = self.gating(h_ctx, h_morph)
            h_final = gate_out['h']
            gate_activations = gate_out['gate']
        else:
            h_final = h_ctx
            gate_activations = None

        # 5. Multi-Task Classification Predictions
        class_logits = self.classification_heads(h_final)

        # 6. Lemma Decoder Predictions
        lemma_logits = self.lemma_decoder(
            h=h_final,
            target_lemma_chars=target_lemma_chars
        )

        outputs = {
            'logits': class_logits,
            'lemma_logits': lemma_logits,
            'gate': gate_activations,
            'h': h_final,
            'padding_mask': padding_mask,
        }

        return outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss including cross-entropy losses, consistency loss, and uncertainty weighting.
        """
        logits = outputs['logits']
        lemma_logits = outputs['lemma_logits']
        padding_mask = outputs['padding_mask']

        task_losses = {}
        total_unweighted = 0.0

        # Mask out padding tokens: valid mask (B, S) where True means VALID
        if padding_mask is not None:
            valid_mask = ~padding_mask
        else:
            valid_mask = torch.ones_like(targets['pos'], dtype=torch.bool)

        for task_name, task_logit in logits.items():
            if task_name in targets:
                target = targets[task_name]
                B, S, C = task_logit.shape
                
                # Dynamic Alpha generation modifying Focal Loss distributions
                # Significantly limits (0.1 weight) the <NONE> classification (Index 1) preventing Negative Transfer collapse
                alpha_w = torch.ones(C, device=task_logit.device)
                if C > 1:
                    alpha_w[1] = 0.1
                
                focal_fn = FocalLoss(gamma=2.0, alpha=alpha_w, ignore_index=0)
                logit_flat = task_logit.view(B * S, C)
                target_flat = target.view(B * S)

                loss = focal_fn(logit_flat, target_flat)
                
                # Zero out loss strictly for <UNK> elements to prevent unannotated-gap poisoning
                unk_mask_flat = (target_flat != 1).float()
                loss = loss * unk_mask_flat
                
                # Average loss only over valid explicit predictions avoiding tensor explosions
                valid_count = unk_mask_flat.sum() + 1e-6
                loss = loss.sum() / valid_count
                
                task_losses[task_name] = loss
                total_unweighted = total_unweighted + loss

        # 2. Lemma Decoder loss
        if 'lemma_chars' in targets:
            lemma_targets = targets['lemma_chars']  # (B, S, L)
            B, S, L, V = lemma_logits.shape

            loss_fn_lemma = nn.CrossEntropyLoss(ignore_index=0)
            lemma_logit_flat = lemma_logits.view(B * S * L, V)
            lemma_target_flat = lemma_targets.view(B * S * L)

            lemma_loss = loss_fn_lemma(lemma_logit_flat, lemma_target_flat)
            task_losses['lemma'] = lemma_loss
            total_unweighted = total_unweighted + lemma_loss

        # 3. Consistency loss
        cons_loss = torch.tensor(0.0, device=lemma_logits.device)
        if self.use_consistency and self.consistency_regularizer is not None:
            cons_loss = self.consistency_regularizer(logits, mask=valid_mask)

        # 4. Total weighted loss
        if self.use_uncertainty and self.uncertainty_weighter is not None:
            total_loss = self.uncertainty_weighter(
                task_losses,
                alpha_cons=self.cfg.alpha_cons,
                cons_loss=cons_loss
            )
        else:
            total_loss = total_unweighted + self.cfg.alpha_cons * cons_loss

        return {
            'total_loss': total_loss,
            'task_losses': task_losses,
            'cons_loss': cons_loss,
        }
