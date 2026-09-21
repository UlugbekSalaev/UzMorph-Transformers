"""
Structured Grammatical Consistency Regularizer.

Formula:
  L_cons = - sum_i sum_{c_a, c_b} P(y_i^a = c_a) * P(y_i^b = c_b) * log C^{(a,b)}_{c_a, c_b}

Enforces morphological rules (e.g. VERB + Case=Ablative is invalid, NOUN + Tense=Past is invalid).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Set


class ConsistencyRegularizer(nn.Module):
    """
    Grammatical Consistency Regularizer using Soft Compatibility Matrices.
    """

    def __init__(self, compat_matrix: Dict[Tuple[str, str], Set[Tuple[str, str]]],
                 label_vocabs: Dict[str, Dict], eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.compat_matrices = {}

        # Precompute compatibility tensor for each feature pair
        for (feat_a, feat_b), valid_combos in compat_matrix.items():
            if feat_a not in label_vocabs or feat_b not in label_vocabs:
                continue

            vocab_a = label_vocabs[feat_a]
            vocab_b = label_vocabs[feat_b]
            dim_a = len(vocab_a)
            dim_b = len(vocab_b)

            # Matrix C: (dim_a, dim_b) initialized to eps (invalid)
            C = torch.full((dim_a, dim_b), fill_value=eps)

            # Fill valid combinations with 1.0
            for val_a, val_b in valid_combos:
                idx_a = vocab_a.get(val_a, None)
                idx_b = vocab_b.get(val_b, None)
                if idx_a is not None and idx_b is not None:
                    C[idx_a, idx_b] = 1.0

            # Special case: <PAD> and <NONE> are always compatible with everything
            for special in ['<PAD>', '<NONE>']:
                if special in vocab_a:
                    C[vocab_a[special], :] = 1.0
                if special in vocab_b:
                    C[:, vocab_b[special]] = 1.0

            # Register as buffer so it moves to GPU with module
            buffer_name = f"C_{feat_a}_{feat_b}"
            self.register_buffer(buffer_name, C)
            self.compat_matrices[(feat_a, feat_b)] = buffer_name

    def forward(self, predictions: Dict[str, torch.Tensor],
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        predictions: dict of logits {task_name: (B, S, num_classes), ...}
        mask: optional (B, S) boolean tensor where True means valid token (not PAD)

        Returns scalar consistency loss tensor.
        """
        total_cons_loss = 0.0
        num_pairs = 0

        for (feat_a, feat_b), buffer_name in self.compat_matrices.items():
            if feat_a not in predictions or feat_b not in predictions:
                continue

            logits_a = predictions[feat_a]  # (B, S, dim_a)
            logits_b = predictions[feat_b]  # (B, S, dim_b)

            probs_a = F.softmax(logits_a, dim=-1)  # (B, S, dim_a)
            probs_b = F.softmax(logits_b, dim=-1)  # (B, S, dim_b)

            C_mat = getattr(self, buffer_name)  # (dim_a, dim_b)
            log_C = torch.log(C_mat + 1e-10)    # (dim_a, dim_b)

            # Compute joint probability P(a) * P(b): (B, S, dim_a, dim_b)
            # P_joint[b, s, i, j] = probs_a[b, s, i] * probs_b[b, s, j]
            probs_a_exp = probs_a.unsqueeze(-1)  # (B, S, dim_a, 1)
            probs_b_exp = probs_b.unsqueeze(-2)  # (B, S, 1, dim_b)

            joint_probs = probs_a_exp * probs_b_exp  # (B, S, dim_a, dim_b)

            # Expected penalty: - sum_{i,j} joint_probs_{i,j} * log_C_{i,j}
            loss_per_token = -torch.sum(joint_probs * log_C, dim=(-2, -1))  # (B, S)

            if mask is not None:
                loss_per_token = loss_per_token * mask.float()
                denom = mask.float().sum() + 1e-8
                pair_loss = loss_per_token.sum() / denom
            else:
                pair_loss = loss_per_token.mean()

            total_cons_loss = total_cons_loss + pair_loss
            num_pairs += 1

        if num_pairs > 0:
            return total_cons_loss / num_pairs
        return torch.tensor(0.0, device=next(self.parameters()).device)
