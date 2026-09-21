r"""
Class-Weighted Focal Loss for Multi-Task Imbalanced Morphological Attribute Classification.

Mathematical formulation:
    FL(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance in multi-task morphological feature prediction.
    Focuses learning on hard negative/positive examples by applying a modulating factor (1 - p_t)^gamma.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = 0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) or (B, S, C) prediction logits
            targets: (N,) or (B, S) target class indices
        """
        if logits.dim() > 2:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        # Mask ignored indices (PAD / NONE)
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather probability of true target class
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_factor = (1.0 - target_probs) ** self.gamma
        loss = -focal_factor * target_log_probs

        if self.alpha is not None:
            alpha_weights = self.alpha.to(logits.device).gather(dim=0, index=targets)
            loss = alpha_weights * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
