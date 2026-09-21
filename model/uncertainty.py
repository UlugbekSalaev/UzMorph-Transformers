"""
Uncertainty-Based Multi-Task Loss Weighting.

Formula:
  L_{weighted} = sum_{k=1}^K (exp(-s_k) * L_k + 0.5 * s_k) + alpha * L_cons + beta * L_align
  where s_k = log(sigma_k^2) is a learnable parameter per task.
"""
import torch
import torch.nn as nn
from typing import Dict, List


class UncertaintyWeighting(nn.Module):
    """
    Automatic multi-task loss weighting using homoscedastic task uncertainty (Kendall & Gal, 2018).
    """

    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names

        # Learnable log variance parameters s_k = log(sigma_k^2)
        # Initialized to 0.0 (so sigma_k^2 = 1, exp(-s_k) = 1)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.tensor(0.0)) for task in task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor],
                alpha_cons: float = 0.1, cons_loss: torch.Tensor = None,
                beta_align: float = 0.05, align_loss: torch.Tensor = None) -> torch.Tensor:
        """
        losses: dict of task losses {task_name: scalar loss tensor}
        cons_loss: optional consistency loss tensor
        align_loss: optional morpheme alignment loss tensor

        Returns total weighted multi-task loss tensor.
        """
        total_loss = 0.0

        for task, loss_val in losses.items():
            if task in self.log_vars:
                # Clamp log variance to prevent Bayesian collapse (-3.0 to 3.0 bounds)
                s_k = torch.clamp(self.log_vars[task], min=-3.0, max=3.0)
                # Weighted loss: exp(-s_k) * L_k + 0.5 * s_k
                weighted_task_loss = torch.exp(-s_k) * loss_val + 0.5 * s_k
                total_loss = total_loss + weighted_task_loss
            else:
                total_loss = total_loss + loss_val

        # Add consistency regularization loss
        if cons_loss is not None:
            total_loss = total_loss + alpha_cons * cons_loss

        # Add morpheme alignment loss
        if align_loss is not None:
            total_loss = total_loss + beta_align * align_loss

        return total_loss
