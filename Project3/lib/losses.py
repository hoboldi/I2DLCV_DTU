import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        intersection = torch.mean(2.0*torch.mul(y_true,y_pred)+1.0)
        sum = torch.mean(y_pred+y_true) + 1.0
        return 1.0-intersection/sum

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean", ignore_index: Optional[int] = None,
                 max_labeled_points: Optional[int] = None,
                 raise_on_too_many: bool = False):
        super().__init__()
        assert reduction in ("none", "mean", "sum")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        # Optional debug: if set, check per-sample labeled pixel counts
        # (non-ignore) and either print a warning or raise an error.
        self.max_labeled_points = max_labeled_points
        self.raise_on_too_many = raise_on_too_many

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: (N,H,W) in {0,1}
        # Support both (N,1,H,W) or (N,2,H,W) logits
        if logits.dim() == 4 and logits.size(1) == 2:
            # convert to a single binary logit (class1 vs class0)
            logits = logits[:, 1] - logits[:, 0]        # -> (N,H,W)
        elif logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)                   # -> (N,H,W)

        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)                 # -> (N,H,W)

        # Keep a copy for checks before casting
        raw_targets = targets
        targets = targets.float()

        # optional ignore mask
        if self.ignore_index is not None:
            valid = (raw_targets != self.ignore_index)
            t = torch.where(valid, targets, torch.zeros_like(targets))
        else:
            valid = torch.ones_like(targets, dtype=torch.bool)
            t = targets

        """# Simple checks: print if any sample has more than 2 or more than 3 labeled pixels
        v = valid
        if v.dim() == 2:
            v = v.unsqueeze(0)
        per_sample_counts = v.view(v.size(0), -1).sum(dim=1)
        gt2_idx = (per_sample_counts > 2).nonzero(as_tuple=False).squeeze(1)
        if gt2_idx.numel() > 0:
            samples = [(int(i.item()), int(per_sample_counts[i].item())) for i in gt2_idx[:10]]
            print(f"[warning] FocalLoss: samples with >2 labeled pixels: {samples}")
        gt3_idx = (per_sample_counts > 3).nonzero(as_tuple=False).squeeze(1)
        if gt3_idx.numel() > 0:
            samples = [(int(i.item()), int(per_sample_counts[i].item())) for i in gt3_idx[:10]]
            print(f"[warning] FocalLoss: samples with >3 labeled pixels: {samples}")"""

        # BCE with logits (per-pixel)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none") 
        log_pt = -bce
        pt = torch.exp(log_pt).clamp(min=1e-8, max=1.0)

        alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
        loss = -(alpha_t * (1 - pt) ** self.gamma * log_pt)

        if valid is not None:
            loss = loss * valid

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        # mean
        if valid is not None:
            denom = valid.sum().clamp(min=1)
            return loss.sum() / denom
        return loss.mean()

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        regularization = 0.0
        return loss + 0.1*regularization
