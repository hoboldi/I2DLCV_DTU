import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 reduction: str = "mean", ignore_index: int | None = None):
        super().__init__()
        assert reduction in ("none", "mean", "sum")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # squeeze channel dim if it's 1
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        targets = targets.float()

        # optional ignore mask
        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            t = torch.where(valid, targets, torch.zeros_like(targets))
        else:
            valid = None
            t = targets

        # per-pixel BCE with logits
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

        # reduction == "mean"
        if valid is not None:
            denom = valid.sum()  # tensor
            return loss.sum() / denom.clamp(min=1)
        else:
            return loss.mean()

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        regularization = 0.0
        return loss + 0.1*regularization
