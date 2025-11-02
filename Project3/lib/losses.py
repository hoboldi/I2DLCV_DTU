import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    """Binary cross entropy with logits (stable)."""
    def __init__(self, pos_weight: float | None = None, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = None if pos_weight is None else torch.tensor(pos_weight)
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # y_pred: logits; y_true: {0,1} float
        pos_w = self.pos_weight
        if pos_w is not None and pos_w.device != y_pred.device:
            pos_w = pos_w.to(y_pred.device)
        return F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=pos_w, reduction=self.reduction)


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Uses probabilities = sigmoid(logits); computes per-sample Dice then averages.
    """
    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # y_pred: logits (N, 1, H, W) or (N, H, W); y_true: same shape as probabilities
        probs = torch.sigmoid(y_pred)
        # Ensure shape compatibility (squeeze channel dim if present)
        if probs.dim() == y_true.dim() + 1 and probs.size(1) == 1:
            probs = probs.squeeze(1)

        # Flatten per sample
        probs_f = probs.view(probs.size(0), -1)
        true_f  = y_true.view(y_true.size(0), -1)

        intersection = (probs_f * true_f).sum(dim=1)
        denom = probs_f.sum(dim=1) + true_f.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice  # per-sample

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Binary focal loss with logits.
    α balances classes (default 0.25 from RetinaNet), γ focuses on hard examples (default 2).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # logits -> probs
        p = torch.sigmoid(y_pred).clamp(min=1e-8, max=1.0 - 1e-8)

        # Positive and negative focal terms
        pt_pos = p
        pt_neg = 1.0 - p

        loss_pos = -self.alpha * (1.0 - pt_pos).pow(self.gamma) * y_true * torch.log(pt_pos)
        loss_neg = -(1.0 - self.alpha) * (pt_neg).pow(self.gamma) * (1.0 - y_true) * torch.log(pt_neg)

        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCELoss_TotalVariation(nn.Module):
    """
    BCE with logits + Total Variation (TV) on probabilities to encourage smooth masks.
    TV is isotropic: mean(|dx| + |dy|).
    """
    def __init__(self, tv_weight: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.tv_weight = tv_weight
        self.reduction = reduction
        self._bce = BCELoss(reduction=reduction)

    @staticmethod
    def _total_variation(probs: torch.Tensor) -> torch.Tensor:
        # probs: (N, H, W) or (N, 1, H, W)
        if probs.dim() == 4 and probs.size(1) == 1:
            probs = probs[:, 0]
        elif probs.dim() == 4:
            # If channels>1, sum TV across channels
            probs = probs

        if probs.dim() == 3:
            # (N, H, W)
            dx = probs[:, :, 1:] - probs[:, :, :-1]
            dy = probs[:, 1:, :] - probs[:, :-1, :]
            tv = dx.abs().mean() + dy.abs().mean()
        else:
            # (N, C, H, W)
            dx = probs[:, :, :, 1:] - probs[:, :, :, :-1]
            dy = probs[:, :, 1:, :] - probs[:, :, :-1, :]
            tv = dx.abs().mean() + dy.abs().mean()
        return tv

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        bce = self._bce(y_pred, y_true)
        probs = torch.sigmoid(y_pred)
        tv = self._total_variation(probs)
        return bce + self.tv_weight * tv