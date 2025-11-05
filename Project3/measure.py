import torch

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred   = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred   = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

def accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    pred_bin = (pred >= threshold).float()
    return (pred_bin == target.float()).float().mean()

def sensitivity(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    pred_bin = (pred >= threshold).float()
    target   = target.float()
    TP = (pred_bin * target).sum()
    FN = ((1.0 - pred_bin) * target).sum()
    return (TP + eps) / (TP + FN + eps)

def specificity(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    pred_bin = (pred >= threshold).float()
    target   = target.float()
    TN = ((1.0 - pred_bin) * (1.0 - target)).sum()
    FP = (pred_bin * (1.0 - target)).sum()
    return (TN + eps) / (TN + FP + eps)
