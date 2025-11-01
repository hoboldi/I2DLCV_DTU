import torch

def dice_coefficient(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

def accuracy(pred, target):
    pred_bin = (pred > 0.5).float()
    return (pred_bin == target).float().mean()

def sensitivity(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()
    TP = (pred_bin * target).sum()
    FN = ((1 - pred_bin) * target).sum()
    return (TP + eps) / (TP + FN + eps)

def specificity(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()
    TN = ((1 - pred_bin) * (1 - target)).sum()
    FP = (pred_bin * (1 - target)).sum()
    return (TN + eps) / (TN + FP + eps)
