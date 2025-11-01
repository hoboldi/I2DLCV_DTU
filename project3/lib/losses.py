import torch
import torch.nn as nn

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
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, gamma=2):
        probs = torch.sigmoid(y_pred)
        probs = probs.clamp(min=1e-8, max=1.0 - 1e-8)
        term1 = (1 - probs) ** gamma * y_true * torch.log(probs)
        term2 = (1 - y_true) * torch.log(1 - probs)
        loss = -(term1 + term2)
        return torch.mean(loss)

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        regularization = 0.0
        return loss + 0.1*regularization

