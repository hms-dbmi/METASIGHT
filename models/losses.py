"""Loss functions for metastasis prediction."""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance (Lin et al., 2017).
    
    Downweights easy examples and focuses on hard examples.
    """
    
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class TPRLoss(nn.Module):
    """
    True Positive Rate loss to improve recall for metastasis class.
    """
    
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta
    
    def forward(self, outputs, targets):
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = (probs > 0.5).float()
        
        TP = (preds * targets).sum()
        FN = ((1 - preds) * targets).sum()
        TPR = TP / (TP + FN + 1e-6)
        
        tpr_loss = 1 - TPR
        return self.beta * tpr_loss


def get_loss_function(loss_type, class_weights=None, device='cuda'):
    """
    Factory function for loss functions.
    
    Args:
        loss_type: 'ce' | 'focal' | 'combined'
        class_weights: Class weights tensor
        device: Device for loss functions
    
    Returns:
        Loss function
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    ce_loss = nn.CrossEntropyLoss(weight=class_weights).to(device)
    focal_loss = FocalLoss(alpha=0.75, gamma=2, weight=class_weights).to(device)
    tpr_loss = TPRLoss(beta=0.5).to(device)
    
    if loss_type == "ce":
        return ce_loss
    elif loss_type == "focal":
        return focal_loss
    elif loss_type == "combined":
        def combined_loss(outputs, targets):
            return ce_loss(outputs, targets) + 0.5 * tpr_loss(outputs, targets)
        return combined_loss
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

