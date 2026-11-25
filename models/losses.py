"""Loss functions for metastasis prediction."""

import torch
import torch.nn as nn


def _reduce(loss, reduction):
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance (Lin et al., 2017).
    
    Downweights easy examples and focuses on hard examples.
    """
    
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return _reduce(focal_loss, self.reduction)


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


class DistanceAwareLoss(nn.Module):
    """
    Distance-aware cross-entropy for ordinal trajectory classes.
    
    L = CE + lambda_dist * sum_k C[y, k] * p_k
    
    with C[y, k] = |y - k| ('linear') or (y - k)^2 ('squared'), so probability
    placed on classes far from the observed one (e.g. stable vs. distant
    metastasis) is penalized more than on neighbouring classes.
    
    Args:
        lambda_dist: Weight of the expected-distance penalty (default: 1.0)
        class_weights: Optional per-class weights for the CE term
        cost_type: 'linear' | 'squared'
        reduction: 'mean' | 'sum' | 'none' (per-sample, for IPCW weighting)
    """
    
    def __init__(self, lambda_dist=1.0, class_weights=None, cost_type='linear',
                 n_classes=3, reduction='mean'):
        super().__init__()
        self.lambda_dist = lambda_dist
        self.class_weights = class_weights
        self.reduction = reduction
        idx = torch.arange(n_classes, dtype=torch.float32)
        dist = (idx[:, None] - idx[None, :]).abs()
        if cost_type == 'squared':
            dist = dist ** 2
        elif cost_type != 'linear':
            raise ValueError(f"Unknown cost_type: {cost_type}")
        self.register_buffer('cost_matrix', dist)
    
    def forward(self, logits, targets):
        logits = logits.float()
        ce = nn.functional.cross_entropy(logits, targets, weight=self.class_weights,
                                         reduction='none')
        probs = torch.softmax(logits, dim=1)
        dist_penalty = (self.cost_matrix[targets] * probs).sum(dim=1)
        return _reduce(ce + self.lambda_dist * dist_penalty, self.reduction)


def get_loss_function(loss_type, class_weights=None, device='cuda', reduction='mean'):
    """
    Factory function for loss functions.
    
    Args:
        loss_type: 'ce' | 'focal' | 'combined' | 'ordinal' (alias: 'distance')
        class_weights: Class weights tensor
        device: Device for loss functions
        reduction: 'mean' (scalar loss) | 'none' (per-sample loss, e.g. for IPCW weighting)
    
    Returns:
        Loss function
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction).to(device)
    focal_loss = FocalLoss(alpha=0.75, gamma=2, weight=class_weights, reduction=reduction).to(device)
    tpr_loss = TPRLoss(beta=0.5).to(device)
    
    if loss_type == "ce":
        return ce_loss
    elif loss_type == "focal":
        return focal_loss
    elif loss_type == "combined":
        def combined_loss(outputs, targets):
            return ce_loss(outputs, targets) + 0.5 * tpr_loss(outputs, targets)
        return combined_loss
    elif loss_type in ("ordinal", "distance"):
        return DistanceAwareLoss(lambda_dist=1.0, class_weights=class_weights,
                                 reduction=reduction).to(device)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
