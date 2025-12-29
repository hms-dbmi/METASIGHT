"""
Neural network architectures for metastasis prediction.

This module implements attention-based Multiple Instance Learning (MIL) for 
whole-slide image analysis using foundation model features.
"""

import torch
import torch.nn as nn


class DropoutOrIdentity(nn.Module):
    """Apply dropout if rate > 0, otherwise identity mapping."""
    
    def __init__(self, dropout=None):
        super().__init__()
        self.layer = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
    
    def forward(self, x):
        return self.layer(x)


class GatedAttention(nn.Module):
    """
    Gated attention mechanism for MIL (Ilse et al., 2018).
    
    Combines tanh (value) and sigmoid (gate) pathways to compute
    attention weights over instances.
    """
    
    def __init__(self, in_dim, hidden_dim, dropout=None):
        super().__init__()
        self.V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            DropoutOrIdentity(dropout)
        )
        self.U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            DropoutOrIdentity(dropout)
        )
        self.w = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Args:
            x: [batch, n_instances, in_dim]
        Returns:
            attention_weights: [batch, n_instances, 1]
        """
        A_V = self.V(x)
        A_U = self.U(x)
        A = self.w(A_V * A_U)
        return self.softmax(A)


class MILAttentionEncoder(nn.Module):
    """
    Attention-based aggregation of variable-length instance sets.
    
    Aggregates patch features into a fixed-length slide representation
    using learned attention weights.
    """
    
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            DropoutOrIdentity(dropout)
        )
        self.U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
            DropoutOrIdentity(dropout)
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, n_patches, feature_dim]
            mask: [batch, n_patches] (1=valid, 0=padding)
        Returns:
            aggregated: [batch, feature_dim]
        """
        attn = self.attention_weights(self.V(x) * self.U(x))
        
        if mask is not None:
            mask = mask.unsqueeze(-1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=1)
        aggregated = (attn * x).sum(dim=1)
        
        return aggregated


class MILNet(nn.Module):
    """
    MIL network for binary metastasis prediction.
    
    Architecture:
        1. Attention-based MIL encoder (aggregates patches)
        2. Prediction head with batch normalization and residual connection
    
    Args:
        feature_dim: Foundation model feature dimension (e.g., 768 for CHIEF)
        n_output: Number of output classes (2 for binary)
        hidden_dim: Hidden dimension for attention (default: 128)
        dropout: Dropout rate (default: 0.3)
        use_batchnorm: Whether to use batch normalization (default: True)
    """
    
    def __init__(self, feature_dim, n_output, hidden_dim=128, dropout=0.3, use_batchnorm=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_output = n_output
        
        # MIL encoder with attention
        self.encoder = MILAttentionEncoder(feature_dim, hidden_dim, dropout)
        
        # Prediction head with residual connection
        if use_batchnorm:
            self.input_bn = nn.BatchNorm1d(feature_dim)
            self.fc1 = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fc2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fc3 = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.out_layer = nn.Linear(64 + feature_dim, n_output)
        else:
            self.input_bn = nn.Identity()
            self.fc1 = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fc2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fc3 = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.out_layer = nn.Linear(64 + feature_dim, n_output)
    
    def forward(self, features, mask=None):
        """
        Args:
            features: [batch, n_tiles, feature_dim] - tile features from foundation model
            mask: [batch, n_tiles] (1=valid, 0=padding)
        Returns:
            predictions: [batch, n_output] - class logits
        """
        # Aggregate tiles using attention
        aggregated = self.encoder(features, mask)
        
        # Prediction head with residual connection
        residual = self.input_bn(aggregated)
        x = self.fc1(residual)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.cat([x, residual], dim=1)
        
        return self.out_layer(x)


class TrajectoryMILNet(nn.Module):
    """
    MIL network for future trajectory prediction.
    
    Multi-class classification for predicting patient outcome at a time horizon:
    - Class 0: Stable disease (no event)
    - Class 1: Locoregional recurrence
    - Class 2: Distant metastasis
    
    Args:
        feature_dim: Foundation model feature dimension
        n_classes: Number of output classes (default: 3)
        hidden_dim: Hidden dimension for attention (default: 128)
        dropout: Dropout rate (default: 0.1)
        use_batchnorm: Whether to use batch normalization (default: True)
    """
    
    def __init__(self, feature_dim, n_classes=3, hidden_dim=128, dropout=0.1, use_batchnorm=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        
        # MIL encoder with attention
        self.encoder = MILAttentionEncoder(feature_dim, hidden_dim, dropout)
        
        # Classification head
        if use_batchnorm:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_classes)
            )
    
    def forward(self, features, mask=None):
        """
        Args:
            features: [batch, n_tiles, feature_dim] - tile features from foundation model
            mask: [batch, n_tiles] (1=valid, 0=padding)
        Returns:
            class_logits: [batch, n_classes] - logits for each outcome class
        """
        # Aggregate tiles using attention
        aggregated = self.encoder(features, mask)
        return self.head(aggregated)

