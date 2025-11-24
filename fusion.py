"""
Fusion methods for combining embeddings from multiple layers.
Implements the fusion strategies described in the LLMEmbed paper.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
import config


def mean_fusion(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Simple mean fusion: average embeddings across layers.
    
    Args:
        embeddings: List of embedding arrays, each of shape (n_samples, hidden_dim)
        
    Returns:
        Fused embedding array of shape (n_samples, hidden_dim)
    """
    if len(embeddings) == 1:
        return embeddings[0]
    
    # Stack and average
    stacked = np.stack(embeddings, axis=0)  # (n_layers, n_samples, hidden_dim)
    fused = np.mean(stacked, axis=0)  # (n_samples, hidden_dim)
    return fused


def concatenation_fusion(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Concatenation fusion: concatenate embeddings from all layers.
    
    Args:
        embeddings: List of embedding arrays, each of shape (n_samples, hidden_dim)
        
    Returns:
        Fused embedding array of shape (n_samples, n_layers * hidden_dim)
    """
    if len(embeddings) == 1:
        return embeddings[0]
    
    # Concatenate along feature dimension
    fused = np.concatenate(embeddings, axis=1)  # (n_samples, n_layers * hidden_dim)
    return fused


class LayerWeightedFusion(nn.Module):
    """
    Layer-wise learned scalar weighting fusion.
    Learns a scalar weight for each layer and applies weighted combination.
    """
    
    def __init__(self, num_layers: int):
        """
        Initialize layer-weighted fusion.
        
        Args:
            num_layers: Number of layers to fuse
        """
        super(LayerWeightedFusion, self).__init__()
        # Learnable scalar weights for each layer
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply weighted fusion.
        
        Args:
            embeddings: List of embedding tensors, each of shape (batch_size, hidden_dim)
            
        Returns:
            Fused embedding tensor of shape (batch_size, hidden_dim)
        """
        # Stack embeddings: (batch_size, num_layers, hidden_dim)
        stacked = torch.stack(embeddings, dim=1)
        
        # Apply softmax to weights for normalization
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        # Weighted sum: (batch_size, hidden_dim)
        weighted = torch.sum(
            stacked * normalized_weights.unsqueeze(0).unsqueeze(-1),
            dim=1
        )
        
        return weighted


def layer_weighted_fusion(
    embeddings: List[np.ndarray],
    train_embeddings: Optional[List[np.ndarray]] = None,
    train_labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Layer-weighted fusion with learned weights.
    If training data is provided, learns optimal weights.
    Otherwise, uses uniform weights (equivalent to mean fusion).
    
    Args:
        embeddings: List of embedding arrays to fuse
        train_embeddings: Optional training embeddings for learning weights
        train_labels: Optional training labels for learning weights
        
    Returns:
        Fused embedding array
    """
    num_layers = len(embeddings)
    
    # If no training data, fall back to mean fusion
    if train_embeddings is None or train_labels is None:
        return mean_fusion(embeddings)
    
    # Convert to tensors
    train_tensors = [torch.FloatTensor(emb) for emb in train_embeddings]
    train_labels_tensor = torch.LongTensor(train_labels)
    
    # Initialize fusion module
    fusion_module = LayerWeightedFusion(num_layers)
    optimizer = torch.optim.Adam(fusion_module.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Simple linear classifier for learning weights
    # We'll use a simple approach: learn weights that maximize separability
    hidden_dim = train_tensors[0].shape[1]
    classifier = nn.Linear(hidden_dim, len(np.unique(train_labels)))
    
    # Training loop (simplified - just a few iterations)
    fusion_module.train()
    classifier.train()
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Fuse embeddings
        fused = fusion_module(train_tensors)
        
        # Classify
        logits = classifier(fused)
        loss = criterion(logits, train_labels_tensor)
        
        loss.backward()
        optimizer.step()
    
    # Apply learned weights to input embeddings
    fusion_module.eval()
    with torch.no_grad():
        test_tensors = [torch.FloatTensor(emb) for emb in embeddings]
        fused = fusion_module(test_tensors)
        fused_np = fused.numpy()
    
    return fused_np


def apply_fusion(
    embeddings: List[np.ndarray],
    method: str = None,
    train_embeddings: Optional[List[np.ndarray]] = None,
    train_labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply the specified fusion method.
    
    Args:
        embeddings: List of embedding arrays to fuse
        method: Fusion method ('mean', 'concatenation', 'layer_weighted')
        train_embeddings: Optional training embeddings for learned fusion
        train_labels: Optional training labels for learned fusion
        
    Returns:
        Fused embedding array
    """
    method = method or config.FUSION_METHOD
    
    if method == "mean":
        return mean_fusion(embeddings)
    elif method == "concatenation":
        return concatenation_fusion(embeddings)
    elif method == "layer_weighted":
        return layer_weighted_fusion(embeddings, train_embeddings, train_labels)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

