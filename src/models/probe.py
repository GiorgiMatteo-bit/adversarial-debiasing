"""
Probe classifier for measuring gender bias in learned representations.
Trained on frozen baseline encoder features to quantify residual gender information.
Architecture: 768 → 64 → 2 (49,282 parameters)
"""

import torch
import torch.nn as nn


class ProbeClassifier(nn.Module):
    """
    Gender probe classifier for bias measurement.
    
    Trained exclusively on frozen encoder features to measure
    maximum gender information exploitable from representations.
    """
    
    def __init__(self, input_dim=768, hidden_dim=64, num_classes=2):
        """
        Args:
            input_dim: encoder feature dimension (768 for ModernBERT)
            hidden_dim: hidden layer size
            num_classes: binary gender classification (2)
        """
        super().__init__()
        
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, features):
        """
        Args:
            features: [batch_size, 768] - frozen encoder outputs
        Returns:
            logits: [batch_size, 2] - gender prediction logits
        """
        return self.probe(features)
    
    def predict(self, features):
        """
        Predict gender labels from features.
        
        Args:
            features: [batch_size, 768]
        Returns:
            predictions: [batch_size] - predicted gender labels (0/1)
        """
        with torch.no_grad():
            logits = self.forward(features)
            predictions = torch.argmax(logits, dim=1)
        return predictions
