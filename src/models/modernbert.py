"""
ModernBERT-based encoder for resume classification.
Uses answerdotai/ModernBERT-base (110M parameters, 768-dim hidden states)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ModernBERTEncoder(nn.Module):
    """ModernBERT encoder with dropout for resume encoding."""
    
    def __init__(self, model_name='answerdotai/ModernBERT-base', dropout=0.4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = 768
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            pooled_output: [batch_size, 768]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.dropout(pooled)


class JobClassifier(nn.Module):
    """Binary ICT vs non-ICT classifier."""
    
    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, features):
        """
        Args:
            features: [batch_size, 768]
        Returns:
            logits: [batch_size, 2]
        """
        return self.classifier(features)


class GenderClassifier(nn.Module):
    """Gender classifier: 768 → 256 → 2."""
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, dropout=0.4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, features):
        """
        Args:
            features: [batch_size, 768]
        Returns:
            logits: [batch_size, 2]
        """
        return self.network(features)


def load_tokenizer(model_name='answerdotai/ModernBERT-base'):
    """Load ModernBERT tokenizer with max_length=256."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 256
    return tokenizer
