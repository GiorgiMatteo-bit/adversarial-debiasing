"""
Complete adversarial debiasing model combining:
- ModernBERT encoder
- Job classifier (ICT prediction)
- Gradient Reversal Layer
- Gender classifier (adversary)
"""

import torch
import torch.nn as nn
from .modernbert import ModernBERTEncoder, JobClassifier, GenderClassifier
from .gradient_reversal import GradientReversalLayer


class AdversarialModel(nn.Module):
    """
    Adversarial debiasing architecture for fair ICT classification.
    
    Architecture flow:
        Input Text [batch, 256] 
        → ModernBERT [batch, 768]
        → Dropout(0.4)
        ├─→ Job Classifier → Job Logits
        └─→ GRL(λ) → Gender Classifier → Gender Logits
    """
    
    def __init__(self, encoder_name='answerdotai/ModernBERT-base', 
                 lambda_=1.0, dropout=0.4):
        """
        Args:
            encoder_name: ModernBERT model identifier
            lambda_: gradient reversal strength
            dropout: dropout rate for encoder output
        """
        super().__init__()
        
        # Core components
        self.encoder = ModernBERTEncoder(encoder_name, dropout=dropout)
        self.job_classifier = JobClassifier(input_dim=768, num_classes=2)
        self.grl = GradientReversalLayer(lambda_=lambda_)
        self.gender_classifier = GenderClassifier(
            input_dim=768, 
            hidden_dim=256, 
            num_classes=2,
            dropout=dropout
        )
        
    def forward(self, input_ids, attention_mask, return_features=False):
        """
        Forward pass with adversarial architecture.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_features: if True, return encoder features
            
        Returns:
            job_logits: [batch_size, 2]
            gender_logits: [batch_size, 2]
            features: [batch_size, 768] (if return_features=True)
        """
        # Encode resume text
        features = self.encoder(input_ids, attention_mask)
        
        # Job classification path (normal gradients)
        job_logits = self.job_classifier(features)
        
        # Adversarial path (reversed gradients)
        reversed_features = self.grl(features)
        gender_logits = self.gender_classifier(reversed_features)
        
        if return_features:
            return job_logits, gender_logits, features
        return job_logits, gender_logits
    
    def set_lambda(self, lambda_):
        """Update gradient reversal strength (for progressive scheduling)."""
        self.grl.set_lambda(lambda_)
    
    def freeze_encoder(self):
        """Freeze encoder weights (used during adversary pre-training)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights (for full adversarial training)."""
        for param in self.encoder.parameters():
            param.requires_grad = True


class BaselineModel(nn.Module):
    """
    Baseline ICT classifier without adversarial training.
    Used for comparison and probe evaluation.
    """
    
    def __init__(self, encoder_name='answerdotai/ModernBERT-base', dropout=0.4):
        super().__init__()
        self.encoder = ModernBERTEncoder(encoder_name, dropout=dropout)
        self.classifier = JobClassifier(input_dim=768, num_classes=2)
        
    def forward(self, input_ids, attention_mask, return_features=False):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_features: if True, return encoder features
            
        Returns:
            logits: [batch_size, 2]
            features: [batch_size, 768] (if return_features=True)
        """
        features = self.encoder(input_ids, attention_mask)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def freeze_encoder(self):
        """Freeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = False
