"""
Probe classifier training for measuring bias in baseline model.
Trained on frozen encoder features to quantify gender information leakage.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


class ProbeTrainer:
    """Train probe classifier on frozen baseline features."""
    
    def __init__(self, probe_model, baseline_model, device):
        """
        Args:
            probe_model: ProbeClassifier instance
            baseline_model: BaselineModel with frozen encoder
            device: torch device
        """
        self.probe = probe_model.to(device)
        self.baseline = baseline_model.to(device)
        self.device = device
        
        # Freeze baseline encoder
        self.baseline.freeze_encoder()
        self.baseline.eval()
        
        # Probe optimizer (from thesis: lr=1e-4, 15 epochs)
        self.optimizer = Adam(self.probe.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        """Train probe for one epoch on frozen features."""
        self.probe.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            gender_labels = batch['gender_label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Extract frozen features
            with torch.no_grad():
                _, features = self.baseline(input_ids, attention_mask, return_features=True)
            
            # Train probe on features
            logits = self.probe(features)
            loss = self.criterion(logits, gender_labels)
            
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == gender_labels).sum().item()
            total_samples += len(gender_labels)
            total_loss += loss.item()
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """Evaluate probe on test set."""
        self.probe.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                gender_labels = batch['gender_label'].to(self.device)
                
                # Extract frozen features
                _, features = self.baseline(input_ids, attention_mask, return_features=True)
                
                # Probe prediction
                logits = self.probe(features)
                preds = torch.argmax(logits, dim=1)
                
                total_correct += (preds == gender_labels).sum().item()
                total_samples += len(gender_labels)
        
        return total_correct / total_samples
    
    def train(self, train_loader, test_loader, epochs=15):
        """
        Full probe training (15 epochs from thesis).
        
        Returns:
            final_test_accuracy: probe accuracy on test set
        """
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_acc = self.evaluate(test_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        final_test_acc = self.evaluate(test_loader)
        print(f"\nFinal Probe Gender Accuracy: {final_test_acc:.4f}")
        
        return final_test_acc
