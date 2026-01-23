"""
Baseline ICT classifier training (no adversarial debiasing).
Used for comparison with adversarial model.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


class BaselineTrainer:
    """Train baseline ICT classifier without fairness constraints."""
    
    def __init__(self, model, device, learning_rate=1e-6, weight_decay=0.1):
        """
        Args:
            model: BaselineModel instance
            device: torch device
            learning_rate: AdamW learning rate (1e-6 from thesis)
            weight_decay: L2 regularization (0.1 from thesis)
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
    def train_epoch(self, train_loader, class_weights=None):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader with resume data
            class_weights: [weight_non_ict, weight_ict]
            
        Returns:
            avg_loss: average training loss
        """
        self.model.train()
        total_loss = 0
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['job_label'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set.
        
        Returns:
            accuracy: classification accuracy
            all_preds: predictions for all samples
            all_labels: ground truth labels
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['job_label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        return accuracy, all_preds, all_labels
    
    def train(self, train_loader, test_loader, epochs=15, class_weights=None):
        """
        Full training loop.
        
        Args:
            train_loader: training DataLoader
            test_loader: test DataLoader
            epochs: number of epochs (15 from thesis)
            class_weights: class weights for loss
            
        Returns:
            train_losses: list of training losses
            test_accuracies: list of test accuracies
        """
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, class_weights)
            train_losses.append(train_loss)
            
            # Evaluate
            test_acc, _, _ = self.evaluate(test_loader)
            test_accuracies.append(test_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        return train_losses, test_accuracies
