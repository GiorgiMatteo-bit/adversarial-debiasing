"""
4-Phase Adversarial Debiasing Pipeline:
1. Baseline establishment (fine-tuned from transfer learning)
2. Adversary pre-training (frozen encoder)
3. Lambda optimization (grid search)
4. Final adversarial training (progressive scheduling)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np


class AdversarialTrainer:
    """Complete adversarial debiasing pipeline."""
    
    def __init__(self, model, device):
        """
        Args:
            model: AdversarialModel instance
            device: torch device
        """
        self.model = model.to(device)
        self.device = device
        
    def phase2_pretrain_adversary(self, train_loader, epochs=10):
        """
        Phase 2: Pre-train gender classifier on frozen encoder features.
        
        Hyperparameters from thesis Table 9:
        - Learning rate: 1e-3
        - Epochs: 10
        - Batch size: 32
        - Optimizer: AdamW (weight_decay=0.01)
        
        Args:
            train_loader: training DataLoader
            epochs: number of epochs (10 from thesis)
            
        Returns:
            gender_accuracies: list of accuracies per epoch
        """
        # Freeze encoder
        self.model.freeze_encoder()
        
        # Only optimize gender classifier
        optimizer = AdamW(
            self.model.gender_classifier.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()
        
        gender_accuracies = []
        
        for epoch in range(epochs):
            self.model.train()
            total_correct = 0
            total_samples = 0
            epoch_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Pre-training Adversary Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                gender_labels = batch['gender_label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass (encoder frozen)
                _, gender_logits = self.model(input_ids, attention_mask)
                loss = criterion(gender_logits, gender_labels)
                
                loss.backward()
                optimizer.step()
                
                # Track accuracy
                preds = torch.argmax(gender_logits, dim=1)
                total_correct += (preds == gender_labels).sum().item()
                total_samples += len(gender_labels)
                epoch_loss += loss.item()
            
            accuracy = total_correct / total_samples
            gender_accuracies.append(accuracy)
            avg_loss = epoch_loss / len(train_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Gender Acc: {accuracy:.4f}")
        
        # Unfreeze encoder for subsequent training
        self.model.unfreeze_encoder()
        
        return gender_accuracies
    
    def train_adversarial_epoch(self, train_loader, job_weights=None, lambda_=1.0):
        """
        Train one epoch of adversarial training.
        
        Combined loss: L_total = L_job + L_gender
        (GRL handles the gradient reversal, so we add losses normally)
        
        Args:
            train_loader: training DataLoader
            job_weights: class weights for job classification
            lambda_: gradient reversal strength
            
        Returns:
            avg_job_loss: average job classification loss
            avg_gender_loss: average gender prediction loss
        """
        self.model.train()
        self.model.set_lambda(lambda_)
        
        job_criterion = nn.CrossEntropyLoss(weight=job_weights)
        gender_criterion = nn.CrossEntropyLoss()
        
        total_job_loss = 0
        total_gender_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            job_labels = batch['job_label'].to(self.device)
            gender_labels = batch['gender_label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            job_logits, gender_logits = self.model(input_ids, attention_mask)
            
            # Calculate losses
            job_loss = job_criterion(job_logits, job_labels)
            gender_loss = gender_criterion(gender_logits, gender_labels)
            
            # Combined loss (GRL reverses gender gradients in backward pass)
            total_loss = job_loss + gender_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            total_job_loss += job_loss.item()
            total_gender_loss += gender_loss.item()
        
        return total_job_loss / len(train_loader), total_gender_loss / len(train_loader)
    
    def evaluate_adversarial(self, test_loader):
        """
        Evaluate adversarial model.
        
        Returns:
            job_accuracy: ICT classification accuracy
            gender_accuracy: gender prediction accuracy
            job_preds: job predictions
            gender_preds: gender predictions
        """
        self.model.eval()
        
        all_job_preds = []
        all_gender_preds = []
        all_job_labels = []
        all_gender_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                job_labels = batch['job_label'].to(self.device)
                gender_labels = batch['gender_label'].to(self.device)
                
                job_logits, gender_logits = self.model(input_ids, attention_mask)
                
                job_preds = torch.argmax(job_logits, dim=1)
                gender_preds = torch.argmax(gender_logits, dim=1)
                
                all_job_preds.extend(job_preds.cpu().numpy())
                all_gender_preds.extend(gender_preds.cpu().numpy())
                all_job_labels.extend(job_labels.cpu().numpy())
                all_gender_labels.extend(gender_labels.cpu().numpy())
        
        job_acc = sum(p == l for p, l in zip(all_job_preds, all_job_labels)) / len(all_job_labels)
        gender_acc = sum(p == l for p, l in zip(all_gender_preds, all_gender_labels)) / len(all_gender_labels)
        
        return job_acc, gender_acc, all_job_preds, all_gender_preds
    
    def phase4_adversarial_training(self, train_loader, test_loader, 
                                   optimal_lambda=2.0, epochs=15, job_weights=None):
        """
        Phase 4: Full adversarial training with progressive lambda scheduling.
        
        Hyperparameters from thesis Table 9:
        - BERT learning rate: 5e-6
        - Gender learning rate: 5e-4
        - Epochs: 15
        - Batch size: 32
        - Weight decay: 0.02
        
        Args:
            train_loader: training DataLoader
            test_loader: test DataLoader
            optimal_lambda: optimal λ from phase 3 (2.0 from thesis)
            epochs: number of epochs (15 from thesis)
            job_weights: class weights for job loss
            
        Returns:
            results: dict with training history
        """
        if job_weights is not None:
            job_weights = job_weights.to(self.device)
        
        # Differential learning rates
        encoder_params = list(self.model.encoder.parameters()) + \
                        list(self.model.job_classifier.parameters())
        gender_params = list(self.model.gender_classifier.parameters())
        
        self.optimizer = AdamW([
            {'params': encoder_params, 'lr': 5e-6},
            {'params': gender_params, 'lr': 5e-4}
        ], weight_decay=0.02)
        
        # Track metrics
        job_losses = []
        gender_losses = []
        test_job_accs = []
        test_gender_accs = []
        
        for epoch in range(epochs):
            # Progressive lambda scheduling: 0 → optimal_lambda
            current_lambda = optimal_lambda * (epoch / epochs)
            
            # Train
            job_loss, gender_loss = self.train_adversarial_epoch(
                train_loader, job_weights, lambda_=current_lambda
            )
            
            # Evaluate
            job_acc, gender_acc, _, _ = self.evaluate_adversarial(test_loader)
            
            # Store metrics
            job_losses.append(job_loss)
            gender_losses.append(gender_loss)
            test_job_accs.append(job_acc)
            test_gender_accs.append(gender_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - λ={current_lambda:.2f}")
            print(f"  Job Loss: {job_loss:.4f}, Gender Loss: {gender_loss:.4f}")
            print(f"  Job Acc: {job_acc:.4f}, Gender Acc: {gender_acc:.4f}")
        
        return {
            'job_losses': job_losses,
            'gender_losses': gender_losses,
            'test_job_accuracies': test_job_accs,
            'test_gender_accuracies': test_gender_accs
        }
