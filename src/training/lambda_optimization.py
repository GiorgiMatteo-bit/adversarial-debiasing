"""
Phase 3: Lambda parameter optimization via grid search.
Tests λ ∈ {0.1, 0.5, 1.0, 2.0, 5.0} to find optimal fairness-utility trade-off.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score


class LambdaOptimizer:
    """Grid search for optimal lambda parameter."""
    
    def __init__(self, model, device):
        """
        Args:
            model: AdversarialModel instance
            device: torch device
        """
        self.model = model.to(device)
        self.device = device
        
    def evaluate_lambda(self, lambda_val, train_loader, test_loader, 
                       epochs=6, job_weights=None):
        """
        Evaluate single lambda value.
        
        Hyperparameters from thesis Table 9:
        - BERT learning rate: 5e-6
        - Gender learning rate: 5e-4
        - Epochs: 6 (quick evaluation)
        - Batch size: 16
        - Weight decay: 0.02
        
        Args:
            lambda_val: gradient reversal strength to test
            train_loader: training DataLoader
            test_loader: test DataLoader
            epochs: training epochs for evaluation
            job_weights: class weights for job loss
            
        Returns:
            results: dict with job_f1 and gender_accuracy
        """
        if job_weights is not None:
            job_weights = job_weights.to(self.device)
        
        # Reset model weights (reinitialize from baseline checkpoint)
        # In practice, load from saved baseline checkpoint
        
        # Differential learning rates
        encoder_params = list(self.model.encoder.parameters()) + \
                        list(self.model.job_classifier.parameters())
        gender_params = list(self.model.gender_classifier.parameters())
        
        optimizer = AdamW([
            {'params': encoder_params, 'lr': 5e-6},
            {'params': gender_params, 'lr': 5e-4}
        ], weight_decay=0.02)
        
        self.model.set_lambda(lambda_val)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                job_labels = batch['job_label'].to(self.device)
                gender_labels = batch['gender_label'].to(self.device)
                
                optimizer.zero_grad()
                
                job_logits, gender_logits = self.model(input_ids, attention_mask)
                
                job_loss = nn.CrossEntropyLoss(weight=job_weights)(job_logits, job_labels)
                gender_loss = nn.CrossEntropyLoss()(gender_logits, gender_labels)
                
                total_loss = job_loss + gender_loss
                total_loss.backward()
                optimizer.step()
        
        # Evaluate on test set
        self.model.eval()
        all_job_preds = []
        all_job_labels = []
        all_gender_preds = []
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
                all_job_labels.extend(job_labels.cpu().numpy())
                all_gender_preds.extend(gender_preds.cpu().numpy())
                all_gender_labels.extend(gender_labels.cpu().numpy())
        
        # Calculate metrics
        job_f1 = f1_score(all_job_labels, all_job_preds, average='weighted')
        gender_acc = sum(p == l for p, l in zip(all_gender_preds, all_gender_labels)) / len(all_gender_labels)
        
        return {
            'lambda': lambda_val,
            'job_f1': job_f1,
            'gender_accuracy': gender_acc
        }
    
    def optimize(self, train_loader, test_loader, job_weights=None,
                lambdas=[0.1, 0.5, 1.0, 2.0, 5.0]):
        """
        Grid search over lambda values.
        
        Selection criterion (from thesis):
        Score = 0.6 × F1_job + 0.4 × (1.0 - 2 × |Acc_gender - 0.5|)
        
        Args:
            train_loader: training DataLoader
            test_loader: test DataLoader
            job_weights: class weights for job loss
            lambdas: lambda values to test
            
        Returns:
            results: list of dicts with metrics for each lambda
            best_lambda: optimal lambda value
        """
        results = []
        
        for lambda_val in lambdas:
            print(f"\n{'='*50}")
            print(f"Testing λ = {lambda_val}")
            print(f"{'='*50}")
            
            result = self.evaluate_lambda(
                lambda_val, train_loader, test_loader, 
                job_weights=job_weights
            )
            results.append(result)
            
            print(f"Job F1: {result['job_f1']:.4f}")
            print(f"Gender Accuracy: {result['gender_accuracy']:.4f}")
        
        # Select best lambda using weighted score
        best_result = max(results, key=lambda r: 
            0.6 * r['job_f1'] + 0.4 * (1.0 - 2 * abs(r['gender_accuracy'] - 0.5))
        )
        
        print(f"\n{'='*50}")
        print(f"Optimal λ = {best_result['lambda']}")
        print(f"Job F1: {best_result['job_f1']:.4f}")
        print(f"Gender Acc: {best_result['gender_accuracy']:.4f}")
        print(f"{'='*50}\n")
        
        return results, best_result['lambda']
