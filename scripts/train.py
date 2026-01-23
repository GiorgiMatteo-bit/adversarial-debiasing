"""
Main training script for adversarial debiasing pipeline.
Executes all 4 phases sequentially.
"""

import torch
import yaml
from pathlib import Path

from src.models.modernbert import load_tokenizer
from src.models.adversarial import AdversarialModel, BaselineModel
from src.models.probe import ProbeClassifier
from src.preprocessing.data_loader import create_dataloaders, calculate_class_weights
from src.training.baseline import BaselineTrainer
from src.training.adversarial_trainer import AdversarialTrainer
from src.training.lambda_optimization import LambdaOptimizer
from src.training.probe_trainer import ProbeTrainer
from src.evaluation.metrics import evaluate_model, print_evaluation_report, compare_models


def load_config(config_path='configs/default.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def phase1_baseline(config, train_loader, test_loader, device):
    """
    Phase 1: Establish baseline ICT classifier.
    (Assumes model already pre-trained from Phase 2 transfer learning)
    """
    print(f"\n{'='*60}")
    print("PHASE 1: Baseline ICT Classifier")
    print(f"{'='*60}\n")
    
    # Load or create baseline model
    baseline = BaselineModel(
        encoder_name=config['model']['encoder'],
        dropout=config['model']['dropout']
    )
    
    # Calculate class weights
    job_weights, _ = calculate_class_weights(config['data']['train_path'])
    
    # Train baseline
    trainer = BaselineTrainer(
        baseline, device,
        learning_rate=config['baseline']['unfrozen_lr'],
        weight_decay=config['baseline']['unfrozen_weight_decay']
    )
    
    trainer.train(
        train_loader, test_loader,
        epochs=config['baseline']['unfrozen_epochs'],
        class_weights=job_weights
    )
    
    return baseline


def phase2_adversary_pretrain(config, adversarial_model, train_loader, device):
    """Phase 2: Pre-train gender classifier on frozen encoder."""
    print(f"\n{'='*60}")
    print("PHASE 2: Adversary Pre-training")
    print(f"{'='*60}\n")
    
    trainer = AdversarialTrainer(adversarial_model, device)
    trainer.phase2_pretrain_adversary(
        train_loader,
        epochs=config['adversary_pretrain']['epochs']
    )


def phase3_lambda_optimization(config, adversarial_model, train_loader, test_loader, device):
    """Phase 3: Grid search for optimal lambda."""
    print(f"\n{'='*60}")
    print("PHASE 3: Lambda Optimization")
    print(f"{'='*60}\n")
    
    job_weights, _ = calculate_class_weights(config['data']['train_path'])
    
    optimizer = LambdaOptimizer(adversarial_model, device)
    results, best_lambda = optimizer.optimize(
        train_loader, test_loader,
        job_weights=job_weights,
        lambdas=config['lambda_optimization']['lambdas']
    )
    
    return best_lambda


def phase4_adversarial_training(config, adversarial_model, train_loader, test_loader, 
                               device, optimal_lambda):
    """Phase 4: Full adversarial training with progressive scheduling."""
    print(f"\n{'='*60}")
    print("PHASE 4: Adversarial Training")
    print(f"{'='*60}\n")
    
    job_weights, _ = calculate_class_weights(config['data']['train_path'])
    
    trainer = AdversarialTrainer(adversarial_model, device)
    results = trainer.phase4_adversarial_training(
        train_loader, test_loader,
        optimal_lambda=optimal_lambda,
        epochs=config['adversarial_training']['epochs'],
        job_weights=job_weights
    )
    
    return results


def phase5_evaluation(config, baseline_model, adversarial_model, train_loader, 
                     test_loader, device):
    """Phase 5: Evaluate baseline vs adversarial model."""
    print(f"\n{'='*60}")
    print("PHASE 5: Comparative Evaluation")
    print(f"{'='*60}\n")
    
    # 5a: Extended baseline training
    print("Training extended baseline...")
    baseline_trainer = BaselineTrainer(
        baseline_model, device,
        learning_rate=config['extended_baseline']['learning_rate'],
        weight_decay=config['extended_baseline']['weight_decay']
    )
    job_weights, _ = calculate_class_weights(config['data']['train_path'])
    baseline_trainer.train(
        train_loader, test_loader,
        epochs=config['extended_baseline']['epochs'],
        class_weights=job_weights
    )
    
    # 5b: Train probe on baseline
    print("\nTraining probe classifier on baseline...")
    probe = ProbeClassifier(
        input_dim=config['model']['hidden_size'],
        hidden_dim=config['model']['probe']['hidden_dim']
    )
    probe_trainer = ProbeTrainer(probe, baseline_model, device)
    probe_accuracy = probe_trainer.train(
        train_loader, test_loader,
        epochs=config['probe_training']['epochs']
    )
    
    # Evaluate both models
    print("\nEvaluating models...")
    
    # Baseline evaluation
    baseline_model.eval()
    baseline_job_acc, all_baseline_job_preds, all_job_labels = baseline_trainer.evaluate(test_loader)
    
    # Get baseline gender predictions via probe
    probe.eval()
    baseline_gender_preds = []
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            _, features = baseline_model(input_ids, attention_mask, return_features=True)
            gender_preds = probe.predict(features)
            baseline_gender_preds.extend(gender_preds.cpu().numpy())
    
    # Get gender labels
    all_gender_labels = []
    for batch in test_loader:
        all_gender_labels.extend(batch['gender_label'].numpy())
    
    baseline_metrics = evaluate_model(
        all_baseline_job_preds, all_job_labels,
        baseline_gender_preds, all_gender_labels
    )
    
    # Adversarial evaluation
    adversarial_trainer = AdversarialTrainer(adversarial_model, device)
    adv_job_acc, adv_gender_acc, adv_job_preds, adv_gender_preds = \
        adversarial_trainer.evaluate_adversarial(test_loader)
    
    adversarial_metrics = evaluate_model(
        adv_job_preds, all_job_labels,
        adv_gender_preds, all_gender_labels
    )
    
    # Print reports
    print_evaluation_report(baseline_metrics, "Baseline")
    print_evaluation_report(adversarial_metrics, "Adversarial")
    compare_models(baseline_metrics, adversarial_metrics)
    
    return baseline_metrics, adversarial_metrics


def main():
    """Execute complete 4-phase adversarial debiasing pipeline."""
    
    # Setup
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and data
    tokenizer = load_tokenizer(config['model']['encoder'])
    train_loader, test_loader = create_dataloaders(
        config['data']['train_path'],
        config['data']['test_path'],
        tokenizer,
        batch_size=config['data']['batch_size']
    )
    
    # Initialize models
    baseline_model = BaselineModel(
        encoder_name=config['model']['encoder'],
        dropout=config['model']['dropout']
    )
    
    adversarial_model = AdversarialModel(
        encoder_name=config['model']['encoder'],
        lambda_=config['adversarial_training']['optimal_lambda'],
        dropout=config['model']['dropout']
    )
    
    # Execute pipeline
    # Phase 1: Baseline (or load pre-trained from transfer learning)
    baseline_model = phase1_baseline(config, train_loader, test_loader, device)
    
    # Phase 2: Pre-train adversary
    phase2_adversary_pretrain(config, adversarial_model, train_loader, device)
    
    # Phase 3: Optimize lambda
    optimal_lambda = phase3_lambda_optimization(
        config, adversarial_model, train_loader, test_loader, device
    )
    
    # Phase 4: Adversarial training
    phase4_adversarial_training(
        config, adversarial_model, train_loader, test_loader, device, optimal_lambda
    )
    
    # Phase 5: Evaluation
    baseline_metrics, adversarial_metrics = phase5_evaluation(
        config, baseline_model, adversarial_model, train_loader, test_loader, device
    )
    
    # Save models
    torch.save(baseline_model.state_dict(), 'results/baseline_model.pth')
    torch.save(adversarial_model.state_dict(), 'results/adversarial_model.pth')
    
    print("\nTraining complete! Models saved to results/")


if __name__ == "__main__":
    main()
