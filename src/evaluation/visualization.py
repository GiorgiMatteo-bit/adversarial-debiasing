"""Visualization utilities for training results and model comparisons."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_training_curves(results, save_path=None):
    """
    Plot training dynamics from adversarial training.
    
    Args:
        results: dict with job_losses, gender_losses, test_job_accuracies, test_gender_accuracies
        save_path: path to save figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(results['job_losses']) + 1)
    
    # Job loss
    ax1.plot(epochs, results['job_losses'], 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Job Classification Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Gender loss
    ax2.plot(epochs, results['gender_losses'], 'r-o', linewidth=2, markersize=6)
    ax2.set_title('Gender Prediction Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Job accuracy
    ax3.plot(epochs, results['test_job_accuracies'], 'b-o', linewidth=2, markersize=6)
    ax3.set_title('Job Classification Accuracy', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # Gender accuracy
    ax4.plot(epochs, results['test_gender_accuracies'], 'r-o', linewidth=2, markersize=6)
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random (50%)')
    ax4.set_title('Gender Prediction Accuracy\n(Lower = Better Debiasing)', fontsize=14)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_lambda_tradeoff(results, save_path=None):
    """
    Plot fairness-utility trade-off across lambda values.
    
    Args:
        results: list of dicts with lambda, job_f1, gender_accuracy
        save_path: path to save figure
    """
    lambdas = [r['lambda'] for r in results]
    job_f1s = [r['job_f1'] for r in results]
    gender_accs = [r['gender_accuracy'] for r in results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Job performance vs lambda
    ax1.plot(lambdas, job_f1s, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Lambda (λ)')
    ax1.set_ylabel('Job F1 Score')
    ax1.set_title('Job Performance vs Lambda\n(Higher is Better)')
    ax1.grid(True, alpha=0.3)
    
    # Gender accuracy vs lambda
    ax2.plot(lambdas, gender_accs, 'r-o', linewidth=2, markersize=8)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random (50%)')
    ax2.set_xlabel('Lambda (λ)')
    ax2.set_ylabel('Gender Prediction Accuracy')
    ax2.set_title('Gender Predictability vs Lambda\n(Lower is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Trade-off scatter
    sc = ax3.scatter(gender_accs, job_f1s, c=lambdas, cmap='viridis', s=150, alpha=0.7)
    for i, lambda_val in enumerate(lambdas):
        ax3.annotate(f'λ={lambda_val}', (gender_accs[i], job_f1s[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Gender Prediction Accuracy (Lower = More Debiased)')
    ax3.set_ylabel('Job F1 Score (Higher = Better Performance)')
    ax3.set_title('Fairness-Utility Trade-off\n(Bottom-Right is Ideal)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax3, label='Lambda (λ)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=['non-ICT', 'ICT'], 
                         title='Confusion Matrix', save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(baseline_metrics, adversarial_metrics, save_path=None):
    """
    Side-by-side comparison of baseline vs adversarial models.
    """
    metrics_names = ['Job F1', 'Job Accuracy', 'Gender Accuracy', 'Demographic Parity']
    baseline_vals = [
        baseline_metrics['job_f1'],
        baseline_metrics['job_accuracy'],
        baseline_metrics['gender_accuracy'],
        baseline_metrics['demographic_parity']
    ]
    adversarial_vals = [
        adversarial_metrics['job_f1'],
        adversarial_metrics['job_accuracy'],
        adversarial_metrics['gender_accuracy'],
        adversarial_metrics['demographic_parity']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, adversarial_vals, width, label='Adversarial', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Baseline vs Adversarial Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
