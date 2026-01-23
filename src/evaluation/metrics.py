"""
Evaluation metrics for adversarial debiasing.
Includes classification performance and fairness metrics.
"""

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def calculate_f1_score(y_true, y_pred, average='weighted'):
    """Calculate F1 score."""
    return f1_score(y_true, y_pred, average=average)


def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy."""
    return np.mean(np.array(y_true) == np.array(y_pred))


def calculate_demographic_parity(job_preds, gender_labels):
    """
    Calculate demographic parity difference.
    
    Measures: |P(ICT|Female) - P(ICT|Male)|
    Values near 0 indicate equitable treatment.
    
    Args:
        job_preds: predicted job labels (0/1)
        gender_labels: gender labels (0=female, 1=male)
        
    Returns:
        parity_diff: demographic parity difference
        female_rate: P(ICT|Female)
        male_rate: P(ICT|Male)
    """
    job_preds = np.array(job_preds)
    gender_labels = np.array(gender_labels)
    
    # Female prediction rate
    female_mask = gender_labels == 0
    female_rate = np.mean(job_preds[female_mask]) if female_mask.any() else 0.0
    
    # Male prediction rate
    male_mask = gender_labels == 1
    male_rate = np.mean(job_preds[male_mask]) if male_mask.any() else 0.0
    
    parity_diff = abs(female_rate - male_rate)
    
    return parity_diff, female_rate, male_rate


def evaluate_model(job_preds, job_labels, gender_preds, gender_labels):
    """
    Comprehensive model evaluation.
    
    Returns:
        metrics: dict with all evaluation metrics
    """
    # Classification metrics
    job_f1 = calculate_f1_score(job_labels, job_preds)
    job_acc = calculate_accuracy(job_labels, job_preds)
    gender_acc = calculate_accuracy(gender_labels, gender_preds)
    
    # Fairness metrics
    parity_diff, female_rate, male_rate = calculate_demographic_parity(
        job_preds, gender_labels
    )
    
    metrics = {
        'job_f1': job_f1,
        'job_accuracy': job_acc,
        'gender_accuracy': gender_acc,
        'demographic_parity': parity_diff,
        'female_prediction_rate': female_rate,
        'male_prediction_rate': male_rate
    }
    
    return metrics


def print_evaluation_report(metrics, model_name="Model"):
    """Print formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Report")
    print(f"{'='*60}")
    print(f"Classification Performance:")
    print(f"  Job F1 Score:       {metrics['job_f1']:.4f}")
    print(f"  Job Accuracy:       {metrics['job_accuracy']:.4f}")
    print(f"\nFairness Metrics:")
    print(f"  Gender Accuracy:    {metrics['gender_accuracy']:.4f}")
    print(f"  Demographic Parity: {metrics['demographic_parity']:.4f}")
    print(f"  Female ICT Rate:    {metrics['female_prediction_rate']:.4f}")
    print(f"  Male ICT Rate:      {metrics['male_prediction_rate']:.4f}")
    print(f"{'='*60}\n")


def compare_models(baseline_metrics, adversarial_metrics):
    """
    Compare baseline vs adversarial model.
    
    Prints side-by-side comparison and improvement percentages.
    """
    print(f"\n{'='*80}")
    print(f"{'Metric':<30} {'Baseline':<15} {'Adversarial':<15} {'Change':<15}")
    print(f"{'='*80}")
    
    # Job performance
    job_f1_change = (adversarial_metrics['job_f1'] - baseline_metrics['job_f1']) / baseline_metrics['job_f1'] * 100
    print(f"{'Job F1 Score':<30} {baseline_metrics['job_f1']:<15.4f} "
          f"{adversarial_metrics['job_f1']:<15.4f} {job_f1_change:>+.2f}%")
    
    job_acc_change = (adversarial_metrics['job_accuracy'] - baseline_metrics['job_accuracy']) / baseline_metrics['job_accuracy'] * 100
    print(f"{'Job Accuracy':<30} {baseline_metrics['job_accuracy']:<15.4f} "
          f"{adversarial_metrics['job_accuracy']:<15.4f} {job_acc_change:>+.2f}%")
    
    # Fairness metrics
    gender_change = (adversarial_metrics['gender_accuracy'] - baseline_metrics['gender_accuracy']) / baseline_metrics['gender_accuracy'] * 100
    print(f"{'Gender Accuracy':<30} {baseline_metrics['gender_accuracy']:<15.4f} "
          f"{adversarial_metrics['gender_accuracy']:<15.4f} {gender_change:>+.2f}%")
    
    parity_change = (adversarial_metrics['demographic_parity'] - baseline_metrics['demographic_parity']) / baseline_metrics['demographic_parity'] * 100
    print(f"{'Demographic Parity':<30} {baseline_metrics['demographic_parity']:<15.4f} "
          f"{adversarial_metrics['demographic_parity']:<15.4f} {parity_change:>+.2f}%")
    
    print(f"{'='*80}")
    print(f"\nKey Findings:")
    print(f"  • Job performance retained: {100 + job_f1_change:.1f}% of baseline")
    print(f"  • Bias reduction: {abs(gender_change):.1f}% decrease in gender predictability")
    print(f"  • Fairness improvement: {abs(parity_change):.1f}% reduction in demographic disparity")
    print(f"{'='*80}\n")


def calculate_confusion_matrix(y_true, y_pred, labels=['non-ICT', 'ICT']):
    """Calculate and format confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nConfusion Matrix:")
    print(f"{'':>12} {labels[0]:<12} {labels[1]:<12}")
    for i, label in enumerate(labels):
        print(f"{label:<12} {cm[i][0]:<12} {cm[i][1]:<12}")
    
    return cm


def generate_classification_report(y_true, y_pred, labels=['non-ICT', 'ICT']):
    """Generate detailed classification report."""
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(f"\nClassification Report:")
    print(report)
    return report
