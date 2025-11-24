"""
Utility functions for LLMEmbed reproduction.
Includes evaluation metrics, logging, and helper functions.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, Any
import config


def print_metrics(metrics: Dict[str, Any], dataset_name: str = "", classifier_type: str = ""):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing 'accuracy', 'f1_macro', and 'confusion_matrix'
        dataset_name: Name of the dataset
        classifier_type: Type of classifier used
    """
    print("\n" + "="*60)
    if dataset_name:
        print(f"Dataset: {dataset_name}")
    if classifier_type:
        print(f"Classifier: {classifier_type}")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['f1_macro']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*60 + "\n")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with accuracy, f1_macro, and confusion_matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }


def set_random_seed(seed: int = None):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    seed = seed or config.RANDOM_SEED
    import random
    import torch
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_results_summary(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Format results summary for all datasets and classifiers.
    
    Args:
        results: Nested dictionary with results
        
    Returns:
        Formatted string summary
    """
    summary = "\n" + "="*80 + "\n"
    summary += "LLMEmbed Reproduction Results Summary\n"
    summary += "="*80 + "\n\n"
    
    for dataset_name, dataset_results in results.items():
        summary += f"Dataset: {dataset_name}\n"
        summary += "-" * 80 + "\n"
        
        for classifier_type, metrics in dataset_results.items():
            summary += f"  Classifier: {classifier_type}\n"
            summary += f"    Accuracy: {metrics['accuracy']:.4f}\n"
            summary += f"    Macro F1: {metrics['f1_macro']:.4f}\n"
        
        summary += "\n"
    
    summary += "="*80 + "\n"
    return summary

