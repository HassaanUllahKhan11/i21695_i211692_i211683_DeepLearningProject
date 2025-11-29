"""
Advanced fusion and ensemble strategies for LLMEmbed.

This module adds:
  - mean pooling
  - concatenation
  - weighted layer fusion
and simple ensemble classifiers (majority vote / probability averaging).

These operate on top of the existing embedding outputs from Part 2
without modifying that pipeline.
"""

from typing import List, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import ClassifierMixin


def mean_pool(layers: List[np.ndarray]) -> np.ndarray:
    """Mean pooling across layers [num_layers, dim] -> [dim]."""
    return np.mean(np.stack(layers, axis=0), axis=0)


def concat_pool(layers: List[np.ndarray]) -> np.ndarray:
    """Concatenate layer embeddings [num_layers, dim] -> [num_layers * dim]."""
    return np.concatenate(layers, axis=-1)


def weighted_fusion(layers: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    Weighted fusion of layers with learnable or fixed weights.
    """
    w = np.array(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    stacked = np.stack(layers, axis=0)  # [L, D]
    return (w[:, None] * stacked).sum(axis=0)


def build_ensemble_classifiers(
    X: np.ndarray, y: np.ndarray
) -> Dict[str, ClassifierMixin]:
    """
    Train a small set of base classifiers on top of embeddings.
    """
    clf_lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf_lr.fit(X, y)
    return {"logreg": clf_lr}


def ensemble_predict_proba(
    classifiers: Dict[str, ClassifierMixin], X: np.ndarray
) -> np.ndarray:
    """
    Probability averaging ensemble.
    """
    probas = [clf.predict_proba(X) for clf in classifiers.values()]
    return np.mean(np.stack(probas, axis=0), axis=0)


def ensemble_predict_majority(
    classifiers: Dict[str, ClassifierMixin], X: np.ndarray
) -> np.ndarray:
    """
    Majority voting ensemble.
    """
    preds = [clf.predict(X) for clf in classifiers.values()]
    preds = np.stack(preds, axis=0)  # [num_clf, N]
    # Mode along classifier axis
    out = []
    for i in range(preds.shape[1]):
        values, counts = np.unique(preds[:, i], return_counts=True)
        out.append(values[counts.argmax()])
    return np.array(out)


def evaluate_ensemble(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute accuracy and macro F1 for ensemble outputs.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


