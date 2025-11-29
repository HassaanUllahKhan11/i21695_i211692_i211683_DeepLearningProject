"""
Additional lightweight classifiers for LLMEmbed.

Implements:
  - Support Vector Machine (linear kernel)
  - Deeper MLP classifier (3 layers)

These models sit on top of frozen embeddings and can be toggled via config.
"""

from typing import Dict

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


def train_svm_classifier(X: np.ndarray, y: np.ndarray) -> LinearSVC:
    clf = LinearSVC()
    clf.fit(X, y)
    return clf


def train_deep_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 512,
) -> MLPClassifier:
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_dim, hidden_dim, hidden_dim),
        activation="relu",
        alpha=1e-4,
        max_iter=50,
        batch_size=64,
        learning_rate_init=1e-3,
    )
    clf.fit(X, y)
    return clf


def evaluate_classifier(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    y_pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }


