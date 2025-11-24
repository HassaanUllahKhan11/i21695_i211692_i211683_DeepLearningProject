"""
Classifier implementations for LLMEmbed reproduction.
Supports Logistic Regression and 2-layer MLP.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import config


class TextDataset(Dataset):
    """PyTorch dataset for embeddings and labels."""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    """
    2-layer feedforward MLP classifier.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = None, dropout: float = None):
        """
        Initialize MLP classifier.
        
        Args:
            input_dim: Input embedding dimension
            num_classes: Number of classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(MLPClassifier, self).__init__()
        hidden_dim = hidden_dim or config.MLP_HIDDEN_DIM
        dropout = dropout if dropout is not None else config.MLP_DROPOUT
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 1000
) -> Tuple[LogisticRegression, dict]:
    """
    Train and evaluate a Logistic Regression classifier.
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        y_test: Test labels
        max_iter: Maximum iterations for training
        
    Returns:
        Trained model and evaluation metrics
    """
    print("Training Logistic Regression classifier...")
    
    # Train classifier
    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=1 if config.VERBOSE else 0
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }
    
    return clf, metrics


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    hidden_dim: int = None,
    dropout: float = None,
    learning_rate: float = None,
    epochs: int = None,
    batch_size: int = None,
    device: str = None
) -> Tuple[MLPClassifier, dict]:
    """
    Train and evaluate a 2-layer MLP classifier.
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        y_test: Test labels
        num_classes: Number of classes
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
        learning_rate: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to run on
        
    Returns:
        Trained model and evaluation metrics
    """
    print("Training 2-layer MLP classifier...")
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = hidden_dim or config.MLP_HIDDEN_DIM
    dropout = dropout if dropout is not None else config.MLP_DROPOUT
    learning_rate = learning_rate or config.MLP_LEARNING_RATE
    epochs = epochs or config.MLP_EPOCHS
    batch_size = batch_size or config.MLP_BATCH_SIZE
    
    input_dim = X_train.shape[1]
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MLPClassifier(input_dim, num_classes, hidden_dim, dropout)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_embeddings)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if config.VERBOSE and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings = batch_embeddings.to(device)
            logits = model(batch_embeddings)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }
    
    return model, metrics


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str = None,
    num_classes: int = None,
    **kwargs
) -> Tuple[object, dict]:
    """
    Train a classifier based on the specified type.
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        y_test: Test labels
        classifier_type: Type of classifier ('logistic_regression' or 'mlp')
        num_classes: Number of classes (required for MLP)
        **kwargs: Additional arguments for classifier training
        
    Returns:
        Trained model and evaluation metrics
    """
    classifier_type = classifier_type or config.CLASSIFIER_TYPE
    
    if classifier_type == "logistic_regression":
        return train_logistic_regression(X_train, y_train, X_test, y_test, **kwargs)
    elif classifier_type == "mlp":
        if num_classes is None:
            num_classes = len(np.unique(y_train))
        return train_mlp(X_train, y_train, X_test, y_test, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

