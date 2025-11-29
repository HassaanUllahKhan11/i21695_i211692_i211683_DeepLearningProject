"""
Simple data augmentation utilities for text classification.

Implements light‑weight synonym replacement and random deletion
to expand the training set by ~20% when enabled.
"""

from typing import List, Tuple
import random


def random_deletion(tokens: List[str], p: float = 0.1) -> List[str]:
    """Randomly delete each token with probability p."""
    if len(tokens) == 1:
        return tokens
    out = []
    for t in tokens:
        if random.random() > p:
            out.append(t)
    if not out:
        return [tokens[random.randint(0, len(tokens) - 1)]]
    return out


def synonym_replace(tokens: List[str], synonym_dict: dict, p: float = 0.1) -> List[str]:
    """
    Replace tokens with predefined synonyms with probability p.

    This avoids any external API calls and keeps augmentation deterministic.
    """
    out = []
    for t in tokens:
        if t.lower() in synonym_dict and random.random() < p:
            out.append(random.choice(synonym_dict[t.lower()]))
        else:
            out.append(t)
    return out


def augment_dataset(
    texts: List[str],
    labels: List[int],
    max_ratio: float = 0.2,
) -> Tuple[List[str], List[int]]:
    """
    Augment a portion of the training data using simple operations.
    """
    n = len(texts)
    k = max(1, int(n * max_ratio))
    indices = random.sample(range(n), k)

    synonym_dict = {
        "good": ["great", "nice", "decent"],
        "bad": ["terrible", "awful", "poor"],
        "movie": ["film", "picture"],
        "story": ["plot", "narrative"],
    }

    aug_texts = []
    aug_labels = []
    for idx in indices:
        tokens = texts[idx].split()
        tokens = synonym_replace(tokens, synonym_dict, p=0.15)
        tokens = random_deletion(tokens, p=0.1)
        aug_texts.append(" ".join(tokens))
        aug_labels.append(labels[idx])

    return texts + aug_texts, labels + aug_labels


