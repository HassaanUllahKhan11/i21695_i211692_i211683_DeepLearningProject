"""
Main pipeline for LLMEmbed reproduction.
Orchestrates the complete workflow: dataset loading, embedding extraction,
fusion, classification, and evaluation.
"""

import numpy as np
import torch
from embedder import LLMEmbedder
from fusion import apply_fusion
from classifier import train_classifier
from dataset_loader import load_dataset_by_name, get_num_classes
from utils import print_metrics, set_random_seed, format_results_summary
import config


def run_reproduction():
    """
    Main function to run the complete LLMEmbed reproduction pipeline.
    """
    print("="*80)
    print("LLMEmbed Reproduction - ACL 2024")
    print("="*80)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Fusion Method: {config.FUSION_METHOD}")
    print(f"Classifier: {config.CLASSIFIER_TYPE}")
    print(f"Extraction Method: {config.EXTRACTION_METHOD}")
    print("="*80 + "\n")
    
    # Set random seed for reproducibility
    set_random_seed()
    
    # Initialize embedder
    print("Initializing embedder...")
    embedder = LLMEmbedder()
    print()
    
    # Store results
    all_results = {}
    
    # Process each dataset
    for dataset_name in config.DATASETS:
        print(f"\n{'='*80}")
        print(f"Processing Dataset: {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        # Load dataset
        train_texts, train_labels, test_texts, test_labels = load_dataset_by_name(dataset_name)
        num_classes = get_num_classes(dataset_name)
        
        # Limit dataset size for faster reproduction (optional - remove for full run)
        # Uncomment these lines to use smaller subsets for testing
        # max_train_samples = 5000
        # max_test_samples = 1000
        # if len(train_texts) > max_train_samples:
        #     train_texts = train_texts[:max_train_samples]
        #     train_labels = train_labels[:max_train_samples]
        # if len(test_texts) > max_test_samples:
        #     test_texts = test_texts[:max_test_samples]
        #     test_labels = test_labels[:max_test_samples]
        
        # Extract embeddings
        print("\nExtracting embeddings from training set...")
        if config.FUSION_METHOD in ["mean", "concatenation", "layer_weighted"]:
            # Extract from multiple layers for fusion
            train_layer_embeddings = embedder.extract_multi_layer_embeddings(
                train_texts,
                layer_indices=config.LAYER_INDICES
            )
            print(f"Extracted embeddings from {len(train_layer_embeddings)} layers")
            print(f"Embedding dimension per layer: {train_layer_embeddings[0].shape[1]}")
        else:
            # Single layer extraction
            train_embeddings = embedder.extract_embeddings(train_texts)
            train_layer_embeddings = [train_embeddings]
        
        print("\nExtracting embeddings from test set...")
        if config.FUSION_METHOD in ["mean", "concatenation", "layer_weighted"]:
            # Extract from multiple layers for fusion
            test_layer_embeddings = embedder.extract_multi_layer_embeddings(
                test_texts,
                layer_indices=config.LAYER_INDICES
            )
        else:
            # Single layer extraction
            test_embeddings = embedder.extract_embeddings(test_texts)
            test_layer_embeddings = [test_embeddings]
        
        # Apply fusion
        print(f"\nApplying fusion method: {config.FUSION_METHOD}")
        if config.FUSION_METHOD == "layer_weighted":
            # For learned fusion, use training data to learn weights
            train_fused = apply_fusion(
                train_layer_embeddings,
                method=config.FUSION_METHOD,
                train_embeddings=train_layer_embeddings,
                train_labels=train_labels
            )
            # Apply same learned fusion to test
            test_fused = apply_fusion(
                test_layer_embeddings,
                method=config.FUSION_METHOD,
                train_embeddings=train_layer_embeddings,
                train_labels=train_labels
            )
        else:
            train_fused = apply_fusion(train_layer_embeddings, method=config.FUSION_METHOD)
            test_fused = apply_fusion(test_layer_embeddings, method=config.FUSION_METHOD)
        
        print(f"Fused embedding dimension: {train_fused.shape[1]}")
        
        # Train and evaluate classifier
        print(f"\nTraining {config.CLASSIFIER_TYPE} classifier...")
        model, metrics = train_classifier(
            train_fused,
            train_labels,
            test_fused,
            test_labels,
            classifier_type=config.CLASSIFIER_TYPE,
            num_classes=num_classes
        )
        
        # Print results
        print_metrics(metrics, dataset_name=dataset_name, classifier_type=config.CLASSIFIER_TYPE)
        
        # Store results
        all_results[dataset_name] = {
            config.CLASSIFIER_TYPE: metrics
        }
    
    # Print summary
    print(format_results_summary(all_results))
    
    print("Reproduction complete!")


if __name__ == "__main__":
    run_reproduction()

