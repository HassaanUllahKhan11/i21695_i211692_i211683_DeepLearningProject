"""
Part 3: Improvements pipeline for LLMEmbed.

This script keeps the original Part 2 reproduction intact and adds
optional improvements controlled via flags in `config.py`:

  - Contrastive embedding post-training (Mini-SimCSE style)
  - LoRA adapter fine-tuning
  - Advanced fusion + ensemble
  - Better classifiers (SVM / deep MLP)
  - Data augmentation
"""

import numpy as np
import torch

import config
from embedder import LLMEmbedder
from fusion import apply_fusion
from classifier import train_classifier
from dataset_loader import load_dataset_by_name, get_num_classes
from utils import print_metrics, set_random_seed, format_results_summary

from llmembed_reproduction.improvements.contrastive import (
    run_contrastive_post_training,
)
from llmembed_reproduction.improvements.lora_finetune import (
    lora_finetune_last_layers,
)
from llmembed_reproduction.improvements.advanced_fusion import (
    mean_pool,
    concat_pool,
    weighted_fusion,
    build_ensemble_classifiers,
    ensemble_predict_proba,
    ensemble_predict_majority,
    evaluate_ensemble,
)
from llmembed_reproduction.improvements.better_classifier import (
    train_svm_classifier,
    train_deep_mlp,
    evaluate_classifier,
)
from llmembed_reproduction.improvements.augmentation import augment_dataset


def run_improvements():
    """
    Main function to run LLMEmbed with optional improvements enabled via config.
    """
    print("=" * 80)
    print("LLMEmbed Improvements - Part 3")
    print("=" * 80)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Fusion Method: {config.FUSION_METHOD}")
    print(f"Classifier: {config.CLASSIFIER_TYPE}")
    print(f"USE_CONTRASTIVE: {config.USE_CONTRASTIVE}")
    print(f"USE_LORA: {config.USE_LORA}")
    print(f"USE_AUGMENTATION: {config.USE_AUGMENTATION}")
    print(f"USE_ADVANCED_FUSION: {config.USE_ADVANCED_FUSION}")
    print(f"USE_BETTER_CLASSIFIER: {config.USE_BETTER_CLASSIFIER}")
    print("=" * 80 + "\n")

    set_random_seed()
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    else:
        print("Warning: No GPU detected. This will be very slow on full dataset.")
        print("Consider using GPU or reducing dataset size.")

    # Load datasets
    all_results = {}

    for dataset_name in config.DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset (Improvements): {dataset_name.upper()}")
        print(f"{'='*80}\n")

        train_texts, train_labels, test_texts, test_labels = load_dataset_by_name(
            dataset_name
        )
        num_classes = get_num_classes(dataset_name)

        # Optional data augmentation
        if config.USE_AUGMENTATION:
            print("Applying lightweight data augmentation to training set...")
            # Ensure plain Python lists (HF Datasets may return Column objects)
            base_texts = list(train_texts)
            base_labels = list(train_labels)
            train_texts, train_labels = augment_dataset(
                base_texts, base_labels, max_ratio=0.2
            )
            train_labels = np.array(train_labels)

        # Optional LoRA fine-tuning
        if config.USE_LORA:
            print("Running LoRA fine-tuning on a subset of training texts...")
            subset_texts = train_texts[:2000] if len(train_texts) > 2000 else train_texts
            lora_model = lora_finetune_last_layers(
                model_name=config.MODEL_PATH_MAP[config.MODEL_NAME],  # HF name
                train_texts=subset_texts,
            )
            # Use the LoRA-adapted model in a custom embedder instance
            embedder = LLMEmbedder()
            embedder.model = lora_model.to(embedder.device)
        else:
            embedder = LLMEmbedder()

        # Multi-layer embeddings for fusion
        print(f"Extracting multi-layer embeddings (train: {len(train_texts)} samples)...")
        train_layer_embeddings = embedder.extract_multi_layer_embeddings(
            train_texts,
            layer_indices=config.LAYER_INDICES,
        )
        print(f"Extracting multi-layer embeddings (test: {len(test_texts)} samples)...")
        test_layer_embeddings = embedder.extract_multi_layer_embeddings(
            test_texts,
            layer_indices=config.LAYER_INDICES,
        )

        # Base fusion (same as Part 2)
        print("\nApplying base fusion method from Part 2...")
        if config.FUSION_METHOD == "layer_weighted":
            train_fused = apply_fusion(
                train_layer_embeddings,
                method=config.FUSION_METHOD,
                train_embeddings=train_layer_embeddings,
                train_labels=train_labels,
            )
            test_fused = apply_fusion(
                test_layer_embeddings,
                method=config.FUSION_METHOD,
                train_embeddings=train_layer_embeddings,
                train_labels=train_labels,
            )
        else:
            train_fused = apply_fusion(
                train_layer_embeddings, method=config.FUSION_METHOD
            )
            test_fused = apply_fusion(
                test_layer_embeddings, method=config.FUSION_METHOD
            )

        results_for_dataset = {}

        # Baseline (Part 2 style) classifier for reference
        print("\nTraining baseline classifier on fused embeddings (reference)...")
        base_model, base_metrics = train_classifier(
            train_fused,
            train_labels,
            test_fused,
            test_labels,
            classifier_type=config.CLASSIFIER_TYPE,
            num_classes=num_classes,
        )
        print_metrics(
            base_metrics,
            dataset_name=dataset_name + " (baseline fused)",
            classifier_type=config.CLASSIFIER_TYPE,
        )
        results_for_dataset["baseline"] = base_metrics

        # Contrastive post‑training
        if config.USE_CONTRASTIVE:
            print("\nRunning contrastive embedding post-training (Mini-SimCSE style)...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device} for contrastive training")
            # Use larger batch size and fewer epochs for efficiency
            from llmembed_reproduction.improvements.contrastive import ContrastiveTrainer
            trainer = ContrastiveTrainer(embedding_dim=train_fused.shape[1], device=device)
            # Use smaller batch size for CPU
            contrastive_batch_size = 64 if device == 'cpu' else 256
            trainer.fit(torch.from_numpy(train_fused).float(), batch_size=contrastive_batch_size, epochs=1)
            train_refined = trainer.refine_embeddings(torch.from_numpy(train_fused).float()).numpy()
            test_refined = trainer.refine_embeddings(torch.from_numpy(test_fused).float()).numpy()

            model_c, metrics_c = train_classifier(
                train_refined,
                train_labels,
                test_refined,
                test_labels,
                classifier_type=config.CLASSIFIER_TYPE,
                num_classes=num_classes,
            )
            print_metrics(
                metrics_c,
                dataset_name=dataset_name + " (contrastive)",
                classifier_type=config.CLASSIFIER_TYPE,
            )
            results_for_dataset["contrastive"] = metrics_c

        # Advanced fusion + ensemble
        if config.USE_ADVANCED_FUSION:
            print("\nApplying advanced fusion strategies + ensemble...")

            # Optimized: use vectorized operations instead of loops
            train_layers = train_layer_embeddings
            test_layers = test_layer_embeddings
            
            # Mean fusion (vectorized)
            tr_mean = np.mean(np.stack(train_layers, axis=0), axis=0)  # [n_layers, n_samples, dim] -> [n_samples, dim]
            te_mean = np.mean(np.stack(test_layers, axis=0), axis=0)

            # Concatenation (vectorized)
            tr_concat = np.concatenate(train_layers, axis=1)  # [n_samples, n_layers * dim]
            te_concat = np.concatenate(test_layers, axis=1)

            # Weighted fusion with uniform weights (vectorized)
            L = len(train_layers)
            w = np.array([1.0 / L] * L, dtype=np.float32)
            w = w / w.sum()
            tr_weighted = np.sum(np.stack(train_layers, axis=0) * w[:, None, None], axis=0)
            te_weighted = np.sum(np.stack(test_layers, axis=0) * w[:, None, None], axis=0)

            # Train base classifier on each fusion
            clf_mean, _ = train_classifier(
                tr_mean,
                train_labels,
                te_mean,
                test_labels,
                classifier_type=config.CLASSIFIER_TYPE,
                num_classes=num_classes,
            )
            clf_concat, _ = train_classifier(
                tr_concat,
                train_labels,
                te_concat,
                test_labels,
                classifier_type=config.CLASSIFIER_TYPE,
                num_classes=num_classes,
            )
            clf_weighted, _ = train_classifier(
                tr_weighted,
                train_labels,
                te_weighted,
                test_labels,
                classifier_type=config.CLASSIFIER_TYPE,
                num_classes=num_classes,
            )

            # Build ensemble from the best fusion (for simplicity, use concat)
            ensemble_clfs = build_ensemble_classifiers(tr_concat, train_labels)

            # Probability averaging
            proba = ensemble_predict_proba(ensemble_clfs, te_concat)
            y_pred_proba = np.argmax(proba, axis=1)
            metrics_ens_p = evaluate_ensemble(test_labels, y_pred_proba)

            # Majority voting
            y_pred_vote = ensemble_predict_majority(ensemble_clfs, te_concat)
            metrics_ens_v = evaluate_ensemble(test_labels, y_pred_vote)

            print_metrics(
                {
                    "accuracy": metrics_ens_p["accuracy"],
                    "f1_macro": metrics_ens_p["macro_f1"],
                    "confusion_matrix": np.zeros(
                        (num_classes, num_classes), dtype=int
                    ),
                },
                dataset_name=dataset_name + " (ensemble proba)",
                classifier_type="ensemble_proba",
            )
            print_metrics(
                {
                    "accuracy": metrics_ens_v["accuracy"],
                    "f1_macro": metrics_ens_v["macro_f1"],
                    "confusion_matrix": np.zeros(
                        (num_classes, num_classes), dtype=int
                    ),
                },
                dataset_name=dataset_name + " (ensemble vote)",
                classifier_type="ensemble_vote",
            )

            # Convert macro_f1 to f1_macro for consistency
            results_for_dataset["ensemble_proba"] = {
                "accuracy": metrics_ens_p["accuracy"],
                "f1_macro": metrics_ens_p["macro_f1"]
            }
            results_for_dataset["ensemble_vote"] = {
                "accuracy": metrics_ens_v["accuracy"],
                "f1_macro": metrics_ens_v["macro_f1"]
            }

        # Better classifiers
        if config.USE_BETTER_CLASSIFIER:
            print("\nTraining additional classifiers (SVM / deep MLP)...")
            svm_clf = train_svm_classifier(train_fused, train_labels)
            svm_metrics = evaluate_classifier(svm_clf, test_fused, test_labels)

            mlp_clf = train_deep_mlp(train_fused, train_labels)
            mlp_metrics = evaluate_classifier(mlp_clf, test_fused, test_labels)

            print_metrics(
                {
                    "accuracy": svm_metrics["accuracy"],
                    "f1_macro": svm_metrics["macro_f1"],
                    "confusion_matrix": np.zeros(
                        (num_classes, num_classes), dtype=int
                    ),
                },
                dataset_name=dataset_name + " (SVM)",
                classifier_type="svm",
            )
            print_metrics(
                {
                    "accuracy": mlp_metrics["accuracy"],
                    "f1_macro": mlp_metrics["macro_f1"],
                    "confusion_matrix": np.zeros(
                        (num_classes, num_classes), dtype=int
                    ),
                },
                dataset_name=dataset_name + " (deep MLP)",
                classifier_type="deep_mlp",
            )

            # Convert macro_f1 to f1_macro for consistency
            results_for_dataset["svm"] = {
                "accuracy": svm_metrics["accuracy"],
                "f1_macro": svm_metrics["macro_f1"]
            }
            results_for_dataset["deep_mlp"] = {
                "accuracy": mlp_metrics["accuracy"],
                "f1_macro": mlp_metrics["macro_f1"]
            }

        all_results[dataset_name] = results_for_dataset

    print(format_results_summary(all_results))
    print("Improvements run complete!")


if __name__ == "__main__":
    run_improvements()


