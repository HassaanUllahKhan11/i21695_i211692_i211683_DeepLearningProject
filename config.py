"""
Configuration file for LLMEmbed reproduction.
Contains all hyperparameters and settings as described in the paper.
"""

# Model configuration
# Options: "llama3-8b", "mistral-7b", "phi-2", "llama2-7b"
MODEL_NAME = "phi-2"  # Using Phi-2 as default (smallest, fastest)
MODEL_PATH_MAP = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi-2": "microsoft/phi-2",
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf"
}

# Quantization settings (for limited GPU memory)
USE_QUANTIZATION = True
QUANTIZATION_BITS = 4  # 4-bit or 8-bit

# Embedding extraction settings
EXTRACTION_METHOD = "mean_pooling"  # Options: "mean_pooling", "cls_token", "last_token"
# Using last 4 layers for efficiency (layers 28-31 out of 32) - these are typically most informative
LAYER_INDICES = [28, 29, 30, 31]  # None means use all layers, or specify list like [20, 22, 24]

# Fusion method
# Options: "mean", "concatenation", "layer_weighted"
FUSION_METHOD = "mean"

# Classifier settings
CLASSIFIER_TYPE = "logistic_regression"  # Options: "logistic_regression", "mlp"
MLP_HIDDEN_DIM = 512
MLP_DROPOUT = 0.1
MLP_LEARNING_RATE = 1e-3
MLP_EPOCHS = 10
MLP_BATCH_SIZE = 32

# Dataset settings
DATASETS = ["ag_news", "sst2"]  # Options: "ag_news", "sst2"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32  # Batch size for embedding extraction (increased for faster processing)

# Test mode settings (for limited disk space or quick testing)
TEST_MODE = False  # Set to True to use smaller subsets for testing
MAX_TRAIN_SAMPLES = 1000  # Maximum training samples in test mode
MAX_TEST_SAMPLES = 200  # Maximum test samples in test mode

# Training settings
RANDOM_SEED = 42
TEST_SIZE = 0.2  # For train/test split if needed

# Output settings
VERBOSE = True
SAVE_EMBEDDINGS = False  # Set to True to save embeddings for later use

