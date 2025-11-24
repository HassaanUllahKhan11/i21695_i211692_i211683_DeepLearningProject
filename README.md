# LLMEmbed Reproduction

This project reproduces the methodology from the ACL 2024 paper "LLMEmbed: Rethinking Lightweight LLM's Genuine Function in Text Classification" (https://aclanthology.org/2024.acl-long.433/).

## Overview

This is a faithful reproduction of the LLMEmbed method, which:
- Uses lightweight open-source LLMs (e.g., LLaMA-3-Instruct 8B, Mistral-7B-Instruct, Phi-2)
- Extracts sentence/document embeddings from hidden states
- Applies embedding fusion strategies (concatenation, averaging, layer fusion)
- Trains a simple classifier (Logistic Regression or 2-layer MLP) on the embeddings
- Evaluates on standard datasets (AG News, SST-2)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Full Reproduction

Run the full reproduction pipeline:

```bash
python run_reproduction.py
```

This will:
1. Load the datasets (AG News and SST-2)
2. Extract embeddings using a lightweight LLM
3. Apply fusion methods
4. Train and evaluate classifiers
5. Print accuracy, macro F1 score, and confusion matrices

### Minimal Test (Quick Verification)

For quick testing with limited disk space or to verify the setup:

```bash
python run_minimal_test.py
```

This uses very small subsets (100 train / 20 test samples) to quickly verify the pipeline works.

### Test Mode

The code automatically handles disk space issues. If dataset download fails due to insufficient disk space, it will use minimal synthetic data for testing. You can also enable test mode in `config.py`:

```python
TEST_MODE = True  # Use smaller subsets
MAX_TRAIN_SAMPLES = 1000  # Limit training samples
MAX_TEST_SAMPLES = 200   # Limit test samples
```

## Configuration

Edit `config.py` to adjust:
- Model selection (LLaMA-3, Mistral, Phi-2, etc.)
- Dataset selection
- Fusion method
- Classifier type
- Hyperparameters

## Project Structure

- `embedder.py`: Loads LLM and extracts embeddings from hidden states
- `fusion.py`: Implements fusion methods (mean, concatenation, layer-wise)
- `classifier.py`: Implements Logistic Regression and 2-layer MLP classifiers
- `dataset_loader.py`: Loads AG News and SST-2 datasets from HuggingFace
- `run_reproduction.py`: Main pipeline orchestrator
- `utils.py`: Helper functions for evaluation and preprocessing
- `config.py`: Configuration settings

## Notes

- This is a faithful reproduction without improvements or modifications
- Code is designed to run on modest GPU or CPU
- Uses 4-bit or 8-bit quantization if GPU RAM is limited
- Runtime should be under 1 hour

## Model Access

Some models (e.g., LLaMA-3, LLaMA-2) require authentication from HuggingFace. To use these models:
1. Create a HuggingFace account and request access to the model
2. Login using: `huggingface-cli login`
3. Or set your token: `export HF_TOKEN=your_token_here`

For models that don't require authentication (e.g., Phi-2, Mistral), you can use them directly.

