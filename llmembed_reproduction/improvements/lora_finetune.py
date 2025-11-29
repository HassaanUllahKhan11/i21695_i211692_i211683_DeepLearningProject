"""
LoRA adapter fine‑tuning for lightweight adaptation of the LLM.

This module attaches LoRA adapters to the last few transformer layers and
allows small‑scale fine‑tuning without modifying the core reproduction code.
"""

from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRAConfig:
    """
    Simple configuration holder for LoRA fine‑tuning.
    """

    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        num_train_epochs: int = 2,
        lr: float = 2e-4,
        batch_size: int = 8,
    ):
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.num_train_epochs = num_train_epochs
        self.lr = lr
        self.batch_size = batch_size


def add_lora_to_linear(layer: torch.nn.Linear, r: int, alpha: int) -> torch.nn.Module:
    """
    Wrap a Linear layer with a LoRA adapter (simple manual implementation).
    """
    in_features = layer.in_features
    out_features = layer.out_features

    # Freeze original weights
    for param in layer.parameters():
        param.requires_grad = False

    # LoRA parameters
    lora_A = torch.nn.Linear(in_features, r, bias=False)
    lora_B = torch.nn.Linear(r, out_features, bias=False)
    scaling = alpha / r

    class LoRALinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = layer
            self.lora_A = lora_A
            self.lora_B = lora_B

        def forward(self, x):
            return self.base(x) + scaling * self.lora_B(self.lora_A(x))

    return LoRALinear()


def attach_lora_adapters(model: torch.nn.Module, config: LoRAConfig) -> None:
    """
    Attach LoRA adapters to attention projections in the last few layers.
    """
    for name, module in model.named_modules():
        if any(t in name for t in config.target_modules) and isinstance(
            module, torch.nn.Linear
        ):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, add_lora_to_linear(module, config.r, config.alpha))


def lora_finetune_last_layers(
    model_name: str,
    train_texts: list,
    lora_config: Optional[LoRAConfig] = None,
    device: Optional[str] = None,
) -> AutoModelForCausalLM:
    """
    Apply LoRA fine‑tuning on the last few layers using a simple LM objective.

    This function returns a model with trained LoRA adapters that can then be
    reused by the existing embedding pipeline without code changes there.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lora_config = lora_config or LoRAConfig()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.train()

    attach_lora_adapters(model, lora_config)

    # Prepare dataset
    dataset = Dataset.from_dict({"text": train_texts})

    def tok_fn(batch):
        out = tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=128
        )
        out["labels"] = out["input_ids"].copy()
        return out

    dataset = dataset.map(tok_fn, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    loader = DataLoader(dataset, batch_size=lora_config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lora_config.lr
    )

    for _ in range(lora_config.num_train_epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


