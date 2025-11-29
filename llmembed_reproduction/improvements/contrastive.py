"""
Contrastive embedding post-training (Mini-SimCSE style).

This module adds a lightweight contrastive fine-tuning stage on top of the
existing LLM-based embeddings. It operates on sentence embeddings exported
from the base pipeline and refines them using in-batch negatives.

The core reproduction (Part 2) remains unchanged; this module is only used
when explicitly enabled via config flags.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ProjectionHead(nn.Module):
    """
    Small projection head applied on top of frozen sentence embeddings.

    This keeps the base LLM unchanged while allowing a light-weight
    contrastive adaptation of the embedding space.
    """

    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return nn.functional.normalize(x, p=2, dim=-1)


class ContrastiveTrainer:
    """
    Mini‑SimCSE‑style contrastive training on top of precomputed embeddings.

    Given a matrix of sentence embeddings, we apply light contrastive tuning
    with in‑batch negatives for 1–2 epochs.
    """

    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 256,
        temperature: float = 0.05,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        # Force GPU if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.projection = ProjectionHead(embedding_dim, projection_dim).to(self.device)
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(self.projection.parameters(), lr=lr)

    def _nt_xent_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        NT‑Xent loss using in‑batch negatives.

        We assume two augmented views are concatenated in the batch:
        [z1; z2] with batch size 2 * N.
        """
        batch_size = z.size(0) // 2
        z1, z2 = z[:batch_size], z[batch_size:]

        # Cosine similarity matrix
        sim = torch.matmul(z1, z2.t()) / self.temperature  # [N, N]
        labels = torch.arange(batch_size, device=z.device)
        loss_i = nn.functional.cross_entropy(sim, labels)
        loss_j = nn.functional.cross_entropy(sim.t(), labels)
        return (loss_i + loss_j) / 2.0

    def fit(
        self,
        embeddings: torch.Tensor,
        batch_size: int = 64,
        epochs: int = 2,
    ) -> None:
        """
        Contrastively train the projection head on top of static embeddings.

        Args:
            embeddings: Tensor of shape [num_examples, dim]
            batch_size: Mini‑batch size
            epochs: Number of epochs (1–2 as recommended)
        """
        embeddings = embeddings.to(self.device)

        # Create two noisy views by simple dropout noise on embeddings
        dataset = TensorDataset(embeddings)
        # Optimize DataLoader for GPU
        pin_memory = (self.device == 'cuda')
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            pin_memory=pin_memory,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

        self.projection.train()
        for _ in range(epochs):
            for (batch_emb,) in loader:
                # Two stochastic views via dropout noise
                noise1 = torch.randn_like(batch_emb) * 0.01
                noise2 = torch.randn_like(batch_emb) * 0.01
                e1 = batch_emb + noise1
                e2 = batch_emb + noise2

                z1 = self.projection(e1)
                z2 = self.projection(e2)
                z = torch.cat([z1, z2], dim=0)

                loss = self._nt_xent_loss(z)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def refine_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the learned projection to obtain refined embeddings.
        """
        self.projection.eval()
        embeddings = embeddings.to(self.device)
        z = self.projection(embeddings)
        return z.cpu()


def run_contrastive_post_training(
    base_embeddings: torch.Tensor,
    epochs: int = 2,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Convenience function to refine sentence embeddings using contrastive tuning.

    Returns:
        Refined embeddings with the same shape as input.
    """
    trainer = ContrastiveTrainer(embedding_dim=base_embeddings.size(-1))
    trainer.fit(base_embeddings, batch_size=batch_size, epochs=epochs)
    return trainer.refine_embeddings(base_embeddings)


