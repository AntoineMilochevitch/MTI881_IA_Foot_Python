"""Point d'entrée de l'entraînement RL."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Paramètres déclaratifs d'une future session d'entraînement."""

    total_steps: int
    seed: int
    checkpoint_directory: Path


def train(config: TrainingConfig) -> None:
    """Orchestre une future session d'entraînement avec Unity et PyTorch."""
    pass


def main() -> None:
    """Expose le point d'entrée de ligne de commande du projet."""
    pass
