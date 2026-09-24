"""Interface commune aux algorithmes RL entraînés avec PyTorch."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from football_rl.types import EpisodeBatch, FootballAction, ModelArtifact, Observation


class FootballAgent(ABC):
    """Apprentissage sur les lots Unity, puis export du réseau mis à jour."""

    @abstractmethod
    def select_action(self, observation: Observation) -> FootballAction:
        """Interface Python de politique ; l'exécution en jeu se fait dans Unity."""
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: EpisodeBatch) -> dict[str, float]:
        """Entraîne le réseau sur un lot et retourne les métriques d'apprentissage."""
        raise NotImplementedError

    @abstractmethod
    def export_model(self) -> ModelArtifact:
        """Exporte le réseau pour Unity dans un format restant à définir."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        """Sauvegarde l'état entraînable de l'agent."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path, device: torch.device) -> None:
        """Charge l'état entraînable de l'agent sur le périphérique demandé."""
        raise NotImplementedError
