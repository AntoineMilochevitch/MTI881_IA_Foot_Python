"""Interfaces des agents d'apprentissage par renforcement."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from football_rl.types import FootballAction, Observation


class FootballAgent(ABC):
    """Interface commune aux futurs agents RL implémentés avec PyTorch."""

    @abstractmethod
    def select_action(self, observation: Observation) -> FootballAction:
        """Sélectionne une action à partir d'une observation normalisée."""
        pass

    @abstractmethod
    def observe(
        self,
        observation: Observation,
        action: FootballAction,
        reward: float,
        next_observation: Observation,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Enregistre une transition pour la future phase d'apprentissage."""
        pass

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Met à jour les paramètres de l'agent à partir des transitions."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Sauvegarde l'état entraînable de l'agent."""
        pass

    @abstractmethod
    def load(self, path: Path, device: torch.device) -> None:
        """Charge l'état entraînable de l'agent sur le périphérique demandé."""
        pass
