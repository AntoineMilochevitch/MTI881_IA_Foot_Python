"""Contrat PyTorch sans imposer un algorithme ou une architecture de réseau."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class FootballNetwork(nn.Module, ABC):
    """Base des réseaux comparés, avec une entrée fixe incluant les masques."""

    def __init__(self, observation_size: int) -> None:
        super().__init__()
        self.observation_size = observation_size

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sorties et exploitation des masques à définir selon l'algorithme."""
        raise NotImplementedError
