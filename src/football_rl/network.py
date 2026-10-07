"""Contrat PyTorch sans imposer un algorithme ou une architecture de réseau."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class FootballNetwork(nn.Module, ABC):
    """Base des réseaux : dimension d'entrée fixe pour un schéma d'observation."""

    def __init__(self, observation_size: int) -> None:
        super().__init__()
        self.observation_size = observation_size

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sorties nommées propres au réseau, calculées sur un lot d'observations."""
        raise NotImplementedError
