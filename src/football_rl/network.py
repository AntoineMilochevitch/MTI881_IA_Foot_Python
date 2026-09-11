"""Interfaces PyTorch des futurs réseaux de politiques et de valeurs."""

import torch
from torch import nn


class ActorCriticNetwork(nn.Module):
    """Contrat d'un réseau PyTorch partagé par politique et estimateur de valeur."""

    def __init__(self, observation_size: int) -> None:
        """Initialise le futur réseau à partir de la dimension d'observation."""
        pass

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Retourne les sorties de politique hybrides et la valeur d'état."""
        pass
