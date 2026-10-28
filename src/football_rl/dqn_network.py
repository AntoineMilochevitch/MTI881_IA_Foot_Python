"""Réseau de neurones MLP pour le DQN.

Architecture simple à couches entièrement connectées avec activation ReLU.
Prend un vecteur d'observation et produit une Q-valeur par action discrète.
"""

import torch
from torch import nn

from football_rl.network import FootballNetwork


class DQNNetwork(FootballNetwork):
    """Perceptron multicouche estimant Q(s, a) pour chaque action discrète.

    L'architecture par défaut comporte deux couches cachées de 256 neurones.
    Aucune normalisation ni dropout n'est appliqué ; le tampon de rejeu
    et la mise à jour du réseau cible assurent la stabilité.

    Args:
        observation_size: Dimension du vecteur d'observation.
        num_actions: Nombre total d'actions discrètes.
        hidden_size: Nombre de neurones par couche cachée.
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: int = 256,
    ) -> None:
        super().__init__(observation_size)
        self.num_actions = num_actions

        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Retourne les Q-valeurs sous la clé ``"q_values"``.

        Args:
            observations: Tenseur de forme ``(batch, observation_size)``.

        Returns:
            Dictionnaire avec ``"q_values"`` de forme ``(batch, num_actions)``.
        """
        return {"q_values": self.layers(observations)}
