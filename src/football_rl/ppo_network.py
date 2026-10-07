"""Acteur PPO compatible avec le décodeur direct Unity et critique PyTorch.

Les observations sont des lots de forme (B, D), où B est le nombre de
situations et D la dimension obs_size annoncée dans le hello Unity. Le critique
prédit un retour par situation ; l'acteur produit quatre moyennes continues et
trois logits binaires. Quatre log-écarts-types sont appris séparément du MLP.
L'échantillonnage Python, l'évaluation des actions et l'optimisation PPO restent
à construire. Voir README_PPO.md pour les contrats et les prochaines étapes.

Ce module ne normalise pas les données Unity et ne simule pas le match.
"""

import math

import torch
from torch import nn

from football_rl.network import FootballNetwork


class PPOValueNetwork(FootballNetwork):
    """Estime V(s), le retour actualisé attendu sous la politique courante.

    La sortie est un réel, pas une probabilité de victoire. Ce réseau sera
    entraîné en Python à partir de cibles construites avec les trajectoires.
    """

    def __init__(self, observation_size: int) -> None:
        super().__init__(observation_size)

        # Les deux couches cachées apprennent des combinaisons d'observations.
        # La sortie linéaire permet des retours négatifs ou supérieurs à 1.
        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transforme (B, D) en une valeur par observation, de forme (B,)."""
        value = self.network(x)

        # (B, 1) -> (B,) : conserve l'axe du lot, même lorsque B vaut 1.
        value = value.squeeze(-1)
        return {"value": value}


class PPOActorNetwork(FootballNetwork):
    """Politique PPO compatible avec l'espace d'actions Unity.

    Le réseau produit sept sorties brutes dans l'ordre attendu par Unity :

    0 : move_x
    1 : move_z
    2 : shoot
    3 : shoot_power
    4 : shoot_curve
    5 : tackle
    6 : jump

    Les commandes continues utilisent des distributions normales.
    Les commandes binaires utilisent des distributions de Bernoulli.
    """

    CONTINUOUS_INDICES = [0, 1, 3, 4]
    BINARY_INDICES = [2, 5, 6]

    def __init__(self, observation_size: int) -> None:
        super().__init__(observation_size)

        # Un seul MLP produit les sept paramètres dépendant de l'état.
        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 7)
        )

        # Log-écarts-types entraînables, indépendants de l'observation, pour les quatre
        # composantes continues :
        # move_x, move_z, shoot_power, shoot_curve.
        self.log_std = nn.Parameter(torch.full((len(self.CONTINUOUS_INDICES),), math.log(0.5)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Construit les paramètres des distributions de la politique.

        x :
            observations de forme (B, D)

        Retour :
            model_output       : (B, 7)
            continuous_mean    : (B, 4)
            continuous_log_std : (B, 4)
            binary_logits      : (B, 3)
        """

        model_output = self.network(x)

        continuous_mean = model_output[:, self.CONTINUOUS_INDICES]

        binary_logits = model_output[:, self.BINARY_INDICES]

        # Même bornage dans l'export : Unity reçoit ces valeurs, puis les exponentie.
        bounded_log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)

        # (4,) -> (B, 4)
        # Tous les éléments du batch utilisent les mêmes paramètres
        # log_std, car ceux-ci sont indépendants de l'observation.
        continuous_log_std = bounded_log_std.expand(
            x.shape[0], -1
        )

        return {
            "model_output": model_output,
            "continuous_mean": continuous_mean,
            "continuous_log_std": continuous_log_std,
            "binary_logits": binary_logits
        }
