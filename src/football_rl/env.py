"""Environnement Gymnasium connecté au moteur Unity."""

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from football_rl.types import FootballAction, Info, Observation


class Football3DEnv(gym.Env[Observation, FootballAction]):
    """Contrat Gymnasium pour un match de football 3D entièrement observable.

    Unity reste l'unique source de vérité pour la simulation. Cette classe ne
    contient aucune physique ni simulation locale ; ses méthodes seront reliées
    au futur client TCP/UDP.
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        """Déclare les espaces normalisés du scénario initial 1v1."""
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(24,),
            dtype=np.float32,
        )
        self.action_space = spaces.Dict(
            {
                "move_target": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "shoot": spaces.Discrete(2),
                "pass_action": spaces.Discrete(2),
                "jump": spaces.Discrete(2),
                "tackle": spaces.Discrete(2),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, Info]:
        """Réinitialise un épisode en demandant un nouvel état à Unity."""
        pass

    def step(
        self,
        action: FootballAction,
    ) -> tuple[Observation, float, bool, bool, Info]:
        """Envoie une action à Unity et récupère la transition suivante."""
        pass

    def render(self) -> None:
        """Laisse Unity effectuer le rendu du match."""
        pass

    def close(self) -> None:
        """Ferme les futures ressources de communication avec Unity."""
        pass
