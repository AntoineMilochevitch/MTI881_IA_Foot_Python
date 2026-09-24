"""Contrat Gymnasium conservé en complément de l'entraînement par lots."""

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from football_rl.types import FootballAction, Info, Observation, ObservationLayout


class Football3DEnv(gym.Env[Observation, FootballAction]):
    """Interface standardisée avec Unity, sans simulation locale.

    Le projet combine une interface Gymnasium et une collecte d'épisodes dans Unity.
    Le raccordement de reset/step reste à définir ; ces appels ne constituent
    pas la boucle principale d'entraînement par lots.
    """

    metadata = {"render_modes": []}

    def __init__(self, layout: ObservationLayout, *, enable_tackle: bool = False) -> None:
        super().__init__()
        self.layout = layout
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(layout.size,), dtype=np.float32,
        )
        actions: dict[str, gym.Space] = {
            "move_target": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "shoot": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "jump": spaces.Discrete(2),
        }
        if enable_tackle:
            actions["tackle"] = spaces.Discrete(2)
        self.action_space = spaces.Dict(actions)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, Info]:
        """Demande un nouvel épisode à Unity via un raccordement à définir."""
        raise NotImplementedError("Le raccordement Gymnasium à Unity reste à définir.")

    def step(self, action: FootballAction) -> tuple[Observation, float, bool, bool, Info]:
        """Contrat standard d'interaction, distinct de la réception des lots."""
        raise NotImplementedError("Le raccordement Gymnasium à Unity reste à définir.")

    def render(self) -> None:
        """Unity assure le rendu."""

    def close(self) -> None:
        """Aucune ressource réseau n'est encore ouverte par cette interface."""
