"""Contrat Gymnasium conservé en complément de l'entraînement par lots."""

from typing import Any
from numpy.typing import NDArray

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

    def __init__(self, layout: ObservationLayout, *, enable_tackle: bool = False, max_steps: int = 200) -> None:
        super().__init__()
        self.layout = layout
        self.max_steps = max_steps #Limite artificielle
        self._step_count = 0

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


    def _mock_observation(self) -> Observation:
        """Construit une observation factice"""
        rndm = self.np_random
        ball = rndm.uniform(-1.0, 1.0, size=6).astype(np.float32)
        agent = rndm.uniform(-1.0, 1.0, size=3).astype(np.float32)

        def players_block(count: int, max_slots: int) -> NDArray[np.float32]:
            block = np.zeros((max_slots, 7), dtype=np.float32)
            present = min(count, max_slots)
            if present > 0:
                block[:present, :6] = rndm.uniform(-1.0, 1.0, size=(present, 6))
                block[:present, 6] = 1.0
            return block.reshape(-1)

        n_teammates = int(rndm.integers(0, self.layout.max_teammates + 1))
        n_opponents = int(rndm.integers(1, self.layout.max_opponents + 1))
        teammates = players_block(n_teammates, self.layout.max_teammates)
        opponents = players_block(n_opponents, self.layout.max_opponents)

        goals = np.empty(10, dtype=np.float32) #Eviter que largeur et hauteur soient négatifs
        for i in range(2):
            offset = i * 5
            goals[offset:offset + 3] = rndm.uniform(-1.0, 1.0, size=3)
            goals[offset + 3:offset + 5] = rndm.uniform(0.0, 1.0, size=2)

        match = rndm.uniform(-1.0, 1.0, size=3).astype(np.float32)

        obs = np.concatenate([ball, agent, teammates, opponents, goals, match])
        assert obs.shape == (self.layout.size,), "Incohérence entre le mock et ObservationLayout.size"
        return obs.astype(np.float32)


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, Info]:
        """Demarre un nouvel episode factice."""
        super().reset(seed=seed)
        self._step_count = 0
        observation = self._mock_observation()
        info: Info = {}
        return observation, info

    def step(self, action: FootballAction) -> tuple[Observation, float, bool, bool, Info]:
        """Avance d'un pas (factice, pas d'effet)"""
        self._step_count += 1
        observation = self._mock_observation()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps
        info: Info = {}
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Unity assure le rendu."""

    def close(self) -> None:
        """Aucune ressource réseau n'est ouverte."""
