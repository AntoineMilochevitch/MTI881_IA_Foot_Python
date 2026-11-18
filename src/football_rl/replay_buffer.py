"""Tampon de rejeu circulaire pour l'entraînement DQN.

Stocke les transitions sous forme de tableaux NumPy contigus pour un
échantillonnage rapide. Les observations, récompenses et drapeaux de
terminaison sont pré-alloués ; les actions sont des indices entiers
issus de la discrétisation.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(slots=True)
class BatchTensors:
    """Lot de transitions échantillonnées, déjà converties en tenseurs PyTorch."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    """Tampon circulaire à capacité fixe avec échantillonnage uniforme.

    Les transitions sont stockées dans des tableaux NumPy pré-alloués.
    L'échantillonnage retourne directement des tenseurs PyTorch sur le
    périphérique demandé.

    Args:
        capacity: Nombre maximum de transitions stockées.
        observation_size: Dimension du vecteur d'observation.
    """

    def __init__(self, capacity: int, observation_size: int, seed: int | None = None) -> None:
        self.capacity = capacity
        self._size = 0
        self._pos = 0
        # Générateur dédié : avec une graine, l'ordre des mini-lots est reproductible.
        self._rng = np.random.default_rng(seed)

        self._observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: NDArray[np.float32],
        action: int,
        reward: float,
        next_observation: NDArray[np.float32],
        terminated: bool,
    ) -> None:
        """Ajoute une transition au tampon, en écrasant la plus ancienne si plein."""
        self._observations[self._pos] = observation
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._next_observations[self._pos] = next_observation
        self._terminated[self._pos] = float(terminated)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        observations: NDArray[np.float32],
        actions: NDArray[np.integer],
        rewards: NDArray[np.float32],
        next_observations: NDArray[np.float32],
        terminated: NDArray[np.bool_],
    ) -> None:
        """Ajoute N transitions d'un coup, dans l'ordre (les plus anciennes sont écrasées en premier)."""
        n = len(actions)
        if n == 0:
            return
        if n > self.capacity:  # seules les plus récentes tiennent dans le tampon
            observations, actions, rewards = observations[-self.capacity:], actions[-self.capacity:], rewards[-self.capacity:]
            next_observations, terminated = next_observations[-self.capacity:], terminated[-self.capacity:]
            n = self.capacity
        indices = (self._pos + np.arange(n)) % self.capacity
        self._observations[indices] = observations
        self._actions[indices] = actions
        self._rewards[indices] = rewards
        self._next_observations[indices] = next_observations
        self._terminated[indices] = np.asarray(terminated, dtype=np.float32)

        self._pos = int((self._pos + n) % self.capacity)
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> BatchTensors:
        """Échantillonne un lot aléatoire et le place sur le périphérique demandé.

        Raises:
            ValueError: Si le tampon contient moins de transitions que ``batch_size``.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Le tampon contient {self._size} transitions, "
                f"mais {batch_size} sont demandées."
            )

        indices = self._rng.integers(0, self._size, size=batch_size)

        return BatchTensors(
            observations=torch.from_numpy(self._observations[indices]).to(device),
            actions=torch.from_numpy(self._actions[indices]).to(device),
            rewards=torch.from_numpy(self._rewards[indices]).to(device),
            next_observations=torch.from_numpy(self._next_observations[indices]).to(device),
            terminated=torch.from_numpy(self._terminated[indices]).to(device),
        )

