"""Discrétisation de l'espace d'action continu pour les algorithmes DQN.

L'espace d'action du football 3D est hybride (continu + discret).
DQN nécessitant un espace discret fini, cette classe transforme un indice
entier en ``FootballAction`` et inversement, en découpant chaque axe continu
en un nombre configurable de niveaux régulièrement espacés.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from football_rl.types import FootballAction


@dataclass(frozen=True, slots=True)
class DiscretizationConfig:
    """Paramètres de discrétisation de l'espace d'action.

    Attributes:
        move_bins_per_axis: Nombre de niveaux par axe de déplacement.
            3 → {-1, 0, 1}, 5 → {-1, -0.5, 0, 0.5, 1}, etc.
        shoot_bins: Nombre de niveaux d'intensité de frappe.
            2 → {0, 1}, 3 → {0, 0.5, 1}, etc.
        enable_tackle: Inclure l'action de tacle.
    """

    move_bins_per_axis: int = 3
    shoot_bins: int = 2
    enable_tackle: bool = False

    def __post_init__(self) -> None:
        if self.move_bins_per_axis < 2:
            raise ValueError("Il faut au moins 2 niveaux par axe de déplacement.")
        if self.shoot_bins < 2:
            raise ValueError("Il faut au moins 2 niveaux d'intensité de frappe.")


class ActionDiscretizer:
    """Correspondance bijective entre indices entiers et ``FootballAction``.

    L'ensemble des actions est le produit cartésien de toutes les
    discrétisations. Chaque combinaison reçoit un indice unique.
    """

    def __init__(self, config: DiscretizationConfig) -> None:
        self.config = config

        # Niveaux régulièrement espacés sur chaque axe continu.
        self._move_levels: NDArray[np.float32] = np.linspace(
            -1.0, 1.0, config.move_bins_per_axis, dtype=np.float32,
        )
        self._shoot_levels: NDArray[np.float32] = np.linspace(
            0.0, 1.0, config.shoot_bins, dtype=np.float32,
        )

        # Construire la table de toutes les combinaisons possibles.
        move_indices = range(config.move_bins_per_axis)
        shoot_indices = range(config.shoot_bins)
        jump_indices = range(2)

        if config.enable_tackle:
            tackle_indices = range(2)
            self._combinations = list(
                product(move_indices, move_indices, move_indices,
                        shoot_indices, jump_indices, tackle_indices)
            )
        else:
            self._combinations = list(
                product(move_indices, move_indices, move_indices,
                        shoot_indices, jump_indices)
            )

        # Index inverse pour retrouver l'indice à partir d'un tuple.
        self._combo_to_index: dict[tuple[int, ...], int] = {
            combo: i for i, combo in enumerate(self._combinations)
        }

    @property
    def num_actions(self) -> int:
        """Nombre total d'actions discrètes."""
        return len(self._combinations)

    def index_to_action(self, index: int) -> FootballAction:
        """Convertit un indice entier en ``FootballAction``."""
        combo = self._combinations[index]
        mx, my, mz = combo[0], combo[1], combo[2]
        si = combo[3]
        ji = combo[4]

        action: FootballAction = {
            "move_target": np.array(
                [self._move_levels[mx], self._move_levels[my], self._move_levels[mz]],
                dtype=np.float32,
            ),
            "shoot": np.array([self._shoot_levels[si]], dtype=np.float32),
            "jump": int(ji),
        }

        if self.config.enable_tackle:
            action["tackle"] = int(combo[5])

        return action

    def action_to_index(self, action: FootballAction) -> int:
        """Retrouve l'indice de l'action discrète la plus proche.

        Pour chaque axe continu, sélectionne le niveau le plus proche
        de la valeur continue. Utile quand des transitions proviennent
        d'une source externe qui n'utilise pas exactement les niveaux.
        """
        move = action["move_target"]
        mx = int(np.argmin(np.abs(self._move_levels - move[0])))
        my = int(np.argmin(np.abs(self._move_levels - move[1])))
        mz = int(np.argmin(np.abs(self._move_levels - move[2])))
        si = int(np.argmin(np.abs(self._shoot_levels - action["shoot"][0])))
        ji = int(action["jump"])

        if self.config.enable_tackle:
            ti = int(action.get("tackle", 0))
            combo = (mx, my, mz, si, ji, ti)
        else:
            combo = (mx, my, mz, si, ji)

        return self._combo_to_index[combo]
