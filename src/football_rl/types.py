"""Types partagés pour le contrat entre l'environnement, l'agent et Unity."""

from typing import Any, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


Observation: TypeAlias = NDArray[np.float32]
Info: TypeAlias = dict[str, Any]


class FootballAction(TypedDict):
    """Action hybride normalisée envoyée au moteur Unity."""

    move_target: NDArray[np.float32]
    shoot: int
    pass_action: int
    jump: int
    tackle: int


class StepResult(TypedDict):
    """Données de transition à sérialiser par le futur pont réseau."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Info
