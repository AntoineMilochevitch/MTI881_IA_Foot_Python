"""Interfaces de composition des récompenses du match."""

from dataclasses import dataclass

from football_rl.types import Info


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Coefficients configurables des composantes de récompense."""

    scored_goal: float
    conceded_goal: float
    ball_progress: float
    first_contact: float
    possession_progress: float
    on_target_shot: float
    dangerous_shot_interception: float
    time_penalty: float
    out_of_bounds_penalty: float


class RewardCalculator:
    """Contrat de calcul de récompense à partir des événements Unity."""

    def __init__(self, weights: RewardWeights) -> None:
        """Conserve la configuration de récompense de l'expérience."""
        pass

    def compute(self, events: Info) -> float:
        """Compose la récompense sparse, dense, défensive et les pénalités."""
        pass
