"""Composantes de récompense prévues, sans coefficients imposés."""

from dataclasses import dataclass

from football_rl.types import Info


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Valeurs à déterminer expérimentalement par une étude de sensibilité.

    Les règles d'attribution des récompenses d'objectif restent à définir.
    Les passes et buts précédés de passes concernent la phase en équipe.
    """

    objective: float
    ball_approach: float
    ball_contact: float
    on_target_shot: float
    pass_completed: float
    goal_after_pass: float
    defensive_interception: float
    time_penalty: float
    out_of_bounds_penalty: float


class RewardCalculator:
    """Contrat de composition ; lieu de calcul et événements à convenir avec Unity.

    Une interception défensive exige un contact avec un ballon qui se dirige
    vers le but défendu et aurait pénétré dans la zone de but sans ce contact.
    La pénalité temporelle s'applique à chaque frame de simulation.
    Les lots destinés à l'agent contiennent les récompenses des transitions ;
    leur calcul à la collecte ou à la réception reste à déterminer.
    """

    def __init__(self, weights: RewardWeights) -> None:
        self.weights = weights

    def compute(self, events: Info) -> float:
        """Compose objectifs, apprentissage, défense et pénalités."""
        raise NotImplementedError("Définir les événements Unity et les règles d'attribution.")
