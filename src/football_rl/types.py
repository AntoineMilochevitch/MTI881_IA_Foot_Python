"""Contrats de données Python ; leur sérialisation vers Unity reste à définir."""

from dataclasses import dataclass
from typing import Any, NotRequired, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


Observation: TypeAlias = NDArray[np.float32]
Info: TypeAlias = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    """Capacités fixes communes aux phases 1v1 et en équipe.

    Ordre : ballon (6), agent (3), coéquipiers puis adversaires (7 par
    emplacement : position 3D, vitesse 3D, présence), buts (10), match (3).
    Chaque but : centre relatif (3), largeur et hauteur de son ouverture (2),
    dans l’ordre but adverse puis but défendu. Les buts sont verticaux,
    avec une orientation fixe connue dans les axes du terrain. Le repère
    est centré sur l’agent, avec des axes parallèles à ceux du terrain.
    Largeur et hauteur sont divisées par leurs maxima fixes pour être dans [0, 1].
    Les joueurs absents ont des valeurs et un masque de présence nuls.
    Les capacités sont choisies avant l'entraînement et restent constantes.
    """

    max_teammates: int
    max_opponents: int

    def __post_init__(self) -> None:
        if self.max_teammates < 0 or self.max_opponents < 0:
            raise ValueError("Prévoir au moins zéro coéquipier et zéro adversaire.")

    @property
    def size(self) -> int:
        return 22 + 7 * (self.max_teammates + self.max_opponents)


class FootballAction(TypedDict):
    """Déplacement, frappe à intensité variable, saut et tacle optionnel.

    shoot : tableau (1,) dans [0, 1], zéro sans déclenchement.
    Cette commande unique sert aux tirs et aux passes : l'agent apprend
    à ajuster son intensité selon la situation et son placement.
    move_target : cible relative normalisée dans [-1, 1], tableau (3,).
    L'agent apprend à se placer par ses déplacements avant de tirer ou passer.
    La direction de frappe découle du placement dans Unity ; la règle exacte
    reste à définir côté moteur. Aucune direction de frappe n'est fournie
    dans l'action. La passe concerne le jeu en équipe.
    Le tacle n'est inclus que si cette mécanique optionnelle est activée.
    """

    move_target: NDArray[np.float32]
    shoot: NDArray[np.float32]
    jump: int
    tackle: NotRequired[int]


class StepResult(TypedDict):
    """Résultat d'une interaction via l'interface Gymnasium."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Info


@dataclass(frozen=True, slots=True)
class Transition:
    """Expérience enregistrée dans Unity pour un agent."""

    observation: Observation
    action: FootballAction
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class Episode:
    """Trajectoire d'un agent et statistiques de jeu associées."""

    transitions: tuple[Transition, ...]
    statistics: Info


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """Lot d'expériences envoyé par Unity au backend d'entraînement."""

    episodes: tuple[Episode, ...]


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Réseau destiné à Unity ; format d'export à convenir entre les moteurs."""

    payload: bytes
    format: str
