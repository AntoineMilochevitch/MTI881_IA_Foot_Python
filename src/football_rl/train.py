"""Contrat d'orchestration de l'entraînement à partir des épisodes Unity."""

from dataclasses import dataclass
from pathlib import Path

from football_rl.agent import FootballAgent
from football_rl.communication import UnityTrainingBridge
from football_rl.types import ObservationLayout


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration déclarative ; les critères de convergence restent à définir."""

    seed: int
    checkpoint_directory: Path
    tensorboard_directory: Path
    observation_layout: ObservationLayout


def train(
    config: TrainingConfig,
    agent: FootballAgent,
    bridge: UnityTrainingBridge,
) -> None:
    """Future boucle : envoyer le modèle initial, recevoir un lot, entraîner,
    renvoyer le réseau, puis répéter jusqu'à convergence ou interruption.

    Les récompenses et statistiques de jeu seront suivies dans TensorBoard.
    Exploration/exploitation et collecte simultanée seront coordonnées avec Unity.
    """
    raise NotImplementedError(
        "Implémenter le transport, les algorithmes, le suivi TensorBoard et l'arrêt."
    )


def main() -> None:
    """Futur point d'entrée de la configuration des expériences."""
    raise NotImplementedError("L'entraînement par lots n'est pas encore implémenté.")
