"""Interface des échanges par lots, indépendante du protocole réseau."""

from abc import ABC, abstractmethod

from football_rl.types import EpisodeBatch, ModelArtifact


class UnityTrainingBridge(ABC):
    """Unity exécute le modèle et collecte les épisodes ; Python l'entraîne.

    Transport, sérialisation et chargement du modèle côté Unity restent à
    choisir selon les contraintes de performance et de simplicité.
    """

    @abstractmethod
    def receive_batch(self) -> EpisodeBatch:
        """Reçoit un lot d'épisodes enregistrés dans Unity."""
        raise NotImplementedError

    @abstractmethod
    def send_model(self, model: ModelArtifact) -> None:
        """Transmet le réseau initial ou mis à jour pour son exécution dans Unity."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Libère les ressources de communication."""
        raise NotImplementedError
