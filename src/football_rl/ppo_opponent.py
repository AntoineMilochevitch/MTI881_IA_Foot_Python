"""Snapshots figées et synchronisation de leur installation avant PPO."""

import logging
import torch
from iafoot import UnityConnection

from football_rl.ppo_checkpoint import load_checkpoint, checkpoint_hidden_size
from football_rl.ppo_export import export_actor
from football_rl.ppo_network import PPOActorNetwork


class FrozenOpponent:
    def __init__(self, checkpoint_path: str, behavior: str = "FootSnapshot") -> None:
        checkpoint = load_checkpoint(checkpoint_path)
        schema = checkpoint["schema"]

        with torch.random.fork_rng(devices=[]):
            actor = PPOActorNetwork(observation_size=int(schema[0]),
                                    hidden_size=checkpoint_hidden_size(checkpoint, "actor"))

        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()
        actor.requires_grad_(False)

        self.checkpoint_path = checkpoint_path
        self.behavior = behavior
        self.schema = schema
        self.actor = actor
        self.version_counter = 0
        self.connection = None
        self.pending_version: int | None = None
        self.ready = False

    def export(self):
        return export_actor(self.actor)

    def validate_behavior(self, behavior: dict[str, object]) -> None:
        if any(agent.get("record_experience") is True for agent in behavior.get("agents", [])):
            raise ValueError(f"Désactiver Record Experience sur les adversaires {self.behavior!r}.")
        observation_size = int(behavior["obs_size"])
        observation_names = tuple(behavior["obs_names"])
        actions = tuple(
            (item["name"], item["type"])
            for item in behavior["actions"]
        )
        decision_period = int(behavior["decision_period"])

        if observation_size != len(observation_names):
            raise ValueError(
                "La taille des observations Unity ne correspond pas au nombre de noms."
            )
        if int(behavior["action_size"]) != len(actions):
            raise ValueError(
                "La taille des actions Unity ne correspond pas au nombre d'actions."
            )

        schema = (
            observation_size,
            observation_names,
            actions,
            decision_period,
        )
        if schema != self.schema:
            raise ValueError(
                f"Le schéma du comportement Unity est incompatible : "
                f"{schema!r} != {self.schema!r}."
            )

    def send(self, connection: UnityConnection) -> None:
        behavior = next(
            (
                item for item in connection.behaviors
                if item["name"] == self.behavior
            ),
            None,
        )
        if behavior is None:
            raise ValueError(
                f"Le comportement Unity {self.behavior!r} est absent."
            )

        self.validate_behavior(behavior)
        version = max(int(behavior["model_version"]), self.version_counter) + 1
        self.version_counter = version
        self.pending_version = version
        self.ready = False
        self.connection = connection

        model, decoder = self.export()
        connection.send_model(
            model,
            decoder=decoder,
            behavior=self.behavior,
            version=version,
        )

    def on_model_ack(self, connection: UnityConnection, header: dict[str, object]) -> bool:
        if (
            connection is not self.connection
            or self.pending_version is None
            or header.get("behavior") != self.behavior
            or header.get("version") != self.pending_version
        ):
            return False

        self.pending_version = None
        if header.get("ok") is True:
            self.ready = True
            return True

        self.ready = False
        raise ValueError(header.get("error") or "Unity a refusé la snapshot")

    def on_disconnect(self, connection: UnityConnection) -> None:
        if connection is self.connection:
            self.connection = None
            self.pending_version = None
            self.ready = False


class FrozenOpponentPool:
    """Plusieurs politiques adverses, mais un seul trainer apprenant.

    Chaque comportement est installé une fois par connexion. L'apprenant
    n'est envoyé qu'après TOUS les accusés, même s'ils arrivent dans le désordre.
    Les mises à jour PPO suivantes ne remplacent jamais ces snapshots.
    """

    def __init__(self, opponents: list[FrozenOpponent], trainer) -> None:
        names = [opponent.behavior for opponent in opponents]
        if len(set(names)) != len(names) or any(
            not name.strip() or name == "*" or name == trainer.behavior for name in names
        ):
            raise ValueError("Les comportements adverses doivent être distincts entre eux et de l'apprenant.")
        self.opponents = opponents
        self.trainer = trainer
        self.connection = None

    @property
    def ready(self) -> bool:
        return all(opponent.ready for opponent in self.opponents)

    def on_connect(self, connection: UnityConnection) -> None:
        if self.connection is not None and self.connection is not connection:
            raise RuntimeError("Une autre connexion utilise déjà les adversaires.")
        announced = {item['name']: item for item in connection.behaviors}
        missing = {self.trainer.behavior, *(o.behavior for o in self.opponents)} - announced.keys()
        if missing:
            logging.getLogger(__name__).info("Comportements en attente : %s.", ", ".join(sorted(missing)))
            return
        # Vérifier tous les schémas avant le premier envoi.
        for opponent in self.opponents:
            opponent.validate_behavior(announced[opponent.behavior])
        self.connection = connection
        for opponent in self.opponents:
            if not opponent.ready and opponent.pending_version is None:
                logging.getLogger(__name__).info("Envoi de la snapshot %r.", opponent.behavior)
                opponent.send(connection)
        if self.ready:
            self.trainer.on_connect(connection)

    def on_model_ack(self, connection: UnityConnection, header: dict[str, object]) -> None:
        if connection is not self.connection:
            return
        opponent = next((o for o in self.opponents if o.behavior == header.get('behavior')), None)
        if opponent is not None:
            if opponent.on_model_ack(connection, header) and self.ready:
                logging.getLogger(__name__).info("Toutes les snapshots sont acceptées ; envoi de l'apprenant.")
                self.trainer.on_connect(connection)
        elif self.ready:
            self.trainer.on_model_ack(connection, header)

    def on_batch(self, connection: UnityConnection, batch) -> None:
        if connection is self.connection and self.ready and batch.behavior == self.trainer.behavior:
            self.trainer.on_batch(connection, batch)

    def on_disconnect(self, connection: UnityConnection) -> None:
        if connection is not self.connection:
            return
        self.trainer.on_disconnect(connection)
        for opponent in self.opponents:
            opponent.on_disconnect(connection)
        self.connection = None
