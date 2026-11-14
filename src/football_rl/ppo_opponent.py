import torch
from iafoot import UnityConnection

from football_rl.ppo_checkpoint import load_checkpoint
from football_rl.ppo_export import export_actor
from football_rl.ppo_network import PPOActorNetwork


class FrozenOpponent:
    def __init__(self, checkpoint_path: str, behavior: str = "FootSnapshot") -> None:
        checkpoint = load_checkpoint(checkpoint_path)
        schema = checkpoint["schema"]

        with torch.random.fork_rng(devices=[]):
            actor = PPOActorNetwork(observation_size=int(schema[0]))

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
