"""Serveur de collecte Unity commun aux algorithmes.

Le transport et le format des modèles sont fournis par iafoot. Une fabrique
optionnelle fournit la politique à envoyer ; par défaut, un modèle linéaire
non entraîné sert à observer le fonctionnement du pont.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from importlib import import_module
import sys
from typing import Any

import numpy as np

from iafoot import Batch, DecoderExport, ModelExport, TrainingServer, UnityConnection, models


ACTION_SCHEMA = (
    ("move_x", "continuous"), ("move_z", "continuous"),
    ("shoot", "binary"), ("shoot_power", "continuous"),
    ("shoot_curve", "continuous"), ("tackle", "binary"), ("jump", "binary"),
)
PolicyFactory = Callable[[dict[str, Any], int], tuple[ModelExport, DecoderExport]]


def build_demo_policy(behavior: dict[str, Any], seed: int) -> tuple[ModelExport, DecoderExport]:
    """Modèle linéaire de démonstration, sans apprentissage."""
    obs_size = int(behavior["obs_size"])
    actions = behavior["actions"]
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, (len(actions), obs_size)).astype(np.float32)
    bias = np.asarray([-2.0 if item["type"] == "binary" else 0.0 for item in actions], dtype=np.float32)
    model = models.linear(weights, bias).with_check(np.zeros(obs_size, dtype=np.float32))
    return model, models.direct_decoder(log_std=-0.7, deterministic=False)


def load_policy_factory(reference: str) -> PolicyFactory:
    """Charge une fonction indiquée explicitement par module:fonction."""
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--policy-factory attend module:fonction.")
    factory = getattr(import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"{reference!r} doit désigner une fonction appelable.")
    return factory


class UnityCollector:
    """Publie une politique et affiche les transitions sans les sauvegarder.

    Les callbacks de iafoot sont sérialisés. Les exports sont conservés entre
    les reconnexions ; une modification de schéma est signalée avant envoi.
    Les boucles d'apprentissage pourront utiliser directement les callbacks
    de TrainingServer avec leur propre état d'entraînement.
    """

    def __init__(self, args: argparse.Namespace, policy_factory: PolicyFactory) -> None:
        self.args = args
        self.policy_factory = policy_factory
        self.policies: dict[str, tuple[ModelExport, DecoderExport]] = {}
        self.schemas: dict[str, tuple[Any, ...]] = {}
        self.version = 0
        self.sent: dict[tuple[UnityConnection, str], tuple[str, int]] = {}
        self.received = 0

    def on_connect(self, connection: UnityConnection) -> None:
        hello = connection.hello or {}
        session = str(hello.get("session", ""))
        print(f"Unity connecté : {connection}", flush=True)
        connection.set_config(batch_size=self.args.batch_size, time_scale=self.args.time_scale)
        matched = False
        for behavior in connection.behaviors:
            name = behavior["name"]
            if name != self.args.behavior:
                continue
            matched = True
            actions = tuple((item["name"], item["type"]) for item in behavior["actions"])
            obs_size = int(behavior["obs_size"])
            obs_names = tuple(behavior["obs_names"])
            if actions != ACTION_SCHEMA or int(behavior["action_size"]) != 7:
                raise ValueError(f"Schéma d'actions de football incompatible pour {name!r} : {actions}")
            if obs_size <= 0 or len(obs_names) != obs_size:
                raise ValueError(f"Description des observations incohérente pour {name!r}.")
            schema = (obs_size, obs_names, actions, int(behavior["decision_period"]))
            if name in self.schemas and self.schemas[name] != schema:
                raise ValueError(f"Le schéma de {name!r} a changé. Redémarre avec la configuration voulue.")
            if name not in self.policies:
                model, decoder = self.policy_factory(behavior, self.args.seed)
                if not isinstance(model, ModelExport) or not isinstance(decoder, DecoderExport):
                    raise TypeError("La fabrique doit retourner (ModelExport, DecoderExport).")
                self.policies[name] = model, decoder
                self.schemas[name] = schema

            print(f"  {name} : {obs_size} observations, 7 actions, {len(behavior['agents'])} agent(s)")
            if self.args.verbose:
                print("  Observations : " + ", ".join(obs_names))

            # Un export par connexion/session/comportement, même si plusieurs
            # hello arrivent avant l'accusé de réception du modèle.
            key = (connection, name)
            current = int(behavior["model_version"])
            previous = self.sent.get(key)
            if previous is not None and previous[0] == session:
                continue
            model, decoder = self.policies[name]
            self.version = max(self.version, current) + 1
            connection.send_model(model, version=self.version, behavior=name, decoder=decoder)
            self.sent[key] = (session, self.version)
            print(f"  Politique v{self.version} envoyée ; attente de model_ack.", flush=True)
        if not matched:
            print(f"  Comportement {self.args.behavior!r} absent du hello ; attente de son inscription.", flush=True)

    def on_model_ack(self, connection: UnityConnection, header: dict[str, Any]) -> None:
        if header.get("ok"):
            print(f"Unity a accepté le modèle v{header.get('version')} ({header.get('behavior')}).", flush=True)
        else:
            print(f"Unity a refusé le modèle v{header.get('version')} : {header.get('error')}", flush=True)

    def on_batch(self, connection: UnityConnection, batch: Batch) -> None:
        if batch.behavior != self.args.behavior:
            return
        self.received += len(batch)
        versions = dict(sorted(Counter(batch.model_version.tolist()).items()))
        print(
            f"Batch {batch.batch_id} / {batch.behavior} : {len(batch)} transitions, "
            f"obs={batch.obs.shape}, actions={batch.actions.shape}, "
            f"versions={versions}, terminées={int(batch.terminated.sum())}, "
            f"tronquées={int(batch.truncated.sum())}, total={self.received}",
            flush=True,
        )
        if batch.next_obs is None:
            print("  Active includeNextObservations sur TrainingBridge pour disposer des états suivants.", flush=True)
        if self.args.verbose and len(batch):
            print(f"  Récompense moyenne : {batch.rewards.mean():.5f}")
            print(f"  Première action brute : {batch.actions[0].tolist()}", flush=True)
        # Collecte de diagnostic : les lots sont affichés puis libérés.
        # Chaque algorithme définira son stockage et son traitement des versions.

    def on_disconnect(self, connection: UnityConnection) -> None:
        self.sent = {key: value for key, value in self.sent.items() if key[0] is not connection}
        print(f"Unity déconnecté : {connection}. Les politiques restent en mémoire.", flush=True)


def main(*, policy_factory: PolicyFactory | None = None, description: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--behavior", default="Foot")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--policy-factory", help="fabrique module:fonction recevant (behavior, seed)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port doit être entre 1 et 65535.")
    if args.batch_size < 1:
        parser.error("--batch-size doit être supérieur à zéro.")
    if not 0.1 <= args.time_scale <= 50.0:
        parser.error("--time-scale doit être compris entre 0.1 et 50, comme dans Unity.")
    if args.seed < 0:
        parser.error("--seed doit être positif ou nul.")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    factory = policy_factory or build_demo_policy
    if args.policy_factory:
        factory = load_policy_factory(args.policy_factory)
    collector = UnityCollector(args, factory)
    server = TrainingServer(args.host, args.port, debug=args.verbose)
    server.on_connect = collector.on_connect
    server.on_batch = collector.on_batch
    server.on_model_ack = collector.on_model_ack
    server.on_disconnect = collector.on_disconnect
    print("Collecte de diagnostic : aucune optimisation ni sauvegarde des lots.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
