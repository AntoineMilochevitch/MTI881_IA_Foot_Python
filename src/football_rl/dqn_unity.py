"""Entraînement et évaluation du DQN dans le jeu Unity, via le pont ``iafoot``.

Boucle d'entraînement :
1. ``on_connect`` : Unity annonce ses observations et actions ; on crée le DQN
   (ou on reprend un checkpoint) et on envoie le réseau avec un décodeur
   ``discrete`` en mode ε-greedy : c'est Unity qui explore.
2. ``on_batch`` : on garde les transitions jouées avec une table d'actions DQN
   (``action_index >= 0``), on les ajoute au tampon de rejeu, on fait un pas de
   gradient toutes les ``train_freq`` transitions, puis on renvoie le réseau avec
   le nouvel ε.
3. Sauvegarde régulière dans ``checkpoints/<run>`` et suivi TensorBoard dans ``runs/<run>``.

Évaluation (``--evaluate --resume chemin.pt``) : le réseau est envoyé en mode
``argmax`` (sans exploration), rien n'est entraîné, et on compte les issues des
épisodes joués entièrement avec ce réseau jusqu'à en avoir ``--evaluation-episodes``.

Lancement depuis la racine du dépôt Python (``iafoot`` lié avec
``scripts/link_unity_bridge.py``) :
    python -m football_rl.dqn_unity --time-scale 10
    python -m football_rl.dqn_unity --evaluate --resume checkpoints/<run>/latest.pt
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from iafoot import Batch, TrainingServer, UnityConnection, models

from football_rl.dqn import DQNConfig, DQNLearner
from football_rl.dqn_actions import ACTION_COLUMNS, ACTION_TABLES, ACTION_TYPES, build_action_table, describe_action

CHECKPOINT_FORMAT = "football_rl.dqn_unity/1"
SPORTING_OUTCOMES = ("goal_scored", "goal_conceded", "timeout")
KNOWN_OUTCOMES = SPORTING_OUTCOMES + ("interrupted", "match_over")
OUTCOME_LABELS = {"goal_scored": "but marqué", "goal_conceded": "but encaissé", "timeout": "temps écoulé",
                  "interrupted": "interrompu", "match_over": "fin de match"}


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:  # console Windows en cp1252 (ε, τ...) si main() n'a pas reconfiguré la sortie
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(encoding, errors="replace").decode(encoding), flush=True)


@dataclass(slots=True)
class RunSettings:
    """Réglages de la session (hors hyperparamètres du DQN)."""

    behavior: str = "Foot"
    action_table: str = "extended"
    unity_batch_size: int = 512
    time_scale: float | None = None
    publish_every: int = 1
    checkpoint_dir: Path | None = None
    checkpoint_every: int = 50_000
    log_dir: Path | None = None
    max_transitions: int | None = None
    evaluate: bool = False
    evaluation_episodes: int = 100
    evaluation_output: Path | None = None
    device: str = "cpu"


# --------------------------------------------------------------------------- Checkpoints


def save_checkpoint(path: Path, trainer: DQNUnityTrainer) -> None:
    """Écrit d'abord un fichier temporaire, pour ne jamais laisser un checkpoint à moitié écrit."""
    assert trainer.learner is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "dqn_config": asdict(trainer.config),
        "learner": trainer.learner.state_dict(),
        "action_table_name": trainer.table_name,
        "action_table": trainer.table.tolist(),
        "obs_size": trainer.learner.observation_size,
        "obs_names": list(trainer.obs_names),
        "behavior": trainer.settings.behavior,
        "model_version": trainer.version,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} n'est pas un checkpoint DQN Unity ({CHECKPOINT_FORMAT}).")
    return checkpoint


# --------------------------------------------------------------------------- Entraîneur


class DQNUnityTrainer:
    """Rappels ``TrainingServer`` : on_connect, on_batch, on_model_ack, on_disconnect."""

    def __init__(
        self,
        config: DQNConfig,
        settings: RunSettings,
        checkpoint: dict[str, Any] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.checkpoint = checkpoint
        self.on_finished = on_finished
        if checkpoint is not None:
            # Une reprise garde les hyperparamètres et la table du checkpoint.
            self.config = DQNConfig(**checkpoint["dqn_config"])
            self.table_name = str(checkpoint["action_table_name"])
            self.table = np.asarray(checkpoint["action_table"], dtype=np.float32)
            self.version = int(checkpoint.get("model_version", 0))
        else:
            self.config = config
            self.table_name = settings.action_table
            self.table = build_action_table(settings.action_table)
            self.version = 0

        self.device = torch.device(settings.device)
        self.learner: DQNLearner | None = None
        self.obs_names: list[str] = []
        self.agent_teams: dict[int, str] = {}
        self.decision_seconds: float | None = None
        self.published_versions: set[int] = set()
        self.accepted_once = False
        self.finished = False
        self.warned_next_obs = False

        self.batches = 0
        self.ignored_transitions = 0
        self.last_checkpoint_steps = 0
        self.saved_at_steps = -1
        self.started = time.perf_counter()

        self.writer = None
        if settings.log_dir is not None and not settings.evaluate:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(settings.log_dir))
            text = json.dumps({"dqn": asdict(self.config), "run": {k: str(v) for k, v in asdict(settings).items()},
                               "action_table": self.table_name}, indent=2)
            self.writer.add_text("config", "\n".join("    " + line for line in text.splitlines()), 0)

        # Évaluation : un épisode compte seulement s'il a été joué entièrement par nos réseaux argmax.
        self.episode_clean: dict[tuple[int, int], bool] = {}
        self.evaluation_episodes: list[dict[str, Any]] = []
        self.evaluation_excluded: Counter[str] = Counter()

    # ------------------------------------------------------------------ Connexion

    def on_connect(self, connection: UnityConnection) -> None:
        names = [b["name"] for b in connection.behaviors]
        behavior = next((b for b in connection.behaviors if b["name"] == self.settings.behavior), None)
        if behavior is None:
            raise ValueError(f"Comportement {self.settings.behavior!r} absent de la scène Unity (présents : {names}).")

        actions = tuple((a["name"], a["type"]) for a in behavior["actions"])
        expected = tuple(zip(ACTION_COLUMNS, ACTION_TYPES))
        if actions != expected:
            raise ValueError(f"Actions Unity inattendues : {actions}. La table DQN suppose {expected}.")

        obs_size = int(behavior["obs_size"])
        obs_names = [str(n) for n in behavior["obs_names"]]
        if self.learner is None:
            self._create_learner(obs_size, obs_names)
        elif obs_size != self.learner.observation_size or obs_names != self.obs_names:
            raise ValueError("Les observations Unity ont changé depuis la première connexion : relancer Python.")

        self.agent_teams = {int(a["id"]): str(a["team"]) for a in behavior.get("agents", [])}
        hello = connection.hello or {}
        if hello.get("fixed_delta_time"):
            self.decision_seconds = float(hello["fixed_delta_time"]) * int(behavior["decision_period"])

        connection.set_config(batch_size=self.settings.unity_batch_size, time_scale=self.settings.time_scale)
        self.version = max(self.version, int(behavior.get("model_version", 0)))
        agents = ", ".join(f"#{a['id']} {a['name']} [{a['team']}]" for a in behavior.get("agents", []))
        log(f"Unity connecté {connection} : comportement '{behavior['name']}', {obs_size} observations, "
            f"agents : {agents or 'aucun'}.")
        self._publish(connection)

    def _create_learner(self, obs_size: int, obs_names: list[str]) -> None:
        if self.checkpoint is not None:
            if obs_size != int(self.checkpoint["obs_size"]) or obs_names != list(self.checkpoint["obs_names"]):
                raise ValueError(
                    f"Le checkpoint a été entraîné avec {self.checkpoint['obs_size']} observations, la scène en "
                    f"envoie {obs_size} (ou leurs noms diffèrent) : vérifier Max Teammates / Max Opponents des FootAgent."
                )
        self.learner = DQNLearner(obs_size, len(self.table), self.config, self.device)
        self.obs_names = obs_names
        if self.checkpoint is not None:
            self.learner.load_state_dict(self.checkpoint["learner"])
            log(f"Checkpoint repris : {self.learner.env_steps} transitions, {self.learner.gradient_steps} pas de "
                f"gradient, ε = {self.learner.epsilon:.3f}. Le tampon de rejeu repart vide.")
        self.last_checkpoint_steps = self.learner.env_steps
        mode = "ÉVALUATION (argmax, pas d'entraînement)" if self.settings.evaluate else "entraînement"
        log(f"DQN prêt ({mode}) : {obs_size} observations -> {len(self.table)} actions "
            f"(table '{self.table_name}'), {'Double DQN' if self.config.double_dqn else 'DQN'}, "
            f"lr {self.config.lr:g}, τ {self.config.tau:g}, train_freq {self.config.train_freq}.")

    def on_disconnect(self, connection: UnityConnection) -> None:
        log(f"Unity déconnecté {connection}. Relancer Play pour reprendre (le DQN reste en mémoire).")
        if not self.settings.evaluate and self.learner is not None:
            self._save("latest.pt")

    # ------------------------------------------------------------------ Python -> Unity

    def _publish(self, connection: UnityConnection) -> None:
        assert self.learner is not None
        layers = self.learner.online_net.layers
        if self.device.type != "cpu":  # from_torch calcule sa vérification sur CPU
            layers = copy.deepcopy(layers).cpu()
        model = models.from_torch(layers)
        if self.settings.evaluate:
            decoder = models.discrete_decoder(self.table, mode="argmax")
        else:
            decoder = models.discrete_decoder(self.table, mode="epsilon_greedy", epsilon=self.learner.epsilon)
        self.version += 1
        connection.send_model(model, version=self.version, behavior=self.settings.behavior, decoder=decoder)
        self.published_versions.add(self.version)

    def on_model_ack(self, connection: UnityConnection, header: dict[str, Any]) -> None:
        if not header.get("ok"):
            log(f"ERREUR : Unity a refusé le modèle v{header.get('version')} : {header.get('error')}")
        elif not self.accepted_once:
            self.accepted_once = True
            log(f"Unity a installé le DQN (v{header.get('version')}) : {header.get('description')}.")

    # ------------------------------------------------------------------ Unity -> Python

    def on_batch(self, connection: UnityConnection, batch: Batch) -> None:
        if self.finished or self.learner is None or batch.behavior != self.settings.behavior:
            return
        if batch.next_obs is None:
            if not self.warned_next_obs:
                self.warned_next_obs = True
                log("ERREUR : les batchs n'ont pas de next_obs. Cocher 'Include Next Observations' dans le "
                    "TrainingBridge d'Unity : le DQN en a besoin pour sa cible de Bellman.")
            return
        self.batches += 1
        if self.settings.evaluate:
            self._evaluation_batch(batch)  # le réseau ne change pas : rien à renvoyer à Unity
        else:
            self._training_batch(connection, batch)

    def _training_batch(self, connection: UnityConnection, batch: Batch) -> None:
        learner = self.learner
        assert learner is not None
        # action_index = -1 : action tirée par un autre décodeur (modèle aléatoire v0 d'avant notre envoi).
        valid = (batch.action_index >= 0) & (batch.action_index < len(self.table))
        n_valid = int(valid.sum())
        ignored = len(batch) - n_valid
        self.ignored_transitions += ignored
        if n_valid:
            learner.store_batch(batch.obs[valid], batch.action_index[valid].astype(np.int64), batch.rewards[valid],
                                batch.next_obs[valid], batch.terminated[valid])
            learner.env_steps += n_valid

        start = time.perf_counter()
        result = learner.train_after(n_valid)
        train_seconds = time.perf_counter() - start

        self._report_training(batch, valid, n_valid, ignored, result, train_seconds)

        if self.batches % self.settings.publish_every == 0:
            self._publish(connection)

        if learner.env_steps - self.last_checkpoint_steps >= self.settings.checkpoint_every:
            self.last_checkpoint_steps = learner.env_steps
            self._save(f"step_{learner.env_steps:09d}.pt")
            self._save("latest.pt")

        if self.settings.max_transitions is not None and learner.env_steps >= self.settings.max_transitions:
            self._save("latest.pt")
            log(f"Budget atteint ({learner.env_steps} transitions) : fin de l'entraînement.")
            self._finish()

    def _report_training(self, batch: Batch, valid: np.ndarray, n_valid: int, ignored: int,
                         result: dict[str, float], train_seconds: float) -> None:
        learner = self.learner
        assert learner is not None
        outcomes = Counter(str(e["outcome"]) for e in batch.episodes)
        returns = [float(e["return"]) for e in batch.episodes]

        parts = [f"batch n°{batch.batch_id}", f"+{n_valid} transitions" + (f" ({ignored} ignorées)" if ignored else ""),
                 f"total {learner.env_steps}", f"ε {learner.epsilon:.3f}"]
        if result["steps"]:
            parts.append(f"{int(result['steps'])} pas de gradient en {train_seconds:.1f} s, perte {result['loss']:.4f}, "
                         f"Q moyen {result['mean_q']:+.3f}")
        else:
            parts.append(f"remplissage du tampon {len(learner.replay_buffer)}/{self.config.learning_starts}")
        if batch.episodes:
            detail = ", ".join(f"{OUTCOME_LABELS.get(o, o)} {c}" for o, c in outcomes.most_common())
            parts.append(f"{len(batch.episodes)} épisode(s) : {detail}, retour moyen {np.mean(returns):+.3f}")
        log(" | ".join(parts))

        if self.writer is None:
            return
        step = learner.env_steps
        scalars = {
            "train/epsilon": learner.epsilon,
            "train/buffer_size": len(learner.replay_buffer),
            "train/gradient_steps": learner.gradient_steps,
            "train/ignored_transitions": ignored,
            "train/train_seconds_per_batch": train_seconds,
            "batch/reward_mean": float(batch.rewards[valid].mean()) if n_valid else 0.0,
        }
        if result["steps"]:
            scalars["train/loss"] = result["loss"]
            scalars["train/mean_q"] = result["mean_q"]
        if batch.episodes:
            scalars["episodes/count"] = len(batch.episodes)
            scalars["episodes/return_mean"] = float(np.mean(returns))
            scalars["episodes/length_mean"] = float(np.mean([float(e["length"]) for e in batch.episodes]))
            for outcome in KNOWN_OUTCOMES:
                scalars[f"episodes/outcome_{outcome}"] = outcomes.get(outcome, 0) / len(batch.episodes)
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, float(value), step)
        if n_valid:
            # Répartition des actions jouées : un pic unique signale une politique bloquée.
            self.writer.add_histogram("actions/index", batch.action_index[valid], step, bins=len(self.table))
        self.writer.flush()

    # ------------------------------------------------------------------ Évaluation

    def _evaluation_batch(self, batch: Batch) -> None:
        for agent, episode, version in zip(batch.agent_id.tolist(), batch.episode_id.tolist(),
                                           batch.model_version.tolist()):
            key = (agent, episode)
            self.episode_clean[key] = self.episode_clean.get(key, True) and version in self.published_versions

        for summary in batch.episodes:
            key = (int(summary["agent_id"]), int(summary["episode_id"]))
            clean = self.episode_clean.pop(key, None)
            outcome = str(summary["outcome"])
            if not clean:  # None : aucune transition ; False : commencé avec un autre modèle
                self.evaluation_excluded["autre modèle"] += 1
            elif outcome in SPORTING_OUTCOMES:
                self.evaluation_episodes.append(dict(summary))
            else:
                self.evaluation_excluded[OUTCOME_LABELS.get(outcome, outcome)] += 1

        counts = Counter(str(e["outcome"]) for e in self.evaluation_episodes)
        log(f"Évaluation : {len(self.evaluation_episodes)}/{self.settings.evaluation_episodes} épisodes "
            f"(but marqué {counts['goal_scored']}, encaissé {counts['goal_conceded']}, temps écoulé {counts['timeout']})"
            + (f" | exclus : {dict(self.evaluation_excluded)}" if self.evaluation_excluded else ""))
        if len(self.evaluation_episodes) >= self.settings.evaluation_episodes:
            self._report_evaluation()
            self._finish()

    def evaluation_report(self) -> dict[str, Any]:
        episodes = self.evaluation_episodes[: self.settings.evaluation_episodes]
        n = len(episodes)
        counts = Counter(str(e["outcome"]) for e in episodes)
        returns = np.array([float(e["return"]) for e in episodes])
        lengths = np.array([float(e["length"]) for e in episodes])
        per_team: dict[str, Counter[str]] = {}
        for e in episodes:
            per_team.setdefault(self.agent_teams.get(int(e["agent_id"]), "?"), Counter())[str(e["outcome"])] += 1
        return {
            "episodes": n,
            "goal_scored": counts["goal_scored"],
            "goal_conceded": counts["goal_conceded"],
            "timeout": counts["timeout"],
            "win_rate": counts["goal_scored"] / n,
            "loss_rate": counts["goal_conceded"] / n,
            "timeout_rate": counts["timeout"] / n,
            "goal_difference_per_episode": (counts["goal_scored"] - counts["goal_conceded"]) / n,
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "length_mean_decisions": float(lengths.mean()),
            "length_mean_seconds": float(lengths.mean() * self.decision_seconds) if self.decision_seconds else None,
            "per_team": {team: dict(c) for team, c in per_team.items()},
            "excluded": dict(self.evaluation_excluded),
            "action_table": self.table_name,
            "env_steps_trained": self.learner.env_steps if self.learner else None,
        }

    def _report_evaluation(self) -> None:
        r = self.evaluation_report()
        seconds = f" ({r['length_mean_seconds']:.1f} s de jeu)" if r["length_mean_seconds"] else ""
        print("\n=========== Résultat de l'évaluation DQN ===========", flush=True)
        print(f"  Épisodes évalués      : {r['episodes']} (un épisode = une manche, du coup d'envoi au but)")
        print(f"  But marqué            : {r['goal_scored']:4d}  ({r['win_rate']:.1%})")
        print(f"  But encaissé          : {r['goal_conceded']:4d}  ({r['loss_rate']:.1%})")
        print(f"  Temps écoulé          : {r['timeout']:4d}  ({r['timeout_rate']:.1%})")
        print(f"  Différence de buts    : {r['goal_difference_per_episode']:+.3f} par épisode")
        print(f"  Retour moyen          : {r['return_mean']:+.3f} ± {r['return_std']:.3f}")
        print(f"  Durée moyenne         : {r['length_mean_decisions']:.0f} décisions{seconds}")
        for team, counts in r["per_team"].items():
            print(f"  Équipe {team:<14} : {counts}")
        if r["excluded"]:
            print(f"  Exclus                : {r['excluded']}")
        if len(r["per_team"]) > 1:
            print("  Note : les deux équipes sont jouées par le DQN (self-play), donc victoires et défaites se\n"
                  "  compensent. Pour mesurer le niveau, évaluer contre SimpleBotInput (voir README_DQN.md).")
        print("====================================================\n", flush=True)
        if self.settings.evaluation_output is not None:
            self.settings.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
            self.settings.evaluation_output.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"Résultat écrit dans {self.settings.evaluation_output}")

    # ------------------------------------------------------------------ Fin et sauvegarde

    def _save(self, name: str) -> None:
        if self.settings.checkpoint_dir is None or self.learner is None:
            return
        if name == "latest.pt" and self.saved_at_steps == self.learner.env_steps:
            return
        path = self.settings.checkpoint_dir / name
        save_checkpoint(path, self)
        if name == "latest.pt":
            self.saved_at_steps = self.learner.env_steps
        log(f"Checkpoint écrit : {path}")

    def _finish(self) -> None:
        self.finished = True
        if self.on_finished is not None:
            self.on_finished()

    def close(self) -> None:
        if not self.settings.evaluate and self.learner is not None:
            self._save("latest.pt")
        if self.writer is not None:
            self.writer.close()


# --------------------------------------------------------------------------- Ligne de commande


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    connection = parser.add_argument_group("connexion Unity")
    connection.add_argument("--host", default="127.0.0.1")
    connection.add_argument("--port", type=int, default=5005)
    connection.add_argument("--behavior", default="Foot", help="Behavior Name des FootAgent à piloter")
    connection.add_argument("--batch-size", type=int, default=512, help="transitions par batch envoyé par Unity")
    connection.add_argument("--time-scale", type=float, default=None, help="vitesse de simulation (ex. 10)")
    connection.add_argument("--verbose", action="store_true", help="affiche chaque message réseau")

    dqn = parser.add_argument_group("DQN (ignorés avec --resume : les réglages du checkpoint sont repris)")
    dqn.add_argument("--action-table", choices=sorted(ACTION_TABLES), default="extended")
    dqn.add_argument("--seed", type=int, default=0)
    dqn.add_argument("--device", default="cpu", help="cpu ou cuda")
    dqn.add_argument("--lr", type=float, default=2.5e-4)
    dqn.add_argument("--gamma", type=float, default=0.99)
    dqn.add_argument("--hidden", type=int, default=256)
    dqn.add_argument("--mini-batch-size", type=int, default=128, help="transitions par pas de gradient")
    dqn.add_argument("--buffer", type=int, default=200_000)
    dqn.add_argument("--learning-starts", type=int, default=5_000)
    dqn.add_argument("--train-freq", type=int, default=4, help="nouvelles transitions par pas de gradient")
    dqn.add_argument("--tau", type=float, default=0.005, help="1 = copie dure toutes les --target-update")
    dqn.add_argument("--target-update", type=int, default=1_000)
    dqn.add_argument("--eps-start", type=float, default=1.0)
    dqn.add_argument("--eps-end", type=float, default=0.05)
    dqn.add_argument("--eps-decay", type=int, default=200_000, help="transitions pour passer de eps-start à eps-end")
    dqn.add_argument("--no-double", action="store_true", help="DQN classique au lieu du Double DQN")

    run = parser.add_argument_group("session")
    run.add_argument("--run-name", default=None, help="défaut : dqn_AAAAMMJJ-HHMMSS")
    run.add_argument("--max-transitions", type=int, default=None, help="arrête l'entraînement après N transitions")
    run.add_argument("--checkpoint-dir", type=Path, default=None, help="défaut : checkpoints/<run>")
    run.add_argument("--checkpoint-every", type=int, default=50_000, help="en transitions")
    run.add_argument("--publish-every", type=int, default=1, help="batchs entre deux envois du réseau à Unity")
    run.add_argument("--log-dir", type=Path, default=Path("runs"))
    run.add_argument("--no-tensorboard", action="store_true")
    run.add_argument("--resume", type=Path, default=None, help="checkpoint à reprendre ou à évaluer")
    run.add_argument("--evaluate", action="store_true", help="évalue --resume sans exploration ni entraînement")
    run.add_argument("--evaluation-episodes", type=int, default=100)
    run.add_argument("--evaluation-output", type=Path, default=None, help="fichier JSON du résultat")
    return parser


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.evaluate and args.resume is None:
        parser.error("--evaluate exige --resume <checkpoint>.")
    if args.evaluation_episodes < 1 or args.publish_every < 1 or args.batch_size < 1:
        parser.error("--evaluation-episodes, --publish-every et --batch-size doivent être >= 1.")

    checkpoint = None
    if args.resume is not None:
        try:
            checkpoint = load_checkpoint(args.resume)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            parser.error(f"Impossible de charger le checkpoint : {error}")

    prefix = "dqn_eval" if args.evaluate else "dqn"
    run_name = args.run_name or datetime.now().strftime(f"{prefix}_%Y%m%d-%H%M%S")
    try:
        config = DQNConfig(
            lr=args.lr, gamma=args.gamma, epsilon_start=args.eps_start, epsilon_end=args.eps_end,
            epsilon_decay_steps=args.eps_decay, buffer_capacity=args.buffer, batch_size=args.mini_batch_size,
            target_update_freq=args.target_update, hidden_size=args.hidden, double_dqn=not args.no_double,
            learning_starts=args.learning_starts, tau=args.tau, train_freq=args.train_freq, seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    settings = RunSettings(
        behavior=args.behavior,
        action_table=args.action_table,
        unity_batch_size=args.batch_size,
        time_scale=args.time_scale,
        publish_every=args.publish_every,
        checkpoint_dir=None if args.evaluate else (args.checkpoint_dir or Path("checkpoints") / run_name),
        checkpoint_every=args.checkpoint_every,
        log_dir=None if args.no_tensorboard else args.log_dir / run_name,
        max_transitions=args.max_transitions,
        evaluate=args.evaluate,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_output=args.evaluation_output,
        device=args.device,
    )

    server = TrainingServer(args.host, args.port, debug=args.verbose)
    trainer = DQNUnityTrainer(config, settings, checkpoint, on_finished=server.stop)
    server.on_connect = trainer.on_connect
    server.on_batch = trainer.on_batch
    server.on_model_ack = trainer.on_model_ack
    server.on_disconnect = trainer.on_disconnect

    if checkpoint is not None:
        log(f"Checkpoint {args.resume} : table '{trainer.table_name}', réglages DQN du checkpoint utilisés.")
    if not args.evaluate:
        log(f"Run '{run_name}' : checkpoints dans {settings.checkpoint_dir}"
            + (f", TensorBoard dans {settings.log_dir}" if settings.log_dir else "") + ".")
    actions = "; ".join(f"{i}: {describe_action(row)}" for i, row in enumerate(trainer.table))
    log(f"Table d'actions '{trainer.table_name}' ({len(trainer.table)}) : {actions}")
    try:
        server.serve_forever()
    finally:
        server.stop()
        trainer.close()


if __name__ == "__main__":
    main()
