"""Orchestration PPO : collecte d'une version, apprentissage, puis publication.

TrainingServer sérialise les callbacks : l'acteur reste inchangé pendant la
collecte et aucun batch n'est traité pendant une mise à jour. On ne bloque
jamais un callback en attendant model_ack, qui arrivera dans un autre message.
"""

from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
from time import perf_counter
from collections import Counter, deque

import torch
from iafoot import Batch, UnityConnection

from football_rl.ppo_buffer import PPORolloutBuffer
from football_rl.ppo_checkpoint import (
    build_checkpoint,
    restore_networks,
    restore_optimizers,
    save_checkpoint,
    validate_schema,
)
from football_rl.ppo_export import export_actor
from football_rl.ppo_network import PPOActorNetwork, PPOValueNetwork
from football_rl.ppo_training import compute_policy_metrics, train_ppo
from football_rl.unity import ACTION_SCHEMA
from football_rl.ppo_logging import PPOTensorBoardLogger

CONTINUOUS_STD_NAMES = ("move_x", "move_z", "shoot_power", "shoot_curve")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PPOTrainerConfig:
    """Sépare le transport Unity, la taille du rollout et les réglages PPO."""

    behavior: str = "Foot"
    seed: int = 0
    device: str | torch.device = "cpu"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    time_scale: float = 1.0
    rollout_size: int = 2048
    mini_batch_size: int = 128
    num_epochs: int = 4
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.01
    parity_atol: float = 1e-4
    collect_only: bool = False
    log_dir: str | None = None
    evaluate: bool = False
    evaluation_episodes: int = 200

    def __post_init__(self) -> None:
        if not self.behavior.strip():
            raise ValueError("behavior ne doit pas être vide.")
        for name, minimum in (
            ("seed", 0), ("batch_size", 1), ("rollout_size", 2),
            ("mini_batch_size", 1), ("num_epochs", 1),
            ("evaluation_episodes", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} doit être un entier >= {minimum}.")
        for name in (
            "actor_lr", "critic_lr", "time_scale", "gamma", "lambda_",
            "clip_epsilon", "max_grad_norm", "parity_atol", "target_kl",
        ):
            value = getattr(self, name)
            if name == "target_kl" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} doit être un nombre fini.")
            if name in ("gamma", "lambda_"):
                if not 0 <= value <= 1:
                    raise ValueError(f"{name} doit être compris entre 0 et 1.")
            elif value <= 0:
                raise ValueError(f"{name} doit être strictement positif.")
        if not 0.1 <= self.time_scale <= 50:
            raise ValueError("time_scale doit être compris entre 0.1 et 50, comme dans Unity.")
        if self.clip_epsilon >= 1:
            raise ValueError("clip_epsilon doit être inférieur à 1.")
        if self.evaluate and self.collect_only:
            raise ValueError("evaluate et collect_only ne peuvent pas être activés ensemble.")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device doit être cpu ou cuda[:index].")
        if device.type == "cuda" and (
            not torch.cuda.is_available()
            or (device.index is not None and device.index >= torch.cuda.device_count())
        ):
            raise ValueError("Le périphérique CUDA demandé n'est pas disponible.")


class PPOUnityTrainer:
    """Conserve les réseaux et Adam entre les connexions d'une instance Unity."""

    def __init__(
        self,
        config: PPOTrainerConfig,
        checkpoint_dir: str | Path | None = None,
        resume_checkpoint: dict[str, object] | None = None,
        diagnostic_no_jump_tackle: bool = False,
    ) -> None:
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.pending_checkpoint = resume_checkpoint
        self.diagnostic_no_jump_tackle = diagnostic_no_jump_tackle
        self.behavior = config.behavior
        self.seed = config.seed
        self.device = torch.device(config.device)
        self.actor_lr = config.actor_lr
        self.critic_lr = config.critic_lr
        self.batch_size = config.batch_size
        self.time_scale = config.time_scale

        self.actor = None
        self.critic = None
        self.actor_optimizer = None

        self.critic_optimizer = None
        self.schema = None
        self.connection = None
        self.session_id = None
        self.version_counter = 0
        self.pending_version = None
        self.active_version = None
        self.buffer = None
        self.phase = "disconnected"
        self.error = None
        self.update_count = 0
        self.trained_transitions = 0
        self.ignored_transitions = 0
        self.last_history: list[dict[str, float | int | bool]] = []
        self.last_parity: dict[str, float] = {}
        self.rollout_max_log_prob_error = 0.0

        self.collection_started_at = None
        # Borné : évite une croissance sans fin si aucun cycle ne vide la liste.
        self.episode_summaries = deque(maxlen=10_000)
        self.evaluation_episodes: list[dict[str, object]] = []
        self.evaluation_admissible_count = 0
        self.evaluation_finished = False
        self.evaluation_episode_states: dict[
            tuple[str, int, int], dict[str, object]
        ] = {}
        self.evaluation_technical_interruptions = 0
        self.evaluation_other_outcomes: Counter[str] = Counter()
        self.tb_logger = None
        if config.log_dir is not None and not config.collect_only:
            self.tb_logger = PPOTensorBoardLogger(config.log_dir, config)

    def _save_checkpoint(self) -> None:
        if self.config.collect_only or self.config.evaluate or self.checkpoint_dir is None:
            return
        progress = {
            "version_counter": self.version_counter,
            "update_count": self.update_count,
            "trained_transitions": self.trained_transitions,
            "ignored_transitions": self.ignored_transitions,
        }
        checkpoint = build_checkpoint(
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
            progress,
            self.schema,
            asdict(self.config),
        )
        save_checkpoint(checkpoint, str(self.checkpoint_dir / "latest.pt"))
        if self.update_count % 10 == 0:
            save_checkpoint(
                checkpoint,
                str(self.checkpoint_dir / f"cycle_{self.update_count:06d}.pt"),
            )

    def _set_error(self, error: Exception) -> None:
        self.error = str(error)
        self.phase = "error"
        self.active_version = None
        self.pending_version = None
        self.buffer = None
        logger.error("PPO arrêté pour cette connexion : %s", self.error)

    def _publish_actor(self, announced_version: int = 0) -> None:
        """Réserve une version avant l'envoi ; seul son ack autorise la collecte."""
        version = max(self.version_counter, announced_version) + 1
        if version > 2**31 - 1:
            raise ValueError("La version dépasse la capacité int32 du pont.")
        self.version_counter = version
        self.pending_version = version
        self.active_version = None
        self.buffer = None
        self.phase = "waiting_for_ack"
        if any(not torch.isfinite(p).all() for p in self.actor.parameters()):
            raise ValueError("L'acteur contient des paramètres non finis ; export refusé.")
        model, decoder = export_actor(self.actor)
        self.connection.send_model(
            model, version=version, behavior=self.behavior, decoder=decoder
        )
        logger.info("Politique v%d envoyée ; attente de model_ack.", version)

    @staticmethod
    def _disable_jump_and_tackle(actor: PPOActorNetwork) -> None:
        output_layer = actor.network[-1]
        if not isinstance(output_layer, torch.nn.Linear) or output_layer.out_features != 7:
            raise ValueError("La dernière couche de l'acteur n'est pas compatible avec le diagnostic.")
        with torch.no_grad():
            output_layer.weight[5:7].zero_()
            output_layer.bias[5:7].fill_(-20.0)

    def on_connect(self, connection: UnityConnection) -> None:
        # Une connexion étrangère ne doit pas mettre le trainer actif en erreur.
        if self.connection is not None and self.connection is not connection:
            raise RuntimeError("Une autre connexion Unity est déjà active.")

        previous_connection = self.connection
        previous_session = self.session_id
        self.connection = connection
        try:
            hello = connection.hello or {}
            session_id = hello.get("session")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("Le hello doit contenir un identifiant de session.")
            behavior = next(
                (item for item in connection.behaviors if item["name"] == self.behavior),
                None,
            )
            if behavior is None:
                if self.schema is not None and self.phase != "disconnected":
                    raise ValueError("Le comportement actif a disparu du hello.")
                self.connection = None
                logger.info("Comportement %r absent ; attente de son inscription.", self.behavior)
                return
            actions = tuple(
                (item["name"], item["type"])
                for item in behavior["actions"]
            )
            observation_size = int(behavior["obs_size"])
            observation_names = tuple(behavior["obs_names"])
            decision_period = int(behavior["decision_period"])
            if actions != ACTION_SCHEMA or int(behavior["action_size"]) != 7:
                raise ValueError("Le schéma d'actions Unity est incompatible.")
            if observation_size <= 0 or len(observation_names) != observation_size:
                raise ValueError("La description des observations est incohérente.")
            if decision_period <= 0:
                raise ValueError("decision_period doit être strictement positif.")

            schema = (
                observation_size,
                observation_names,
                actions,
                decision_period,
            )
            if self.schema is not None and self.schema != schema:
                raise ValueError("Le schéma Unity a changé.")

            if (
                previous_connection is connection
                and previous_session == session_id
                and self.schema == schema
                and self.phase != "disconnected"
            ):
                return

            # Une erreur nécessite une déconnexion/reconnexion explicite.
            if self.phase == "error":
                return
            if self.actor is None:
                pending_checkpoint = self.pending_checkpoint
                if pending_checkpoint is not None:
                    validate_schema(pending_checkpoint, schema)
                with torch.random.fork_rng(devices=[]):
                    # Initialisation CPU sans modifier le RNG CUDA de l'appelant.
                    torch.random.default_generator.manual_seed(self.seed)
                    actor = PPOActorNetwork(observation_size).to(self.device)
                    critic = PPOValueNetwork(observation_size).to(self.device)
                actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.actor_lr)
                critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
                if pending_checkpoint is not None:
                    restore_networks(pending_checkpoint, actor, critic)
                    restore_optimizers(
                        pending_checkpoint, actor_optimizer, critic_optimizer
                    )
                    if self.diagnostic_no_jump_tackle:
                        self._disable_jump_and_tackle(actor)
                    restored_rng_state = pending_checkpoint["torch_rng_state"]
                    if not isinstance(restored_rng_state, torch.Tensor):
                        raise ValueError("L'état RNG CPU du checkpoint est invalide.")
                    torch.set_rng_state(restored_rng_state)
                    self.version_counter = pending_checkpoint["progress"]["version_counter"]
                    self.update_count = pending_checkpoint["progress"]["update_count"]
                    self.trained_transitions = pending_checkpoint["progress"]["trained_transitions"]
                    self.ignored_transitions = pending_checkpoint["progress"]["ignored_transitions"]
                    self.pending_checkpoint = None
                self.actor, self.critic = actor, critic
                self.actor_optimizer, self.critic_optimizer = actor_optimizer, critic_optimizer

            self.schema = schema
            self.session_id = session_id
            self.error = None
            self.active_version = None
            self.buffer = None

            announced_version = int(behavior["model_version"])
            if announced_version < 0:
                raise ValueError("model_version doit être positif ou nul.")
            connection.set_config(
                batch_size=self.batch_size,
                time_scale=self.time_scale
            )
            logger.info("Unity %s : %d observations, 7 actions, appareil %s.",
                        session_id, observation_size, self.device)
            self._publish_actor(announced_version)
        except Exception as error:
            self._set_error(error)
            raise

    def _on_evaluation_batch(self, connection: UnityConnection, batch: Batch) -> None:
        if self.evaluation_finished or self.phase != "collecting":
            return

        try:
            session_id = (connection.hello or {}).get("session")
            if not isinstance(session_id, str) or self.active_version is None:
                return
            PPORolloutBuffer._validate_shapes(batch, self.schema[0])

            for index in range(len(batch.obs)):
                key = (
                    session_id,
                    int(batch.agent_id[index]),
                    int(batch.episode_id[index]),
                )
                state = self.evaluation_episode_states.setdefault(
                    key, {"steps": set(), "all_active": True}
                )
                steps = state["steps"]
                if not isinstance(steps, set):
                    raise ValueError("État interne des épisodes d'évaluation invalide.")
                step = int(batch.step[index])
                if step in steps:
                    state["all_active"] = False
                steps.add(step)
                if int(batch.model_version[index]) != self.active_version:
                    state["all_active"] = False

            active_mask = batch.model_version == self.active_version
            if active_mask.any():
                accepted = self.buffer.add_batch(batch, batch.batch_id, session_id)
                if accepted:
                    self._check_parity(self.buffer.chunks[-1])
                    self._log_action_diagnostics(self.buffer.chunks[-1])
                self.buffer.clear()

            for summary in getattr(batch, "episodes", None) or ():
                key = (
                    session_id,
                    int(summary["agent_id"]),
                    int(summary["episode_id"]),
                )
                state = self.evaluation_episode_states.pop(key, None)
                length = int(summary["length"])
                steps = state["steps"] if state is not None else set()
                admissible = (
                    state is not None
                    and state["all_active"] is True
                    and steps == set(range(length))
                )
                if not admissible:
                    self.evaluation_technical_interruptions += 1
                    continue
                self.evaluation_admissible_count += 1
                outcome = str(summary["outcome"])
                if outcome in {"goal_scored", "goal_conceded", "timeout"}:
                    self.evaluation_episodes.append(dict(summary))
                else:
                    self.evaluation_other_outcomes[outcome] += 1
                if len(self.evaluation_episodes) >= self.config.evaluation_episodes:
                    self.evaluation_finished = True
                    break

            sporting_count = len(self.evaluation_episodes)
            if self.tb_logger is not None and self.evaluation_episodes:
                step = sporting_count
                self.tb_logger.log_episodes(step, self.evaluation_episodes)
                self.tb_logger.log_evaluation_interruptions(
                    step, self.evaluation_technical_interruptions
                )
            logger.info(
                "Évaluation : %d/%d épisodes sportifs (%d épisodes complets reçus).",
                sporting_count,
                self.config.evaluation_episodes,
                self.evaluation_admissible_count,
            )
            if self.evaluation_finished:
                self.phase = "evaluation_finished"
                outcomes = Counter(
                    str(episode["outcome"]) for episode in self.evaluation_episodes
                )
                logger.info(
                    "Évaluation terminée : %d épisodes sportifs sur %d épisodes complets reçus, goal_scored=%d, "
                    "goal_conceded=%d, timeout=%d, interruptions techniques=%d.",
                    sporting_count,
                    self.evaluation_admissible_count,
                    outcomes.get("goal_scored", 0),
                    outcomes.get("goal_conceded", 0),
                    outcomes.get("timeout", 0),
                    self.evaluation_technical_interruptions,
                )
                if self.evaluation_other_outcomes:
                    logger.info(
                        "Autres issues exclues du bilan sportif : %s.",
                        dict(self.evaluation_other_outcomes),
                    )
        except Exception as error:
            self._set_error(error)
            raise

    def _log_action_diagnostics(self, chunk: dict[str, torch.Tensor]) -> None:
        """Affiche les paramètres, commandes et réponses physiques par agent."""
        observations = chunk["obs"].to(self.device)
        actions = chunk["actions"]
        agent_ids = chunk["agent_id"]
        next_observations = chunk["next_obs"]
        observation_names = self.schema[1]
        required_names = (
            "self_vel_x",
            "self_vel_z",
            "self_stunned",
            "self_tackling",
            "self_grounded",
            "ball_rel_x",
            "ball_rel_z",
        )
        missing_names = [name for name in required_names if name not in observation_names]
        if missing_names:
            raise ValueError(
                "Observations nécessaires au diagnostic absentes : "
                + ", ".join(missing_names)
            )
        observation_indices = {
            name: observation_names.index(name) for name in required_names
        }

        with torch.no_grad():
            output = self.actor(observations)
            continuous_mean = output["continuous_mean"]
            continuous_std = torch.exp(output["continuous_log_std"])
            binary_probabilities = torch.sigmoid(output["binary_logits"])

        for agent_id in torch.unique(agent_ids, sorted=True).tolist():
            mask = agent_ids == agent_id
            means = continuous_mean[mask.to(self.device), :2]
            binary_probs = binary_probabilities[mask.to(self.device)]
            selected_actions = actions[mask, :2]
            stds = continuous_std[mask.to(self.device), :2]
            clamped_actions = selected_actions.clamp(-1.0, 1.0)
            command_norm = clamped_actions.norm(dim=1)
            command = torch.where(
                command_norm.unsqueeze(1) > 1.0,
                clamped_actions / command_norm.clamp_min(1.0).unsqueeze(1),
                clamped_actions,
            )
            horizontal_command = command.norm(dim=1)
            velocity = next_observations[mask][
                :, [observation_indices["self_vel_x"], observation_indices["self_vel_z"]]
            ]
            horizontal_speed = velocity.norm(dim=1) * 6.0
            states = next_observations[mask]
            current_states = chunk["obs"][mask]
            normal_mask = (
                ~chunk["terminated"][mask]
                & ~chunk["truncated"][mask]
                & (current_states[:, observation_indices["self_grounded"]] > 0.5)
                & (current_states[:, observation_indices["self_stunned"]] <= 0.5)
                & (current_states[:, observation_indices["self_tackling"]] <= 0.5)
                & (states[:, observation_indices["self_grounded"]] > 0.5)
                & (states[:, observation_indices["self_stunned"]] <= 0.5)
                & (states[:, observation_indices["self_tackling"]] <= 0.5)
            )
            ball = current_states[
                :, [observation_indices["ball_rel_x"], observation_indices["ball_rel_z"]]
            ]
            vector_norms = command.norm(dim=1)
            ball_norms = ball.norm(dim=1)
            alignment_mask = (vector_norms > 1e-8) & (ball_norms > 1e-8)
            alignments = (
                (command * ball).sum(dim=1)
                / (vector_norms * ball_norms).clamp_min(1e-8)
            )

            diagnostics = []
            for axis, axis_name in enumerate(("move_x", "move_z")):
                predicted = means[:, axis]
                chosen = selected_actions[:, axis]
                std = stds[:, axis].mean().item()
                below = (chosen < -1).to(torch.float32).mean().item()
                above = (chosen > 1).to(torch.float32).mean().item()
                diagnostics.append(
                    f"{axis_name}: moyenne={predicted.mean().item():.4f}, "
                    f"min={predicted.min().item():.4f}, "
                    f"max={predicted.max().item():.4f}, std={std:.4f}, "
                    f"f_-={below:.2%}, f_+={above:.2%}"
                )
            normal_count = int(normal_mask.sum().item())
            if normal_count:
                normal_commands = horizontal_command[normal_mask]
                normal_speeds = horizontal_speed[normal_mask]
                normal_message = (
                    f"normal={normal_count}, commande_normale_moyenne="
                    f"{normal_commands.mean().item():.4f}, "
                    f"vitesse_normale_moyenne={normal_speeds.mean().item():.4f} unité/s"
                )
            else:
                normal_message = "aucune donnée admissible en déplacement normal"
            alignment_message = (
                f"alignement_ballon_moyen={alignments[alignment_mask].mean().item():.4f}"
                if alignment_mask.any()
                else "alignement_ballon_indisponible"
            )
            logger.info(
                "Diagnostic actions agent %s (%d transitions) : %s; "
                "commande horizontale moyenne=%.4f; vitesse horizontale moyenne=%.4f unité/s; "
                "stunned=%.2f%%, tackling=%.2f%%, grounded=%.2f%%; "
                "p_tackle=%.2f%%, p_jump=%.2f%%, "
                "tackle_1=%.2f%%, jump_1=%.2f%%; %s; %s",
                agent_id,
                int(mask.sum().item()),
                "; ".join(diagnostics),
                horizontal_command.mean().item(),
                horizontal_speed.mean().item(),
                states[:, observation_indices["self_stunned"]].mean().item() * 100,
                states[:, observation_indices["self_tackling"]].mean().item() * 100,
                states[:, observation_indices["self_grounded"]].mean().item() * 100,
                binary_probs[:, 1].mean().item() * 100,
                binary_probs[:, 2].mean().item() * 100,
                (actions[mask, 5] == 1).to(torch.float32).mean().item() * 100,
                (actions[mask, 6] == 1).to(torch.float32).mean().item() * 100,
                normal_message,
                alignment_message,
            )

    def on_model_ack(
        self,
        connection: UnityConnection,
        header: dict[str, object]
    ) -> None:
        if connection is not self.connection:
            return
        if header.get("behavior") != self.behavior:
            return
        if self.pending_version is None:
            return

        try:
            version = int(header["version"])
        except (KeyError, TypeError, ValueError):
            return
        if version != self.pending_version:
            return

        if header.get("ok") is True:
            self.active_version = self.pending_version
            observation_size = self.schema[0]
            self.buffer = PPORolloutBuffer(
                session_id=self.session_id,
                behavior=self.behavior,
                policy_version=self.active_version,
                observation_size=observation_size
            )
            self.pending_version = None
            self.phase = "collecting"
            self.rollout_max_log_prob_error = 0.0
            self.last_parity = {}
            logger.info("Politique v%d acceptée ; collecte de %d transitions au minimum.",
                        self.active_version, self.config.rollout_size)
            self.collection_started_at = perf_counter()
            return

        self._set_error(RuntimeError(str(header.get("error", "Unity a refusé le modèle."))))

    def _check_parity(self, chunk: dict[str, torch.Tensor]) -> None:
        """Compare chaque action brute à sa log-probabilité fournie par Unity.

        Cette comparaison précède toute optimisation. Une moyenne proche de
        zéro ne suffit pas : des erreurs de signes opposés pourraient s'annuler.
        """
        for name in ("obs", "next_obs", "actions", "rewards", "old_log_probs"):
            if not torch.isfinite(chunk[name]).all():
                raise ValueError(f"Le batch contient des valeurs non finies dans {name}.")
        buttons = chunk["actions"][:, self.actor.BINARY_INDICES]
        if not ((buttons == 0) | (buttons == 1)).all():
            raise ValueError("Les actions binaires doivent valoir 0 ou 1.")
        log_probs = []
        with torch.no_grad():
            for start in range(0, len(chunk["obs"]), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                log_probs.append(self.actor.evaluate_action(
                    chunk["obs"][start:end].to(self.device),
                    chunk["actions"][start:end].to(self.device),
                ).cpu())
        python_log_probs = torch.cat(log_probs)
        old_log_probs = chunk["old_log_probs"]
        metrics = compute_policy_metrics(python_log_probs, old_log_probs, self.config.clip_epsilon)
        errors = (python_log_probs - old_log_probs).abs()
        self.last_parity = {
            **metrics,
            "max_log_prob_error": errors.max().item(),
            "mean_log_prob_error": errors.mean().item(),
        }
        if not all(math.isfinite(value) for value in self.last_parity.values()):
            raise ValueError("La comparaison des log-probabilités n'est pas finie.")
        if self.last_parity["max_log_prob_error"] > self.config.parity_atol:
            raise ValueError(
                "Incohérence Python/Unity : écart maximal de log-probabilité "
                f"{errors.max().item():.6g} > {self.config.parity_atol:.6g}. "
                "Vérifier les actions brutes, l'ordre des sorties et le décodeur exporté."
            )
        self.rollout_max_log_prob_error = max(
            self.rollout_max_log_prob_error, self.last_parity["max_log_prob_error"]
        )

    def on_batch(self, connection: UnityConnection, batch: Batch) -> None:
        """Garde uniquement la version acquittée, puis déclenche un cycle PPO."""
        if connection is not self.connection or batch.behavior != self.behavior:
            return
        if self.config.evaluate:
            self._on_evaluation_batch(connection, batch)
            return
        if self.tb_logger is not None:
            # Séparé du filtrage du buffer PPO : ces épisodes sont déjà terminés même si les transitions du lot sont ignorées.
            try:
                self.episode_summaries.extend(getattr(batch, "episodes", None) or ())
            except Exception:
                logger.warning("Suivi TensorBoard : résumés d'épisodes ignorés", exc_info=True)
        if self.phase != "collecting":
            self.ignored_transitions += len(batch)
            return
        try:
            session_id = (connection.hello or {}).get("session")
            accepted = self.buffer.add_batch(batch, batch.batch_id, session_id)
            self.ignored_transitions += len(batch) - accepted
            if not accepted:
                return
            # add_batch a déjà copié et filtré ces données sur CPU.
            self._check_parity(self.buffer.chunks[-1])
            logger.debug(
                "Batch %d : %d/%d transitions retenues, buffer=%d, "
                "écart logp max=%.3g, ratio moyen=%.6f.",
                batch.batch_id, accepted, len(batch), len(self.buffer),
                self.last_parity["max_log_prob_error"], self.last_parity["ratio_mean"],
            )
            if len(self.buffer) < self.config.rollout_size:
                return

            sample_count = len(self.buffer)
            logger.info("Collecte v%d prête : %d transitions, écart logp max=%.3g.",
                        self.active_version, sample_count, self.rollout_max_log_prob_error)
            if self.config.collect_only:
                # Vérifie aussi la reconstruction temporelle, sans conserver
                # indéfiniment les données ni modifier réseaux ou optimiseurs.
                segments = self.buffer.build_segment()
                logger.info("Diagnostic : %d segments valides, aucune mise à jour.", len(segments))
                self.buffer.clear()
                self.rollout_max_log_prob_error = 0.0
                return

            self.phase = "training"
            start = perf_counter()
            self.last_history = train_ppo(
                actor=self.actor, critic=self.critic,
                actor_optimizer=self.actor_optimizer, critic_optimizer=self.critic_optimizer,
                buffer=self.buffer, gamma=self.config.gamma, lambda_=self.config.lambda_,
                clip_epsilon=self.config.clip_epsilon, max_grad_norm=self.config.max_grad_norm,
                num_epochs=self.config.num_epochs, mini_batch_size=self.config.mini_batch_size,
                target_kl=self.config.target_kl,
            )
            self.update_count += 1
            self.trained_transitions += sample_count
            for entry in self.last_history:
                logger.info(
                    "Cycle %d / époque %d : acteur=%.5f critique=%.5f "
                    "KL=%.5g clip=%.3f ratio=%.5f arrêt_KL=%s.",
                    self.update_count, entry["epoch"], entry["actor_loss"], entry["critic_loss"],
                    entry["approx_kl"], entry["clip_fraction"], entry["ratio_mean"], entry["kl_limit_reached"],
                )
            logger.info("Entraînement terminé en %.2f s ; %d transitions entraînées, %d ignorées au total.",
                        perf_counter() - start, self.trained_transitions, self.ignored_transitions)
            self._log_tensorboard(sample_count, start)
            self._save_checkpoint()
            self._publish_actor()
        except Exception as error:
            # Le serveur conserve la connexion après une exception : il faut
            # donc interdire explicitement de réentraîner ce buffer.
            self._set_error(error)
            raise

    def on_disconnect(self, connection: UnityConnection) -> None:
        if connection is not self.connection:
            return

        self.buffer = None
        self.connection = None
        self.session_id = None
        self.pending_version = None
        self.active_version = None
        self.phase = "disconnected"
        self.episode_summaries.clear()
        logger.info("Unity déconnecté ; collecte abandonnée, réseaux et optimiseurs conservés en mémoire.")

    def _log_tensorboard(self, sample_count: int, optimization_started_at: float) -> None:
        """Écrit le cycle qui vient de se terminer."""
        if self.tb_logger is None:
            return
        try:
            step = self.trained_transitions
            extras = {
                "time/optimization_s": perf_counter() - optimization_started_at,
                "data/rollout_transitions": sample_count,
                "data/ignored_transitions_total": self.ignored_transitions,
                "parity/max_log_prob_error": self.rollout_max_log_prob_error,
            }
            if self.collection_started_at is not None:
                extras["time/collection_s"] = optimization_started_at - self.collection_started_at
            # Exploration : écarts-types réellement utilisés (log_std borné puis exponentié).
            with torch.no_grad():
                stds = torch.exp(self.actor.log_std.clamp(-5.0, 2.0)).cpu().tolist()
            for name, std in zip(CONTINUOUS_STD_NAMES, stds):
                extras[f"exploration/std_{name}"] = std

            self.tb_logger.log_cycle(step, self.last_history, extras)
            self.tb_logger.log_episodes(step, list(self.episode_summaries))
            self.episode_summaries.clear()
        except Exception:
            # Le suivi est secondaire : on ne met jamais le trainer en erreur pour lui.
            logger.warning("Suivi TensorBoard ignoré pour ce cycle.", exc_info=True)

    def close(self) -> None:
        """Vide et ferme TensorBoard ; à appeler à l'arrêt de Python."""
        if self.tb_logger is not None:
            self.tb_logger.close()
            self.tb_logger = None
