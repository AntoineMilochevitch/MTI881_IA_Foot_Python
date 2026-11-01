"""Suivi TensorBoard de l'entrainement PPO.

Ce module reçoit les données et les écrit dans un dossier d'évènement TensorBoard.
Une erreur de suivi n'intérromps pas l'entrainement, mais donner un avertissement.
"""

from collections import Counter
from dataclasses import asdict, is_dataclass
import json
import logging
import statistics
from typing import Any

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

KNOWN_OUTCOMES = ("goal_scored", "goal_conceded", "timeout", "interrupted")

class PPOTensorBoardLogger:
    """Écrit les métriques d'une expérience dans un dossier TensorBoard"""

    def __init__(self, log_dir: str, config: Any = None) -> None:
        self.writer = SummaryWriter(log_dir=str(log_dir))
        if config is not None:
            self._log_config(config)

    def _log_config(self, config: Any) -> None:
        """Garde les réglages de l'expérience, utile pour comparer plusieurs runs"""
        try:
            values = asdict(config) if is_dataclass(config) else dict(config)
            text = json.dumps(values, indent=2, default=str)
            text = "\n".join("    " + line for line in text.splitlines())
            self.writer.add_text("config", text, global_step=0)
        except Exception:
            logger.warning("Suivi TensorBoard : configuration non enregistrée.", exc_info=True)

    def _write(self, step: int, scalars: dict[str, float]) -> None:
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, float(value), global_step=step)
        self.writer.flush()

    def log_cycle(self, step: int, history: list[dict[str, Any]], extras: dict[str, float] | None = None) -> None:
        """Enregistre un cycle PPO à partir de l'historique de train_ppo
        history : une entrée par epoch effectuée. On garde la dernière.
        KL, clip et ratio y sont mesurées par rapport à l'ancienne politique.
        extras : autres mesures déjà nommées (durées, exploration,...)
        """
        try:
            if not history:
                return
            last = history[-1]
            scalars = {
                "loss/actor": last["actor_loss"],
                "loss/critic": last["critic_loss"],
                "grad/actor_norm": last["actor_grad_norm"],
                "grad/critic_norm": last["critic_grad_norm"],
                "ppo/approx_kl": last["approx_kl"],
                "ppo/clip_fraction": last["clip_fraction"],
                "ppo/ratio_mean": last["ratio_mean"],
                "ppo/epochs_done": len(history),
                "ppo/kl_limit_reached": float(last["kl_limit_reached"])
            }
            scalars.update(extras or {})
            self._write(step, scalars)
        except Exception:
            logger.warning("Suivi TensorBoard : cycle non enregistré.", exc_info=True)

    def log_episodes(self, step: int, episodes: list[dict[str, Any]]) -> None:
        """Agrège les résumés d'épisodes terminés pendant la collecte"""
        try:
            if not episodes:
                return
            returns = [float(episode["return"]) for episode in episodes]
            lengths = [float(episode["length"]) for episode in episodes]
            outcomes = Counter(str(episode["outcome"]) for episode in episodes)

            scalars = {
                "episodes/count": len(episodes),
                "episodes/return_mean": statistics.fmean(returns),
                "episodes/return_std": statistics.pstdev(returns),
                "episodes/length_mean": statistics.fmean(lengths),
            }
            names = list(KNOWN_OUTCOMES) + [o for o in outcomes if o not in KNOWN_OUTCOMES]
            for outcome in names:
                scalars[f"episodes/outcome_{outcome}"] = outcomes.get(outcome, 0) / len(episodes)
            self._write(step, scalars)
        except Exception:
            logger.warning("Suivi TensorBoard : épisodes non enregistrés.", exc_info=True)

    def close(self) -> None:
        """Vide et ferme le fichier d'évènements"""
        self.writer.flush()
        self.writer.close()



