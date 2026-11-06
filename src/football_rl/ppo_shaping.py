"""Signal d'approche du ballon, calculé sur les transitions reçues d'Unity.

Ce module ne décide pas des actions et ne modifie pas la loss PPO. Il ajoute
un terme à la récompense AVANT le calcul des avantages et des cibles du critique.
Les positions relatives sont déjà divisées par positionScale dans Unity :
leur norme horizontale est donc une distance normalisée, pas une distance en mètres.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Sequence

import torch


class BallApproachShaper:
    """Applique F(s, s') = alpha * (gamma * Phi(s') - Phi(s)), Phi(s) = -d(s).

Le même gamma que celui du GAE est utilisé, une fois par décision enregistrée
(pas une fois par frame physique). Aucun facteur dt supplémentaire n'est requis.
"""

    REQUIRED_NAMES = ("ball_rel_x", "ball_rel_z")

    def __init__(self, observation_names: Sequence[str], gamma: float, coefficient: float):
        for name, value in (("gamma", gamma), ("coefficient", coefficient)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} doit être un nombre fini.")
        if not 0 <= gamma <= 1 or coefficient < 0:
            raise ValueError("gamma doit être dans [0, 1] et coefficient >= 0.")
        for name in self.REQUIRED_NAMES:
            if observation_names.count(name) != 1:
                raise ValueError(f"Le shaping exige une observation unique {name!r}.")
        self.indices = [observation_names.index(name) for name in self.REQUIRED_NAMES]
        self.gamma = gamma
        self.coefficient = coefficient
        self.can_kick_index = (
            observation_names.index("self_can_kick")
            if observation_names.count("self_can_kick") == 1 else None
        )

    def distance(self, observations: torch.Tensor) -> torch.Tensor:
        """Ignore la hauteur : sauter ne doit pas modifier le potentiel d'approche."""
        return torch.linalg.vector_norm(observations[:, self.indices], dim=1)

    @torch.no_grad()
    def apply(self, chunk: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Retourne une copie du dictionnaire, sans altérer la récompense du pont.

        Chaque ligne utilise SON obs et SON next_obs. On ne soustrait jamais deux
        lignes voisines : elles pourraient appartenir à des agents, épisodes ou
        versions différents. Le pont capture next_obs avant le repositionnement.
        """
        if "environment_rewards" in chunk:
            raise ValueError("Le shaping a déjà été appliqué à ces transitions.")
        distance = self.distance(chunk["obs"])
        next_distance = self.distance(chunk["next_obs"])
        # Vraie fin (but) : potentiel de l'état absorbant = 0. Une troncature
        # (timeout, interruption) conserve Phi(next_obs), comme le bootstrap GAE.
        # Ne pas annuler ce potentiel à la fin d'un batch ou d'un rollout.
        next_potential = torch.where(chunk["terminated"], 0.0, -next_distance)
        shaping = self.coefficient * (self.gamma * next_potential + distance)
        rewards = chunk["rewards"] + shaping
        if not torch.isfinite(shaping).all() or not torch.isfinite(rewards).all():
            raise ValueError("Le shaping a produit des récompenses non finies.")
        return {
            **chunk,
            "environment_rewards": chunk["rewards"].clone(),
            "shaping_rewards": shaping,
            "rewards": rewards,
        }

    @torch.no_grad()
    def metrics(self, data: dict[str, torch.Tensor]) -> dict[str, float]:
        """Moyennes par transition du rollout, distinctes des retours Unity."""
        distance = self.distance(data["obs"])
        next_distance = self.distance(data["next_obs"])
        base = data.get("environment_rewards", data["rewards"])
        shaping = data.get("shaping_rewards", torch.zeros_like(base))
        metrics = {
            "reward/environment_mean": base.mean().item(),
            "reward/shaping_mean": shaping.mean().item(),
            "reward/shaping_abs_mean": shaping.abs().mean().item(),
            "reward/training_mean": data["rewards"].mean().item(),
            "approach/distance_mean_normalized": distance.mean().item(),
            "approach/distance_decrease_mean_normalized": (distance - next_distance).mean().item(),
        }
        if self.can_kick_index is not None:
            metrics["approach/can_kick_fraction"] = (
                data["obs"][:, self.can_kick_index] > 0.5
            ).float().mean().item()
        return metrics


@dataclass
class _EpisodeApproach:
    next_step: int = 0
    complete: bool = True
    first_opportunity: int | None = None


class KickOpportunityTracker:
    """Mesure les occasions de frappe, PAS les contacts physiques avec le ballon.

    self_can_kick signale un ballon dans la zone de frappe à un instant observé.
    Un contact entre deux décisions peut ne pas être visible. Le délai est en
    décisions, car le pont ne transmet pas le pas de temps physique.
    """

    def __init__(self, observation_names: Sequence[str]):
        self.index = (
            observation_names.index("self_can_kick")
            if observation_names.count("self_can_kick") == 1 else None
        )
        self.states: OrderedDict[tuple[int, int], _EpisodeApproach] = OrderedDict()
        self.completed = 0
        self.with_opportunity = 0
        self.first_step_sum = 0
        self.incomplete = 0

    def observe(self, batch) -> None:
        """Suit les épisodes à travers les batches ET les versions de politique.

        Les résumés sont traités après les transitions. Une observation manquante,
        un doublon ou un début d'épisode absent exclut l'épisode de ces mesures.
        Aucun état n'est conservé à travers une déconnexion (nouveau tracker).
        """
        if self.index is None:
            return
        for i in range(len(batch.obs)):
            key = (int(batch.agent_id[i]), int(batch.episode_id[i]))
            state = self.states.setdefault(key, _EpisodeApproach())
            step = int(batch.step[i])
            state.complete &= step == state.next_step
            state.next_step = step + 1
            if state.first_opportunity is None:
                if batch.obs[i, self.index] > 0.5:
                    state.first_opportunity = step
                elif batch.next_obs[i, self.index] > 0.5:
                    state.first_opportunity = step + 1
            # Protection contre des résumés perdus : pas de croissance illimitée.
            if len(self.states) > 10_000:
                self.states.popitem(last=False)

        for summary in getattr(batch, "episodes", None) or ():
            key = (int(summary["agent_id"]), int(summary["episode_id"]))
            state = self.states.pop(key, None)
            if summary["outcome"] not in {"goal_scored", "goal_conceded", "timeout"}:
                continue
            if state is None or not state.complete or state.next_step != int(summary["length"]):
                self.incomplete += 1
                continue
            self.completed += 1
            if state.first_opportunity is not None:
                self.with_opportunity += 1
                self.first_step_sum += state.first_opportunity

    def pop_metrics(self) -> dict[str, float]:
        """Agrège les épisodes clos depuis le dernier cycle, puis vide ces compteurs."""
        if self.index is None:
            return {}
        metrics = {
            "approach/complete_sporting_episodes": self.completed,
            "approach/incomplete_sporting_episodes": self.incomplete,
        }
        if self.completed:
            metrics["approach/episodes_with_kick_opportunity_fraction"] = self.with_opportunity / self.completed
        # Ne pas écrire zéro lorsqu'aucune occasion n'a été observée : cela
        # ressemblerait à tort à un premier accès immédiat au ballon.
        if self.with_opportunity:
            metrics["approach/first_kick_opportunity_step_mean"] = self.first_step_sum / self.with_opportunity
        self.completed = self.with_opportunity = self.first_step_sum = self.incomplete = 0
        return metrics
