"""Adaptateur DQN pour les anciens contrats Python (``FootballAgent``, ``EpisodeBatch``).

L'algorithme vit dans ``football_rl.dqn`` (``DQNLearner``). Cette classe ne fait
que traduire les actions ``FootballAction`` en indices via ``ActionDiscretizer``.

Le vrai branchement sur le jeu passe par le pont ``iafoot`` : voir
``football_rl.dqn_unity``. Ce module est conservé pour les tests existants et
pourra disparaître lors de la fusion avec la branche ``ppo``, qui supprime
``agent.py`` et ``types.py``.
"""

from pathlib import Path

import torch

from football_rl.agent import FootballAgent
from football_rl.discretization import ActionDiscretizer, DiscretizationConfig
from football_rl.dqn import DQNConfig, DQNLearner
from football_rl.types import (
    EpisodeBatch,
    FootballAction,
    ModelArtifact,
    Observation,
    ObservationLayout,
)

__all__ = ["DQNAgent", "DQNConfig"]


class DQNAgent(FootballAgent):
    """DQN sur l'espace d'action discrétisé des anciens contrats."""

    def __init__(
        self,
        layout: ObservationLayout,
        disc_config: DiscretizationConfig,
        dqn_config: DQNConfig,
        device: torch.device | None = None,
    ) -> None:
        self.layout = layout
        self.config = dqn_config
        self.discretizer = ActionDiscretizer(disc_config)
        self.learner = DQNLearner(layout.size, self.discretizer.num_actions, dqn_config, device)

    @property
    def device(self) -> torch.device:
        return self.learner.device

    @property
    def epsilon(self) -> float:
        return self.learner.epsilon

    def select_action(self, observation: Observation) -> FootballAction:
        """ε-greedy ; chaque appel compte comme un pas d'environnement."""
        self.learner.env_steps += 1
        return self.discretizer.index_to_action(self.learner.act(observation))

    def update(self, batch: EpisodeBatch) -> dict[str, float]:
        """Ajoute les transitions au tampon, puis fait les pas de gradient dus (un tous les ``train_freq``)."""
        added = 0
        for episode in batch.episodes:
            for transition in episode.transitions:
                self.learner.store(
                    transition.observation,
                    self.discretizer.action_to_index(transition.action),
                    transition.reward,
                    transition.next_observation,
                    transition.terminated,
                )
                added += 1

        result = self.learner.train_after(added)
        return {
            "epsilon": self.epsilon,
            "buffer_size": float(len(self.learner.replay_buffer)),
            "transitions_added": float(added),
            "gradient_steps": result["steps"],
            "loss": result.get("loss", 0.0),
            "mean_q": result.get("mean_q", 0.0),
        }

    def export_model(self) -> ModelArtifact:
        raise NotImplementedError(
            "L'export vers Unity passe par iafoot : voir football_rl.dqn_unity "
            "(models.from_torch(learner.online_net.layers) + discrete_decoder)."
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.learner.state_dict(), path)

    def load(self, path: Path, device: torch.device) -> None:
        self.learner.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        self.learner.online_net.to(device)
        self.learner.target_net.to(device)
        self.learner.device = device
