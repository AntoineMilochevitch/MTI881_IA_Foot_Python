"""Agent DQN avec support optionnel du Double DQN.

Implémente l'algorithme DQN classique (Mnih et al., 2015) avec :
- Tampon de rejeu pour casser les corrélations temporelles.
- Réseau cible mis à jour périodiquement pour stabiliser l'apprentissage.
- Exploration ε-greedy avec décroissance linéaire.
- Mode Double DQN (van Hasselt et al., 2016) activable par configuration,
  qui découple la sélection et l'évaluation de l'action pour réduire
  la surestimation des Q-valeurs.

La discrétisation de l'espace d'action est gérée par ``ActionDiscretizer``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from football_rl.agent import FootballAgent
from football_rl.discretization import ActionDiscretizer, DiscretizationConfig
from football_rl.dqn_network import DQNNetwork
from football_rl.replay_buffer import ReplayBuffer
from football_rl.types import (
    EpisodeBatch,
    FootballAction,
    ModelArtifact,
    Observation,
    ObservationLayout,
)


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Hyperparamètres de l'agent DQN.

    Attributes:
        lr: Taux d'apprentissage de l'optimiseur Adam.
        gamma: Facteur d'actualisation des récompenses futures.
        epsilon_start: Probabilité initiale d'exploration aléatoire.
        epsilon_end: Probabilité minimale d'exploration.
        epsilon_decay_steps: Nombre de pas pour la décroissance linéaire de ε.
        buffer_capacity: Capacité du tampon de rejeu.
        batch_size: Taille des lots échantillonnés pour l'entraînement.
        target_update_freq: Fréquence de copie vers le réseau cible (en mises à jour).
        hidden_size: Nombre de neurones par couche cachée du MLP.
        double_dqn: Activer le Double DQN pour réduire la surestimation.
        learning_starts: Nombre minimum de transitions avant le premier entraînement.
    """

    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    buffer_capacity: int = 100_000
    batch_size: int = 64
    target_update_freq: int = 1_000
    hidden_size: int = 256
    double_dqn: bool = False
    learning_starts: int = 1_000


class DQNAgent(FootballAgent):
    """Agent DQN concret avec discrétisation de l'espace d'action.

    Flux d'entraînement :
    1. Recevoir un ``EpisodeBatch`` depuis Unity.
    2. Convertir chaque transition en (obs, action_index, reward, next_obs, done).
    3. Stocker dans le tampon de rejeu.
    4. Échantillonner un mini-lot et calculer la perte de Bellman.
    5. Rétropropager et mettre à jour le réseau en ligne.
    6. Copier périodiquement les poids vers le réseau cible.

    Args:
        layout: Description du vecteur d'observation.
        disc_config: Paramètres de discrétisation des actions.
        dqn_config: Hyperparamètres DQN.
        device: Périphérique PyTorch (cpu ou cuda).
    """

    def __init__(
        self,
        layout: ObservationLayout,
        disc_config: DiscretizationConfig,
        dqn_config: DQNConfig,
        device: torch.device | None = None,
    ) -> None:
        self.layout = layout
        self.config = dqn_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Discrétisation.
        self.discretizer = ActionDiscretizer(disc_config)

        # Réseaux.
        self.online_net = DQNNetwork(
            layout.size, self.discretizer.num_actions, dqn_config.hidden_size,
        ).to(self.device)

        self.target_net = DQNNetwork(
            layout.size, self.discretizer.num_actions, dqn_config.hidden_size,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimiseur.
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=dqn_config.lr)

        # Tampon de rejeu.
        self.replay_buffer = ReplayBuffer(dqn_config.buffer_capacity, layout.size)

        # Compteurs pour ε-greedy et mise à jour du réseau cible.
        self._step_count = 0
        self._update_count = 0

    # Exploration ε-greedy

    @property
    def epsilon(self) -> float:
        """Probabilité courante d'exploration (décroissance linéaire)."""
        cfg = self.config
        progress = min(1.0, self._step_count / max(cfg.epsilon_decay_steps, 1))
        return cfg.epsilon_start + progress * (cfg.epsilon_end - cfg.epsilon_start)

    # Interface FootballAgent

    def select_action(self, observation: Observation) -> FootballAction:
        """Choisit une action par ε-greedy sur les Q-valeurs.

        Avec probabilité ε, une action aléatoire uniforme est choisie.
        Sinon, l'action avec la plus grande Q-valeur estimée est sélectionnée.
        """
        self._step_count += 1

        if np.random.random() < self.epsilon:
            index = np.random.randint(self.discretizer.num_actions)
        else:
            obs_t = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(obs_t)["q_values"]
            index = int(q_values.argmax(dim=1).item())

        return self.discretizer.index_to_action(index)

    def update(self, batch: EpisodeBatch) -> dict[str, float]:
        """Intègre un lot d'épisodes et effectue des mises à jour DQN.

        Les transitions de chaque épisode sont ajoutées au tampon de rejeu.
        Puis, si le tampon contient suffisamment de transitions, un mini-lot
        est échantillonné et une étape de gradient est effectuée.

        Returns:
            Métriques d'apprentissage : perte, ε courant, taille du tampon,
            Q-valeur moyenne et nombre de transitions ajoutées.
        """
        # Stocker les transitions dans le tampon de rejeu.
        transitions_added = 0
        for episode in batch.episodes:
            for transition in episode.transitions:
                action_index = self.discretizer.action_to_index(transition.action)
                self.replay_buffer.add(
                    observation=transition.observation,
                    action=action_index,
                    reward=transition.reward,
                    next_observation=transition.next_observation,
                    terminated=transition.terminated,
                )
                transitions_added += 1

        metrics: dict[str, float] = {
            "epsilon": self.epsilon,
            "buffer_size": float(len(self.replay_buffer)),
            "transitions_added": float(transitions_added),
        }

        # Attendre d'avoir assez de transitions avant d'entraîner.
        if len(self.replay_buffer) < self.config.learning_starts:
            metrics["loss"] = 0.0
            metrics["mean_q"] = 0.0
            return metrics

        # Échantillonner et entraîner.
        loss_val, mean_q = self._train_step()
        metrics["loss"] = loss_val
        metrics["mean_q"] = mean_q

        return metrics

    def export_model(self) -> ModelArtifact:
        """Exporte le réseau en ligne au format TorchScript pour Unity."""
        scripted = torch.jit.script(self.online_net.cpu())
        buffer = torch.jit.save(scripted, "/dev/null")  # TODO: sérialiser en bytes
        self.online_net.to(self.device)
        raise NotImplementedError(
            "L'export TorchScript vers bytes pour Unity reste à implémenter."
        )

    def save(self, path: Path) -> None:
        """Sauvegarde l'état complet de l'agent (réseaux, optimiseur, compteurs)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self._step_count,
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: Path, device: torch.device) -> None:
        """Charge l'état complet de l'agent depuis un checkpoint."""
        self.device = device
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step_count = checkpoint["step_count"]
        self._update_count = checkpoint["update_count"]
        self.online_net.to(device)
        self.target_net.to(device)

    # Logique interne DQN

    def _train_step(self) -> tuple[float, float]:
        """Une étape d'entraînement DQN sur un mini-lot du tampon de rejeu.

        Calcule la cible de Bellman selon le mode (DQN ou Double DQN) :
        - **DQN** : y = r + γ · max_a' Q_target(s', a')
        - **Double DQN** : y = r + γ · Q_target(s', argmax_a' Q_online(s', a'))

        Returns:
            (perte, Q-valeur moyenne) du mini-lot.
        """
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)

        # Q-valeurs du réseau en ligne pour les actions choisies.
        q_values = self.online_net(batch.observations)["q_values"]
        q_selected = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        # Cible de Bellman.
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN : le réseau en ligne sélectionne, le cible évalue.
                online_next_q = self.online_net(batch.next_observations)["q_values"]
                best_actions = online_next_q.argmax(dim=1, keepdim=True)
                target_next_q = self.target_net(batch.next_observations)["q_values"]
                next_q = target_next_q.gather(1, best_actions).squeeze(1)
            else:
                # DQN classique : le réseau cible fait tout.
                target_next_q = self.target_net(batch.next_observations)["q_values"]
                next_q = target_next_q.max(dim=1).values

            target = batch.rewards + self.config.gamma * next_q * (1.0 - batch.terminated)

        # Perte de Huber (smooth L1) pour la robustesse aux valeurs aberrantes.
        loss = nn.functional.smooth_l1_loss(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Clipper les gradients pour la stabilité.
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Mise à jour périodique du réseau cible.
        self._update_count += 1
        if self._update_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item()), float(q_selected.mean().item())
