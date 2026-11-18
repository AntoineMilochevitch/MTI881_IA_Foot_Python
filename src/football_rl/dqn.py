"""Cœur du DQN, indépendant de l'environnement (CartPole, Unity...).

Implémente l'algorithme DQN (Mnih et al., 2015) avec :
- tampon de rejeu pour casser les corrélations temporelles ;
- réseau cible, mis à jour par copie périodique ou par moyenne de Polyak ;
- exploration ε-greedy avec décroissance linéaire ;
- Double DQN optionnel (van Hasselt et al., 2016), qui découple la sélection
  et l'évaluation de l'action pour réduire la surestimation des Q-valeurs.

``DQNLearner`` ne connaît ni Unity ni le football : il reçoit des transitions
(observation, indice d'action, récompense, observation suivante, terminated)
et fait des pas de gradient. Les adaptateurs (``dqn_unity``, script CartPole)
s'occupent de la collecte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, optim

from football_rl.dqn_network import DQNNetwork
from football_rl.replay_buffer import ReplayBuffer


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Hyperparamètres du DQN.

    Les valeurs par défaut sont la configuration la plus stable trouvée sur
    CartPole-v1 (voir ``scripts/dqn_cartpole.py``). Les réglages d'exploration
    dépendent de l'environnement et sont à ajuster.

    Attributes:
        lr: Taux d'apprentissage de l'optimiseur Adam.
        gamma: Facteur d'actualisation des récompenses futures.
        epsilon_start: Probabilité initiale d'exploration aléatoire.
        epsilon_end: Probabilité minimale d'exploration.
        epsilon_decay_steps: Nombre de transitions pour la décroissance linéaire de ε.
        buffer_capacity: Capacité du tampon de rejeu.
        batch_size: Taille des mini-lots échantillonnés pour l'entraînement.
        target_update_freq: Période de copie vers le réseau cible, en pas de
            gradient (utilisée seulement si ``tau == 1``).
        hidden_size: Nombre de neurones par couche cachée du MLP.
        double_dqn: Activer le Double DQN pour réduire la surestimation.
        learning_starts: Nombre minimum de transitions avant le premier entraînement.
        tau: Mise à jour du réseau cible. 1.0 : copie complète toutes les
            ``target_update_freq`` mises à jour. < 1.0 : moyenne de Polyak
            à chaque mise à jour, θ_cible ← (1 − τ)·θ_cible + τ·θ_en_ligne.
        train_freq: Nombre de nouvelles transitions par pas de gradient
            (1 = un pas de gradient par transition collectée).
        max_grad_norm: Norme maximale des gradients (clipping).
        seed: Graine des tirages aléatoires (initialisation des réseaux,
            exploration, échantillonnage du tampon). None : non reproductible.
    """

    lr: float = 2.5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    buffer_capacity: int = 100_000
    batch_size: int = 128
    target_update_freq: int = 1_000
    hidden_size: int = 256
    double_dqn: bool = True
    learning_starts: int = 1_000
    tau: float = 0.005
    train_freq: int = 1
    max_grad_norm: float = 10.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau doit être dans ]0, 1].")
        if self.train_freq < 1:
            raise ValueError("train_freq doit être >= 1.")
        if self.batch_size < 1 or self.buffer_capacity < self.batch_size:
            raise ValueError("Il faut 1 <= batch_size <= buffer_capacity.")
        if not (0.0 <= self.epsilon_start <= 1.0 and 0.0 <= self.epsilon_end <= 1.0):
            raise ValueError("epsilon_start et epsilon_end doivent être dans [0, 1].")


class DQNLearner:
    """Réseaux, tampon de rejeu, ε et pas de gradient du DQN.

    Compteurs :
        ``env_steps`` : transitions collectées (pilote ε) ; c'est à l'adaptateur
        de l'incrémenter, car lui seul sait combien de transitions ont été jouées.
        ``gradient_steps`` : pas de gradient effectués.
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: DQNConfig,
        device: torch.device | None = None,
    ) -> None:
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.config = config
        self.device = device or torch.device("cpu")

        # L'initialisation des poids passe par la graine globale de PyTorch,
        # l'exploration et le tampon par leurs propres générateurs.
        if config.seed is not None:
            torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)

        self.online_net = DQNNetwork(observation_size, num_actions, config.hidden_size).to(self.device)
        self.target_net = DQNNetwork(observation_size, num_actions, config.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)

        buffer_seed = None if config.seed is None else config.seed + 1
        self.replay_buffer = ReplayBuffer(config.buffer_capacity, observation_size, seed=buffer_seed)

        self.env_steps = 0
        self.gradient_steps = 0
        self._train_credit = 0.0

    # ------------------------------------------------------------------ Exploration

    @property
    def epsilon(self) -> float:
        """Probabilité courante d'exploration (décroissance linéaire sur ``env_steps``)."""
        cfg = self.config
        progress = min(1.0, self.env_steps / max(cfg.epsilon_decay_steps, 1))
        return cfg.epsilon_start + progress * (cfg.epsilon_end - cfg.epsilon_start)

    def q_values(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.online_net(obs_t)["q_values"][0].cpu().numpy()

    def greedy_action(self, observation: NDArray[np.float32]) -> int:
        return int(np.argmax(self.q_values(observation)))

    def act(self, observation: NDArray[np.float32]) -> int:
        """ε-greedy local (pour un environnement Python ; dans Unity, c'est le décodeur qui explore)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        return self.greedy_action(observation)

    # ------------------------------------------------------------------ Données

    def store(self, observation, action: int, reward: float, next_observation, terminated: bool) -> None:
        self.replay_buffer.add(observation, action, reward, next_observation, terminated)

    def store_batch(self, observations, actions, rewards, next_observations, terminated) -> None:
        self.replay_buffer.add_batch(observations, actions, rewards, next_observations, terminated)

    # ------------------------------------------------------------------ Entraînement

    def train_after(self, new_transitions: int) -> dict[str, float]:
        """Fait les pas de gradient dus pour ``new_transitions`` nouvelles transitions.

        Un pas tous les ``train_freq`` transitions ; les fractions sont reportées
        au prochain appel. Aucun entraînement avant ``learning_starts`` transitions
        dans le tampon (et pas de rattrapage ensuite).

        Returns:
            ``steps`` (pas effectués), et si > 0 : ``loss`` et ``mean_q`` moyens.
        """
        if len(self.replay_buffer) < max(self.config.learning_starts, self.config.batch_size):
            self._train_credit = 0.0
            return {"steps": 0.0}

        self._train_credit += new_transitions / self.config.train_freq
        steps = int(self._train_credit)
        self._train_credit -= steps
        if steps == 0:
            return {"steps": 0.0}

        losses, qs = [], []
        for _ in range(steps):
            loss, mean_q = self.train_step()
            losses.append(loss)
            qs.append(mean_q)
        return {"steps": float(steps), "loss": float(np.mean(losses)), "mean_q": float(np.mean(qs))}

    def train_step(self) -> tuple[float, float]:
        """Un pas de gradient sur un mini-lot du tampon de rejeu.

        Cible de Bellman selon le mode :
        - **DQN** : y = r + γ · max_a' Q_cible(s', a')
        - **Double DQN** : y = r + γ · Q_cible(s', argmax_a' Q_en_ligne(s', a'))
        Seul ``terminated`` coupe le bootstrap : une coupure (limite de temps,
        remise en jeu) n'est pas une fin réelle de l'épisode.

        Returns:
            (perte, Q-valeur moyenne des actions jouées) du mini-lot.
        """
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)

        q_values = self.online_net(batch.observations)["q_values"]
        q_selected = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_next_q = self.target_net(batch.next_observations)["q_values"]
            if self.config.double_dqn:
                best_actions = self.online_net(batch.next_observations)["q_values"].argmax(dim=1, keepdim=True)
                next_q = target_next_q.gather(1, best_actions).squeeze(1)
            else:
                next_q = target_next_q.max(dim=1).values
            target = batch.rewards + self.config.gamma * next_q * (1.0 - batch.terminated)

        # Perte de Huber (smooth L1) : robuste aux erreurs de Bellman aberrantes.
        loss = nn.functional.smooth_l1_loss(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.config.tau < 1.0:
            with torch.no_grad():
                for target_p, online_p in zip(self.target_net.parameters(), self.online_net.parameters()):
                    target_p.lerp_(online_p, self.config.tau)
        elif self.gradient_steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item()), float(q_selected.mean().item())

    # ------------------------------------------------------------------ Sauvegarde

    def state_dict(self) -> dict[str, Any]:
        """État entraînable (le tampon de rejeu n'est pas sauvegardé : il se remplit à la reprise)."""
        return {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env_steps": self.env_steps,
            "gradient_steps": self.gradient_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.env_steps = int(state["env_steps"])
        self.gradient_steps = int(state["gradient_steps"])
