"""Tests de l'agent DQN : discrétisation, tampon de rejeu et entraînement."""

import numpy as np
import torch

from football_rl.discretization import ActionDiscretizer, DiscretizationConfig
from football_rl.dqn_agent import DQNAgent, DQNConfig
from football_rl.dqn_network import DQNNetwork
from football_rl.env import Football3DEnv
from football_rl.replay_buffer import ReplayBuffer
from football_rl.types import (
    Episode,
    EpisodeBatch,
    ObservationLayout,
    Transition,
)


def test_discretizer_roundtrip() -> None:
    """Vérifie la bijection indice ↔ action."""
    cfg = DiscretizationConfig(move_bins_per_axis=3, shoot_bins=2, enable_tackle=False)
    disc = ActionDiscretizer(cfg)

    assert disc.num_actions == 3**3 * 2 * 2  # 108

    for i in range(disc.num_actions):
        action = disc.index_to_action(i)
        assert disc.action_to_index(action) == i, f"Échec au roundtrip pour l'indice {i}"

    print(f"  Discrétisation : {disc.num_actions} actions, roundtrip OK")


def test_discretizer_with_tackle() -> None:
    """Vérifie le nombre d'actions avec tacle."""
    cfg = DiscretizationConfig(move_bins_per_axis=3, shoot_bins=2, enable_tackle=True)
    disc = ActionDiscretizer(cfg)
    assert disc.num_actions == 3**3 * 2 * 2 * 2  # 216
    print(f"  Discrétisation avec tacle : {disc.num_actions} actions")


def test_replay_buffer() -> None:
    """Vérifie l'ajout, la taille et l'échantillonnage du tampon."""
    obs_size = 57
    buf = ReplayBuffer(capacity=100, observation_size=obs_size)
    rng = np.random.default_rng(42)

    assert len(buf) == 0

    for _ in range(50):
        buf.add(
            observation=rng.standard_normal(obs_size).astype(np.float32),
            action=rng.integers(0, 108),
            reward=rng.standard_normal(),
            next_observation=rng.standard_normal(obs_size).astype(np.float32),
            terminated=bool(rng.integers(0, 2)),
        )

    assert len(buf) == 50

    batch = buf.sample(16, torch.device("cpu"))
    assert batch.observations.shape == (16, obs_size)
    assert batch.actions.shape == (16,)
    assert batch.rewards.shape == (16,)
    assert batch.next_observations.shape == (16, obs_size)
    assert batch.terminated.shape == (16,)

    print("  Tampon de rejeu : ajout et échantillonnage OK")


def test_dqn_network() -> None:
    """Vérifie les dimensions de sortie du réseau."""
    net = DQNNetwork(observation_size=57, num_actions=108)
    obs = torch.randn(4, 57)
    out = net(obs)
    assert "q_values" in out
    assert out["q_values"].shape == (4, 108)
    print("  Réseau DQN : forward OK, sortie (4, 108)")


def test_dqn_agent_select_action() -> None:
    """Vérifie que l'agent retourne une action valide."""
    layout = ObservationLayout(max_teammates=2, max_opponents=3)
    disc_cfg = DiscretizationConfig(move_bins_per_axis=3, shoot_bins=2)
    dqn_cfg = DQNConfig(epsilon_start=0.0)  # Pas d'exploration.

    agent = DQNAgent(layout, disc_cfg, dqn_cfg, device=torch.device("cpu"))
    obs = np.random.default_rng(0).standard_normal(layout.size).astype(np.float32)
    action = agent.select_action(obs)

    assert "move_target" in action
    assert "shoot" in action
    assert "jump" in action
    assert action["move_target"].shape == (3,)
    assert action["shoot"].shape == (1,)
    print("  Agent DQN : select_action OK")


def test_dqn_agent_update() -> None:
    """Vérifie un cycle complet : remplir le tampon puis entraîner."""
    layout = ObservationLayout(max_teammates=2, max_opponents=3)
    disc_cfg = DiscretizationConfig(move_bins_per_axis=3, shoot_bins=2)
    dqn_cfg = DQNConfig(
        learning_starts=10,
        buffer_capacity=200,
        batch_size=8,
        target_update_freq=5,
    )

    agent = DQNAgent(layout, disc_cfg, dqn_cfg, device=torch.device("cpu"))
    rng = np.random.default_rng(42)

    # Créer un lot d'épisodes factice.
    transitions = []
    for _ in range(20):
        obs = rng.standard_normal(layout.size).astype(np.float32)
        action = agent.discretizer.index_to_action(rng.integers(0, agent.discretizer.num_actions))
        next_obs = rng.standard_normal(layout.size).astype(np.float32)
        transitions.append(Transition(
            observation=obs,
            action=action,
            reward=float(rng.standard_normal()),
            next_observation=next_obs,
            terminated=False,
            truncated=False,
        ))

    episode = Episode(transitions=tuple(transitions), statistics={})
    batch = EpisodeBatch(episodes=(episode,))

    metrics = agent.update(batch)

    assert "loss" in metrics
    assert "epsilon" in metrics
    assert "buffer_size" in metrics
    assert "mean_q" in metrics
    assert metrics["buffer_size"] == 20.0
    assert metrics["loss"] >= 0.0
    print(f"  Agent DQN update : perte={metrics['loss']:.4f}, "
          f"Q moyen={metrics['mean_q']:.4f}, epsilon={metrics['epsilon']:.3f}")


def test_dqn_agent_with_env() -> None:
    """Vérifie que l'agent interagit correctement avec l'environnement mock."""
    layout = ObservationLayout(max_teammates=2, max_opponents=3)
    disc_cfg = DiscretizationConfig(move_bins_per_axis=3, shoot_bins=2)
    dqn_cfg = DQNConfig(epsilon_start=1.0)  # Exploration pure.

    agent = DQNAgent(layout, disc_cfg, dqn_cfg, device=torch.device("cpu"))
    env = Football3DEnv(layout, enable_tackle=False, max_steps=10)

    obs, info = env.reset(seed=42)
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    assert steps == 10  # max_steps
    print(f"  Agent DQN + env : {steps} pas, récompense totale = {total_reward}")


def test_double_dqn_flag() -> None:
    """Vérifie que le mode Double DQN s'active sans erreur."""
    layout = ObservationLayout(max_teammates=2, max_opponents=3)
    disc_cfg = DiscretizationConfig(move_bins_per_axis=3, shoot_bins=2)
    dqn_cfg = DQNConfig(double_dqn=True, learning_starts=5, batch_size=4)

    agent = DQNAgent(layout, disc_cfg, dqn_cfg, device=torch.device("cpu"))
    rng = np.random.default_rng(0)

    transitions = []
    for _ in range(10):
        obs = rng.standard_normal(layout.size).astype(np.float32)
        action = agent.discretizer.index_to_action(rng.integers(0, agent.discretizer.num_actions))
        next_obs = rng.standard_normal(layout.size).astype(np.float32)
        transitions.append(Transition(
            observation=obs, action=action, reward=0.5,
            next_observation=next_obs, terminated=False, truncated=False,
        ))

    episode = Episode(transitions=tuple(transitions), statistics={})
    batch = EpisodeBatch(episodes=(episode,))
    metrics = agent.update(batch)

    assert metrics["loss"] >= 0.0
    print(f"  Double DQN : perte={metrics['loss']:.4f}, Q moyen={metrics['mean_q']:.4f}")


if __name__ == "__main__":
    print("Tests DQN :")
    test_discretizer_roundtrip()
    test_discretizer_with_tackle()
    test_replay_buffer()
    test_dqn_network()
    test_dqn_agent_select_action()
    test_dqn_agent_update()
    test_dqn_agent_with_env()
    test_double_dqn_flag()
    print("\nTous les tests DQN sont passés !")
