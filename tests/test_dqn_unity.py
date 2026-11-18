"""Tests du DQN branché sur le pont iafoot, sans lancer Unity.

Une fausse connexion remplace Unity : elle reçoit les modèles envoyés et on lui
fait « jouer » des batchs synthétiques. Le modèle exporté est évalué avec
``ModelExport.predict``, l'implémentation numpy de référence du calcul fait par
Unity : si ces tests passent, Unity reçoit le bon réseau.

Lancement (``iafoot`` lié avec scripts/link_unity_bridge.py) :
    $env:PYTHONPATH = "src"; python tests/test_dqn_unity.py
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from iafoot import Batch

from football_rl.dqn import DQNConfig
from football_rl.dqn_actions import ACTION_COLUMNS, ACTION_TYPES, build_action_table
from football_rl.dqn_unity import DQNUnityTrainer, RunSettings, load_checkpoint, save_checkpoint
from football_rl.replay_buffer import ReplayBuffer

OBS = 30


def make_hello(obs_size: int = OBS, model_version: int = 0, actions=None) -> dict:
    actions = actions or [{"name": n, "type": t} for n, t in zip(ACTION_COLUMNS, ACTION_TYPES)]
    return {
        "type": "hello", "protocol": 1, "session": "test", "fixed_delta_time": 0.02, "time_scale": 1.0,
        "batch_size": 1024,
        "behaviors": [{
            "name": "Foot", "obs_size": obs_size, "obs_names": [f"o{i}" for i in range(obs_size)],
            "action_size": 7, "actions": actions, "decision_period": 5, "model_version": model_version,
            "agents": [{"id": 0, "name": "Bleu1", "team": "Blue"}, {"id": 1, "name": "Rouge1", "team": "Red"}],
        }],
    }


class FakeConnection:
    """Ce que le trainer utilise d'une UnityConnection."""

    def __init__(self, hello: dict) -> None:
        self.hello = hello
        self.models: list[SimpleNamespace] = []
        self.configs: list[dict] = []

    @property
    def behaviors(self) -> list[dict]:
        return list(self.hello["behaviors"])

    def send_model(self, model, *, version, behavior="*", decoder=None) -> None:
        self.models.append(SimpleNamespace(model=model, version=version, behavior=behavior, decoder=decoder))

    def set_config(self, **settings) -> None:
        self.configs.append(settings)


def make_batch(obs, action_index, rewards, next_obs, terminated, *, version, agent_id=None, episode_id=None,
               episodes=(), batch_id=0) -> Batch:
    n = len(action_index)
    return Batch(
        behavior="Foot", batch_id=batch_id,
        obs=np.asarray(obs, np.float32), actions=np.zeros((n, 7), np.float32),
        rewards=np.asarray(rewards, np.float32), next_obs=np.asarray(next_obs, np.float32),
        terminated=np.asarray(terminated, bool), truncated=np.zeros(n, bool), log_probs=np.zeros(n, np.float32),
        action_index=np.asarray(action_index, np.int32),
        agent_id=np.asarray(agent_id if agent_id is not None else np.zeros(n), np.int32),
        episode_id=np.asarray(episode_id if episode_id is not None else np.zeros(n), np.int32),
        step=np.arange(n, dtype=np.int32), model_version=np.full(n, version, np.int32),
        episodes=list(episodes), header={},
    )


def random_batch(rng, n, version, num_actions=28, invalid=0) -> Batch:
    actions = rng.integers(0, num_actions, n)
    actions[:invalid] = -1
    return make_batch(rng.standard_normal((n, OBS)), actions, rng.standard_normal(n),
                      rng.standard_normal((n, OBS)), rng.random(n) < 0.05, version=version)


def small_config(**overrides) -> DQNConfig:
    values = dict(seed=0, hidden_size=64, batch_size=32, buffer_capacity=5_000, learning_starts=100,
                  epsilon_decay_steps=1_000, train_freq=4)
    values.update(overrides)
    return DQNConfig(**values)


# --------------------------------------------------------------------------- Tests


def test_connect_publishes_model_matching_torch() -> None:
    trainer = DQNUnityTrainer(small_config(), RunSettings())
    conn = FakeConnection(make_hello(model_version=5))
    trainer.on_connect(conn)

    assert conn.configs[0]["batch_size"] == 512
    sent = conn.models[-1]
    assert sent.version == 6, "la version doit dépasser celle annoncée par Unity"
    assert sent.behavior == "Foot"
    assert sent.decoder.spec["type"] == "discrete"
    assert sent.decoder.spec["mode"] == "epsilon_greedy"
    assert sent.decoder.spec["epsilon"] == 1.0
    (table,) = sent.decoder.tensors.values()
    assert table.shape == (28, 7)

    # Ce qu'Unity vérifie à la réception : son calcul doit redonner la sortie PyTorch jointe.
    check_input, check_output = sent.model.check
    assert np.allclose(sent.model.predict(check_input), check_output, atol=1e-4)
    rng = np.random.default_rng(1)
    for _ in range(5):
        x = rng.standard_normal(OBS).astype(np.float32)
        assert np.allclose(sent.model.predict(x), trainer.learner.q_values(x), atol=1e-4)
    print("  connexion : modèle exporté identique à PyTorch, décodeur ε-greedy, version > annoncée")


def test_rejects_unexpected_actions() -> None:
    trainer = DQNUnityTrainer(small_config(), RunSettings())
    actions = [{"name": n, "type": t} for n, t in zip(ACTION_COLUMNS, ACTION_TYPES)]
    actions[0]["name"] = "move_forward"
    try:
        trainer.on_connect(FakeConnection(make_hello(actions=actions)))
    except ValueError as error:
        assert "Actions Unity inattendues" in str(error)
    else:
        raise AssertionError("un schéma d'actions différent doit être refusé")
    print("  connexion : schéma d'actions inattendu refusé")


def test_training_batches() -> None:
    trainer = DQNUnityTrainer(small_config(), RunSettings())
    conn = FakeConnection(make_hello())
    trainer.on_connect(conn)
    rng = np.random.default_rng(0)

    # Batch 1 : 16 transitions jouées avant notre modèle (action_index = -1), 2 hors table.
    batch = random_batch(rng, 80, version=0, invalid=16)
    batch.action_index[16:18] = 99
    trainer.on_batch(conn, batch)
    learner = trainer.learner
    assert learner.env_steps == 62 and len(learner.replay_buffer) == 62
    assert trainer.ignored_transitions == 18
    assert learner.gradient_steps == 0, "pas d'entraînement avant learning_starts"
    assert len(conn.models) == 2, "le réseau (et ε) est renvoyé après chaque batch"

    # Batch 2 : 80 transitions valides -> tampon à 142 >= 100 -> 80 / train_freq(4) = 20 pas.
    trainer.on_batch(conn, random_batch(rng, 80, version=conn.models[-1].version))
    assert learner.gradient_steps == 20
    assert conn.models[-1].decoder.spec["epsilon"] < conn.models[0].decoder.spec["epsilon"]
    assert conn.models[-1].version == conn.models[-2].version + 1
    print("  entraînement : filtrage des actions hors table, 1 pas / train_freq transitions, ε renvoyé")


def test_learns_contextual_bandit_end_to_end() -> None:
    """Le bon choix dépend de l'observation : action 3 si o0 > 0, sinon action 10.

    On vérifie sur le modèle exporté (celui qu'Unity exécuterait) que le DQN l'a appris.
    """
    config = small_config(lr=1e-3, train_freq=1, learning_starts=256, epsilon_decay_steps=1)
    trainer = DQNUnityTrainer(config, RunSettings())
    conn = FakeConnection(make_hello())
    trainer.on_connect(conn)
    rng = np.random.default_rng(0)

    def correct(obs):
        return np.where(obs[:, 0] > 0, 3, 10)

    for batch_id in range(20):
        obs = rng.standard_normal((256, OBS)).astype(np.float32)
        actions = rng.integers(0, 28, 256)  # exploration uniforme, comme ε = 1
        rewards = (actions == correct(obs)).astype(np.float32)
        trainer.on_batch(conn, make_batch(obs, actions, rewards, obs, np.ones(256, bool),
                                          version=conn.models[-1].version, batch_id=batch_id))

    exported = conn.models[-1].model
    test_obs = rng.standard_normal((600, OBS)).astype(np.float32)
    test_obs = test_obs[np.abs(test_obs[:, 0]) > 0.25]  # loin de la frontière o0 = 0, où l'erreur ne dit rien
    chosen = np.array([np.argmax(exported.predict(x)) for x in test_obs])
    accuracy = float(np.mean(chosen == correct(test_obs)))
    assert accuracy >= 0.9, f"précision {accuracy:.0%}"
    print(f"  apprentissage de bout en bout : {accuracy:.0%} de bons choix par le modèle exporté")


def test_checkpoint_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = RunSettings(action_table="foot", checkpoint_dir=Path(tmp))
        trainer = DQNUnityTrainer(small_config(), settings)
        conn = FakeConnection(make_hello())
        trainer.on_connect(conn)
        rng = np.random.default_rng(0)
        for _ in range(3):
            trainer.on_batch(conn, random_batch(rng, 80, version=conn.models[-1].version, num_actions=20))
        trainer.close()
        checkpoint = load_checkpoint(Path(tmp) / "latest.pt")

    # La reprise garde la table du checkpoint ('foot'), même si les réglages disent 'extended'.
    resumed = DQNUnityTrainer(small_config(lr=0.5), RunSettings(), checkpoint)
    conn2 = FakeConnection(make_hello(model_version=1))
    resumed.on_connect(conn2)
    assert resumed.table_name == "foot" and resumed.table.shape == (20, 7)
    assert resumed.config.lr == trainer.config.lr, "les hyperparamètres du checkpoint sont repris"
    assert resumed.learner.env_steps == trainer.learner.env_steps
    assert resumed.learner.gradient_steps == trainer.learner.gradient_steps
    for a, b in zip(resumed.learner.online_net.parameters(), trainer.learner.online_net.parameters()):
        assert torch.equal(a, b)
    assert conn2.models[-1].version > trainer.version

    mismatch = DQNUnityTrainer(small_config(), RunSettings(), checkpoint)
    try:
        mismatch.on_connect(FakeConnection(make_hello(obs_size=OBS + 6)))
    except ValueError as error:
        assert "observations" in str(error)
    else:
        raise AssertionError("une reprise avec d'autres observations doit être refusée")
    print("  checkpoint : reprise à l'identique, table conservée, observations différentes refusées")


def test_evaluation_counts_only_clean_episodes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        trainer = DQNUnityTrainer(small_config(), RunSettings(checkpoint_dir=Path(tmp)))
        trainer.on_connect(FakeConnection(make_hello()))
        save_checkpoint(Path(tmp) / "x.pt", trainer)
        checkpoint = load_checkpoint(Path(tmp) / "x.pt")

    finished = []
    evaluator = DQNUnityTrainer(small_config(), RunSettings(evaluate=True, evaluation_episodes=3),
                                checkpoint, on_finished=lambda: finished.append(True))
    conn = FakeConnection(make_hello())
    evaluator.on_connect(conn)
    sent = conn.models[-1]
    assert sent.decoder.spec["mode"] == "argmax", "pas d'exploration en évaluation"
    v = sent.version

    # Épisode 1 : commencé avec l'ancien modèle (v0) -> exclu. Épisode 4 : interrompu -> exclu.
    rng = np.random.default_rng(0)
    episodes_of = [1, 1, 2, 2, 3, 4]
    agents = [0, 0, 0, 0, 1, 1]
    batch = make_batch(rng.standard_normal((6, OBS)), [0, 1, 2, 3, 4, 5], np.zeros(6), rng.standard_normal((6, OBS)),
                       np.zeros(6, bool), version=v, agent_id=agents, episode_id=episodes_of,
                       episodes=[{"agent_id": 0, "episode_id": 1, "return": 1.0, "length": 2, "outcome": "goal_scored"},
                                 {"agent_id": 0, "episode_id": 2, "return": 0.9, "length": 2, "outcome": "goal_scored"},
                                 {"agent_id": 1, "episode_id": 3, "return": -1.0, "length": 1, "outcome": "goal_conceded"},
                                 {"agent_id": 1, "episode_id": 4, "return": 0.0, "length": 1, "outcome": "interrupted"}])
    batch.model_version[0] = 0
    evaluator.on_batch(conn, batch)
    assert not finished and len(evaluator.evaluation_episodes) == 2

    batch2 = make_batch(rng.standard_normal((1, OBS)), [0], [0.0], rng.standard_normal((1, OBS)), [False], version=v,
                        agent_id=[0], episode_id=[5],
                        episodes=[{"agent_id": 0, "episode_id": 5, "return": -0.2, "length": 600, "outcome": "timeout"}])
    evaluator.on_batch(conn, batch2)
    assert finished == [True]
    report = evaluator.evaluation_report()
    assert (report["goal_scored"], report["goal_conceded"], report["timeout"]) == (1, 1, 1)
    assert report["per_team"] == {"Blue": {"goal_scored": 1, "timeout": 1}, "Red": {"goal_conceded": 1}}
    assert report["excluded"] == {"autre modèle": 1, "interrompu": 1}
    assert len(evaluator.learner.replay_buffer) == 0, "rien n'est entraîné en évaluation"
    assert len(conn.models) == 1, "le réseau n'est pas renvoyé en évaluation"
    print("  évaluation : argmax, épisodes « propres » seulement, bilan par équipe")


def test_tensorboard_and_missing_next_obs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        trainer = DQNUnityTrainer(small_config(), RunSettings(log_dir=Path(tmp) / "run"))
        conn = FakeConnection(make_hello())
        trainer.on_connect(conn)
        rng = np.random.default_rng(0)
        batch = random_batch(rng, 200, version=1)
        batch.episodes = [{"agent_id": 0, "episode_id": 0, "return": 1.0, "length": 50, "outcome": "goal_scored"}]
        trainer.on_batch(conn, batch)
        no_next = random_batch(rng, 10, version=1)
        no_next.next_obs = None
        trainer.on_batch(conn, no_next)  # signalé, ignoré
        assert trainer.learner.env_steps == 200
        trainer.close()
        assert list((Path(tmp) / "run").glob("events.out.tfevents.*"))
    print("  TensorBoard écrit ; batch sans next_obs ignoré avec un message")


def test_replay_buffer_add_batch() -> None:
    buf = ReplayBuffer(capacity=10, observation_size=2, seed=0)

    def chunk(start, n):
        ids = np.arange(start, start + n)
        obs = np.stack([ids, ids], axis=1).astype(np.float32)
        return obs, ids, ids.astype(np.float32), obs + 0.5, np.zeros(n, bool)

    buf.add_batch(*chunk(0, 7))
    buf.add_batch(*chunk(7, 7))  # déborde : 0..3 écrasés
    assert len(buf) == 10
    assert sorted(buf._actions.tolist()) == list(range(4, 14))
    buf.add_batch(*chunk(100, 25))  # plus grand que le tampon : seules les 10 dernières restent
    assert sorted(buf._actions.tolist()) == list(range(115, 125))
    assert np.allclose(buf._next_observations[:, 0] - buf._observations[:, 0], 0.5)
    print("  tampon : add_batch avec débordement circulaire")


def test_action_tables() -> None:
    for name, rows in (("extended", 28), ("foot", 20)):
        table = build_action_table(name)
        assert table.shape == (rows, 7)
        assert np.all(np.abs(table) <= 1.0 + 1e-6)
        assert len({tuple(r) for r in table.round(4).tolist()}) == rows, "lignes en double"
    print("  tables d'actions : tailles, bornes, pas de doublon")


if __name__ == "__main__":
    import sys
    for stream in (sys.stdout, sys.stderr):  # console Windows : ε, τ...
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    print("Tests DQN Unity :")
    test_action_tables()
    test_replay_buffer_add_batch()
    test_connect_publishes_model_matching_torch()
    test_rejects_unexpected_actions()
    test_training_batches()
    test_learns_contextual_bandit_end_to_end()
    test_checkpoint_resume()
    test_evaluation_counts_only_clean_episodes()
    test_tensorboard_and_missing_next_obs()
    print("\nTous les tests DQN Unity sont passés !")
