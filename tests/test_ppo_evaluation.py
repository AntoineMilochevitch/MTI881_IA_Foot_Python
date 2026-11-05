"""Régressions du comptage sportif et du diagnostic à politique figée."""

from contextlib import redirect_stderr
from dataclasses import asdict
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from iafoot import Batch

from football_rl.ppo_checkpoint import build_checkpoint
from football_rl.ppo_network import PPOActorNetwork, PPOValueNetwork
from football_rl.ppo_trainer import PPOTrainerConfig, PPOUnityTrainer
from football_rl.ppo_unity import main
from football_rl.unity import ACTION_SCHEMA


OBSERVATION_NAMES = (
    "self_vel_x", "self_vel_z", "self_stunned", "self_tackling",
    "self_grounded", "ball_rel_x", "ball_rel_z",
) + tuple(f"observation_{index}" for index in range(39))


class PPOEvaluationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def make_trainer(self, target=10, diagnostic=False):
        actor = PPOActorNetwork(46)
        critic = PPOValueNetwork(46)
        checkpoint = build_checkpoint(
            actor, critic,
            torch.optim.Adam(actor.parameters()), torch.optim.Adam(critic.parameters()),
            progress={
                "version_counter": 3, "update_count": 7,
                "trained_transitions": 1234, "ignored_transitions": 0,
            },
            schema=(46, OBSERVATION_NAMES, ACTION_SCHEMA, 5),
            config=asdict(PPOTrainerConfig()),
        )
        trainer = PPOUnityTrainer(
            PPOTrainerConfig(evaluate=True, evaluation_episodes=target),
            resume_checkpoint=checkpoint,
            diagnostic_no_jump_tackle=diagnostic,
        )
        self.addCleanup(trainer.close)
        connection = SimpleNamespace(
            hello={"session": "evaluation-test"},
            behaviors=[{
                "name": "Foot", "obs_size": 46, "obs_names": OBSERVATION_NAMES,
                "action_size": 7,
                "actions": [{"name": name, "type": kind} for name, kind in ACTION_SCHEMA],
                "decision_period": 5, "model_version": 0,
            }],
            set_config=Mock(), send_model=Mock(),
        )
        trainer.on_connect(connection)
        trainer.on_model_ack(connection, {
            "behavior": "Foot", "version": trainer.pending_version, "ok": True,
        })
        trainer.tb_logger = Mock()
        return trainer, connection, checkpoint

    def make_batch(self, trainer, outcomes, batch_id=0, first_episode=0):
        # Deux transitions contiguës par épisode, avec les vraies log-probabilités.
        n = 2 * len(outcomes)
        observations = torch.randn(n, 46)
        with torch.no_grad():
            actions, log_probs = trainer.actor.sample_action(observations)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        rewards = np.zeros(n, dtype=np.float32)
        episodes = []
        for index, outcome in enumerate(outcomes):
            goal = outcome in {"goal_scored", "goal_conceded"}
            terminated[2 * index + 1] = goal
            truncated[2 * index + 1] = not goal
            reward = 1.0 if outcome == "goal_scored" else -1.0 if goal else 0.0
            rewards[2 * index + 1] = reward
            episodes.append({
                "agent_id": 0, "episode_id": first_episode + index,
                "return": reward, "length": 2, "outcome": outcome,
            })
        return Batch(
            behavior="Foot", batch_id=batch_id,
            obs=observations.numpy(), actions=actions.numpy(), rewards=rewards,
            next_obs=observations.numpy().copy(),
            terminated=terminated, truncated=truncated, log_probs=log_probs.numpy(),
            action_index=np.full(n, -1, dtype=np.int32),
            agent_id=np.zeros(n, dtype=np.int32),
            episode_id=np.repeat(np.arange(first_episode, first_episode + len(outcomes), dtype=np.int32), 2),
            step=np.tile(np.array([0, 1], dtype=np.int32), len(outcomes)),
            model_version=np.full(n, trainer.active_version, dtype=np.int32),
            episodes=episodes, header={},
        )

    def test_target_progress_and_tensorboard_count_only_sporting_episodes(self):
        trainer, connection, checkpoint = self.make_trainer(target=10)
        first = self.make_batch(
            trainer, ["timeout"] * 3 + ["interrupted"] * 3 + ["match_over"] * 4,
        )
        with patch("football_rl.ppo_trainer.train_ppo") as train, \
                patch("football_rl.ppo_trainer.save_checkpoint") as save:
            with self.assertLogs("football_rl.ppo_trainer", level="INFO") as logs:
                trainer.on_batch(connection, first)
            self.assertFalse(trainer.evaluation_finished)
            self.assertEqual(trainer.phase, "collecting")
            self.assertEqual(trainer.evaluation_admissible_count, 10)
            self.assertEqual(len(trainer.evaluation_episodes), 3)
            self.assertEqual(dict(trainer.evaluation_other_outcomes), {
                "interrupted": 3, "match_over": 4,
            })
            self.assertTrue(any("3/10 épisodes sportifs" in line for line in logs.output))
            self.assertEqual(trainer.tb_logger.log_episodes.call_args.args[0], 3)
            trainer.tb_logger.log_evaluation_interruptions.assert_called_with(3, 0)

            # Ce lot dépasse la cible : seuls les sept premiers résultats sont retenus.
            second = self.make_batch(
                trainer, ["goal_scored", "goal_conceded"] + ["timeout"] * 6,
                batch_id=1, first_episode=10,
            )
            with self.assertLogs("football_rl.ppo_trainer", level="INFO") as logs:
                trainer.on_batch(connection, second)
            self.assertTrue(trainer.evaluation_finished)
            self.assertEqual(trainer.phase, "evaluation_finished")
            self.assertEqual(len(trainer.evaluation_episodes), 10)
            self.assertEqual(trainer.evaluation_admissible_count, 17)
            self.assertTrue(any("10 épisodes sportifs sur 17" in line for line in logs.output))
            self.assertEqual(trainer.tb_logger.log_episodes.call_args.args[0], 10)
            self.assertEqual(len(trainer.tb_logger.log_episodes.call_args.args[1]), 10)

            trainer.on_batch(connection, second)
            self.assertEqual(len(trainer.evaluation_episodes), 10)
            self.assertEqual(trainer.tb_logger.log_episodes.call_count, 2)
            trainer._save_checkpoint()
            train.assert_not_called()
            save.assert_not_called()

        self.assertEqual((trainer.update_count, trainer.trained_transitions), (7, 1234))
        for network, key in ((trainer.actor, "actor_state_dict"), (trainer.critic, "critic_state_dict")):
            for name, value in network.state_dict().items():
                torch.testing.assert_close(value, checkpoint[key][name], rtol=0, atol=0)
        self.assertEqual(trainer.actor_optimizer.state_dict(), checkpoint["actor_optimizer_state_dict"])
        self.assertEqual(trainer.critic_optimizer.state_dict(), checkpoint["critic_optimizer_state_dict"])

    def test_excluded_outcomes_and_mixed_policy_do_not_advance_target(self):
        trainer, connection, _ = self.make_trainer(target=2)
        batch = self.make_batch(trainer, ["disabled", "interrupted", "goal_scored"])
        batch.model_version[4] = 0  # Le dernier épisode a commencé avant le modèle évalué.
        with self.assertLogs("football_rl.ppo_trainer", level="INFO") as logs:
            trainer.on_batch(connection, batch)
        self.assertEqual(trainer.evaluation_episodes, [])
        self.assertFalse(trainer.evaluation_finished)
        self.assertEqual(trainer.evaluation_admissible_count, 2)
        self.assertEqual(trainer.evaluation_technical_interruptions, 1)
        self.assertEqual(dict(trainer.evaluation_other_outcomes), {"disabled": 1, "interrupted": 1})
        trainer.tb_logger.log_episodes.assert_not_called()
        self.assertTrue(any("0/2 épisodes sportifs" in line for line in logs.output))

    def test_diagnostic_restores_then_exports_only_jump_and_tackle_changes(self):
        trainer, connection, checkpoint = self.make_trainer(diagnostic=True)
        original = PPOActorNetwork(46)
        original.load_state_dict(checkpoint["actor_state_dict"])
        observations = torch.randn(8, 46)
        with torch.no_grad():
            reference = original(observations)
            actual = trainer.actor(observations)
        torch.testing.assert_close(
            actual["model_output"][:, :5], reference["model_output"][:, :5], rtol=0, atol=0,
        )
        torch.testing.assert_close(actual["continuous_log_std"], reference["continuous_log_std"], rtol=0, atol=0)
        self.assertTrue((torch.sigmoid(actual["binary_logits"][:, 1:]) < 1e-8).all())
        self.assertTrue((original.network[-1].weight[5:7] != 0).any())

        exported = connection.send_model.call_args.args[0]
        predictions = np.stack([exported.predict(obs.numpy()) for obs in observations])
        torch.testing.assert_close(torch.from_numpy(predictions), actual["model_output"], rtol=1e-5, atol=1e-5)
        self.assertFalse(connection.send_model.call_args.kwargs["decoder"].spec["deterministic"])

    def test_cli_rejects_ablation_without_evaluation(self):
        with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised:
            main(["--diagnostic-no-jump-tackle"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--diagnostic-no-jump-tackle exige --evaluate", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
