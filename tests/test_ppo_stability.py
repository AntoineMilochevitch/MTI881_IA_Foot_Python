"""Régressions sur la reprise Adam, l'arrêt KL et le contrôle du pont."""

from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from football_rl.ppo_checkpoint import build_checkpoint
from football_rl.ppo_logging import PPOTensorBoardLogger
from football_rl.ppo_network import PPOActorNetwork, PPOValueNetwork
from football_rl.ppo_trainer import PPOTrainerConfig, PPOUnityTrainer
from football_rl.ppo_training import train_ppo_epochs, update_minibatch
from football_rl.ppo_unity import main
from football_rl.unity import ACTION_SCHEMA


class PPOStabilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.actor = PPOActorNetwork(3)
        self.critic = PPOValueNetwork(3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        obs = torch.randn(32, 3)
        with torch.no_grad():
            actions, log_probs = self.actor.sample_action(obs)
        self.data = dict(obs=obs, actions=actions, old_log_probs=log_probs,
                         advantages=torch.randn(32), returns=torch.randn(32))

    def checkpoint(self):
        return build_checkpoint(
            self.actor, self.critic, self.actor_optimizer, self.critic_optimizer,
            dict(version_counter=3, update_count=2, trained_transitions=64, ignored_transitions=0),
            (3, ("ball_rel_x", "ball_rel_z", "self_can_kick"), ACTION_SCHEMA, 5),
            asdict(PPOTrainerConfig(actor_lr=3e-4, rollout_size=512)),
        )

    def test_cli_preserves_saved_settings_unless_explicit(self):
        checkpoint = self.checkpoint()
        cases = [([], 3e-4, 512, 128, 1e-4),
                 (["--actor-lr", "0.0001", "--rollout-size", "2048",
                   "--mini-batch-size", "256", "--parity-atol", "0.0005"],
                  1e-4, 2048, 256, 5e-4)]
        for options, lr, rollout, minibatch, tolerance in cases:
            with self.subTest(options=options), \
                 patch("football_rl.ppo_unity.load_checkpoint", return_value=checkpoint), \
                 patch("football_rl.ppo_unity.TrainingServer"), \
                 patch("football_rl.ppo_unity.PPOUnityTrainer") as trainer:
                main(["--resume", "unused.pt", "--no-tensorboard", *options])
                config = trainer.call_args.args[0]
                self.assertEqual((config.actor_lr, config.rollout_size,
                                  config.mini_batch_size, config.parity_atol),
                                 (lr, rollout, minibatch, tolerance))

    def test_fresh_cli_uses_defaults(self):
        with patch("football_rl.ppo_unity.TrainingServer"), \
             patch("football_rl.ppo_unity.PPOUnityTrainer") as trainer:
            main(["--no-tensorboard"])
            self.assertEqual(trainer.call_args.args[0], PPOTrainerConfig(log_dir=None))

    def test_resume_overrides_lr_and_preserves_adam_history(self):
        update_minibatch(self.actor, self.critic, self.actor_optimizer,
                         self.critic_optimizer, self.data, 0.2, 0.5)
        checkpoint = self.checkpoint()
        schema = checkpoint["schema"]
        behavior = dict(name="Foot", obs_size=3, obs_names=schema[1],
                        actions=[dict(name=name, type=kind) for name, kind in ACTION_SCHEMA],
                        action_size=7, decision_period=5, model_version=0)
        connection = SimpleNamespace(hello={"session": "test"}, behaviors=[behavior],
                                     set_config=Mock(), send_model=Mock())
        trainer = PPOUnityTrainer(PPOTrainerConfig(actor_lr=1e-4), resume_checkpoint=checkpoint)
        trainer.on_connect(connection)
        self.assertEqual(trainer.actor_optimizer.param_groups[0]["lr"], 1e-4)
        restored = trainer.actor_optimizer.state_dict()["state"]
        for index, saved in checkpoint["actor_optimizer_state_dict"]["state"].items():
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(restored[index][name], saved[name], rtol=0, atol=0)
        self.assertEqual(trainer.trained_transitions, 64)
        self.assertEqual(trainer.phase, "waiting_for_ack")
        connection.send_model.assert_called_once()

    def test_kl_stops_inside_first_epoch(self):
        history = train_ppo_epochs(self.actor, self.critic, self.actor_optimizer,
                                   self.critic_optimizer, self.data, 0.2, 0.5,
                                   num_epochs=4, mini_batch_size=8, target_kl=1e-12)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["num_updates"], 1)
        self.assertTrue(history[0]["kl_limit_reached"])
        self.assertEqual(history[0]["num_samples"], 8)

    def test_without_kl_all_minibatches_are_trained(self):
        history = train_ppo_epochs(self.actor, self.critic, self.actor_optimizer,
                                   self.critic_optimizer, self.data, 0.2, 0.5,
                                   num_epochs=2, mini_batch_size=8)
        self.assertEqual([entry["num_updates"] for entry in history], [4, 4])
        self.assertFalse(any(entry["kl_limit_reached"] for entry in history))

    def test_parity_tolerance_warns_and_still_rejects_larger_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = PPOUnityTrainer(PPOTrainerConfig(parity_atol=5e-4), directory)
            trainer.actor = self.actor
            chunk = dict(obs=self.data["obs"], next_obs=self.data["obs"],
                         actions=self.data["actions"], rewards=torch.zeros(32),
                         old_log_probs=self.data["old_log_probs"] + 0.000138283)
            with self.assertLogs("football_rl.ppo_trainer", level="WARNING"):
                trainer._check_parity(chunk)
            chunk["old_log_probs"] = self.data["old_log_probs"] + 0.001
            with self.assertRaisesRegex(ValueError, "Incohérence"):
                trainer._check_parity(chunk)
            saved = torch.load(Path(directory) / "parity_failure.pt", weights_only=True)
            torch.testing.assert_close(saved["chunk"]["actions"], chunk["actions"])
            self.assertIn("actor_state_dict", saved)

    def test_sporting_rates_exclude_other_outcomes_and_write_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = PPOTensorBoardLogger(directory)
            try:
                with patch.object(logger, "_write") as write:
                    logger.log_episodes(1, [dict(outcome=name, length=10, **{"return": 0})
                                           for name in ("goal_scored", "timeout", "interrupted", "match_over")])
                    scalars = write.call_args.args[1]
                    self.assertEqual(scalars["sport/goal_scored_fraction"], 0.5)
                    self.assertEqual(scalars["episodes/outcome_goal_scored"], 0.25)
                    logger.log_episodes(2, [dict(outcome="timeout", length=10, **{"return": 0})])
                    self.assertEqual(write.call_args.args[1]["episodes/outcome_match_over"], 0)
            finally:
                logger.close()


if __name__ == "__main__":
    unittest.main()
