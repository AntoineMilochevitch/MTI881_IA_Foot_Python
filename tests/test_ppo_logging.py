import tempfile
import unittest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from football_rl.ppo_logging import PPOTensorBoardLogger


def fake_history():
    base = {
        "actor_loss": 0.1, "critic_loss": 2.0,
        "actor_grad_norm": 0.3, "critic_grad_norm": 1.0,
        "num_samples": 2048, "num_updates": 16, "ratio_mean": 1.0,
    }
    return [
        {**base, "epoch": 1, "approx_kl": 0.004, "clip_fraction": 0.05, "kl_limit_reached": False},
        {**base, "epoch": 2, "approx_kl": 0.012, "clip_fraction": 0.10, "kl_limit_reached": True},
    ]


class PPOTensorBoardLoggerTest(unittest.TestCase):
    def test_log_cycle_uses_last_epoch(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tb = PPOTensorBoardLogger(tmp)
            tb.log_cycle(2048, fake_history(), {"time/optimization_s": 0.8})
            tb.close()

            acc = EventAccumulator(tmp)
            acc.Reload()
            kl = acc.Scalars("ppo/approx_kl")[0]
            self.assertEqual(kl.step, 2048)
            self.assertAlmostEqual(kl.value, 0.012, places=5)
            self.assertEqual(acc.Scalars("ppo/epochs_done")[0].value, 2.0)
            self.assertEqual(acc.Scalars("ppo/kl_limit_reached")[0].value, 1.0)
            self.assertIn("time/optimization_s", acc.Tags()["scalars"])

    def test_log_episodes_aggregates_outcomes(self):
        episodes = [
            {"agent_id": 0, "episode_id": 1, "return": 1.0, "length": 100, "outcome": "goal_scored"},
            {"agent_id": 1, "episode_id": 1, "return": -1.0, "length": 100, "outcome": "goal_conceded"},
            {"agent_id": 0, "episode_id": 2, "return": 0.0, "length": 300, "outcome": "timeout"},
            {"agent_id": 1, "episode_id": 2, "return": 0.0, "length": 300, "outcome": "timeout"},
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tb = PPOTensorBoardLogger(tmp)
            tb.log_episodes(4096, episodes)
            tb.close()

            acc = EventAccumulator(tmp)
            acc.Reload()
            self.assertEqual(acc.Scalars("episodes/count")[0].value, 4.0)
            self.assertAlmostEqual(acc.Scalars("episodes/return_mean")[0].value, 0.0, places=5)
            self.assertAlmostEqual(acc.Scalars("episodes/length_mean")[0].value, 200.0, places=3)
            self.assertAlmostEqual(acc.Scalars("episodes/outcome_timeout")[0].value, 0.5, places=5)
            self.assertAlmostEqual(acc.Scalars("episodes/outcome_interrupted")[0].value, 0.0, places=5)

    def test_empty_inputs_write_nothing(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tb = PPOTensorBoardLogger(tmp)
            tb.log_cycle(0, [])
            tb.log_episodes(0, [])
            tb.close()


if __name__ == "__main__":
    unittest.main()