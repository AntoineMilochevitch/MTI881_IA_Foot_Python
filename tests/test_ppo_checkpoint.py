import tempfile
import unittest
from pathlib import Path

import torch

from football_rl.ppo_checkpoint import (
    build_checkpoint,
    load_checkpoint,
    restore_networks,
    restore_optimizers,
    save_checkpoint,
)
from football_rl.ppo_network import PPOActorNetwork, PPOValueNetwork


class PPOCheckpointTest(unittest.TestCase):
    def test_restores_critic_weights_and_adam_history(self):
        torch.manual_seed(0)
        actor = PPOActorNetwork(observation_size=46)
        critic = PPOValueNetwork(observation_size=46)
        actor_optimizer = torch.optim.Adam(actor.parameters())
        critic_optimizer = torch.optim.Adam(critic.parameters())

        observations = torch.randn(8, 46)
        predictions = critic(observations)["value"]
        targets = torch.ones_like(predictions)
        loss = torch.mean((predictions - targets) ** 2)
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()

        critic_parameter = next(critic.parameters())
        critic_state = critic_optimizer.state[critic_parameter]
        self.assertEqual(critic_state["step"], 1)
        self.assertEqual(critic_state["exp_avg"].shape, critic_parameter.shape)
        self.assertEqual(critic_state["exp_avg_sq"].shape, critic_parameter.shape)
        saved_parameter = critic_parameter.detach().clone()
        saved_exp_avg = critic_state["exp_avg"].detach().clone()
        saved_exp_avg_sq = critic_state["exp_avg_sq"].detach().clone()

        checkpoint = build_checkpoint(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            progress={
                "version_counter": 0,
                "update_count": 1,
                "trained_transitions": 8,
                "ignored_transitions": 0,
            },
            schema=(46,),
            config={"device": torch.device("cpu")},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
            save_checkpoint(checkpoint, str(checkpoint_path))
            loaded_checkpoint = load_checkpoint(str(checkpoint_path))

            restored_actor = PPOActorNetwork(observation_size=46)
            restored_critic = PPOValueNetwork(observation_size=46)
            restored_actor_optimizer = torch.optim.Adam(restored_actor.parameters())
            restored_critic_optimizer = torch.optim.Adam(restored_critic.parameters())
            restore_networks(
                loaded_checkpoint, restored_actor, restored_critic
            )
            restore_optimizers(
                loaded_checkpoint,
                restored_actor_optimizer,
                restored_critic_optimizer,
            )

        restored_parameter = next(restored_critic.parameters())
        restored_state = restored_critic_optimizer.state[restored_parameter]
        self.assertTrue(torch.allclose(saved_parameter, restored_parameter))
        self.assertEqual(restored_state["step"], 1)
        self.assertTrue(torch.allclose(saved_exp_avg, restored_state["exp_avg"]))
        self.assertTrue(
            torch.allclose(saved_exp_avg_sq, restored_state["exp_avg_sq"])
        )
        self.assertEqual(restored_actor_optimizer.state, {})


if __name__ == "__main__":
    unittest.main()
