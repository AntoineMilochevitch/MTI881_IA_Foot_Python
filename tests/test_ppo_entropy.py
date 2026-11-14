"""Vérifie le signe du bonus, sa portée et la reprise d'anciens checkpoints."""

import copy
import math
import tempfile
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from football_rl.ppo_network import PPOActorNetwork, PPOValueNetwork
from football_rl.ppo_trainer import PPOTrainerConfig, PPOUnityTrainer
from football_rl.ppo_training import train_ppo_epochs
from football_rl.ppo_unity import main


class PPOEntropyTest(unittest.TestCase):
    def test_entropy_matches_known_distributions_and_shapes(self):
        actor = PPOActorNetwork(3)
        with torch.no_grad():
            for parameter in actor.network.parameters():
                parameter.zero_()
        for size in (1, 8):
            entropy = actor.evaluate_entropy(torch.zeros(size, 3))
            self.assertEqual(entropy['total'].shape, (size,))
            torch.testing.assert_close(entropy['continuous'], torch.full(
                (size,), 4 * (0.5 * math.log(2 * math.pi * math.e) + math.log(0.5))))
            torch.testing.assert_close(entropy['binary'], torch.full((size,), 3 * math.log(2)))
            torch.testing.assert_close(entropy['total'], entropy['continuous'] + entropy['binary'])

    def test_bonus_increases_exploration_without_changing_critic(self):
        torch.manual_seed(19)
        actor = PPOActorNetwork(3)
        critic = PPOValueNetwork(3)
        # Une probabilité initiale > 0,5 permet de vérifier le sens du gradient.
        with torch.no_grad():
            for parameter in actor.network.parameters():
                parameter.zero_()
            actor.network[-1].bias[actor.BINARY_INDICES] = 2.0
        reference_actor, reference_critic = copy.deepcopy(actor), copy.deepcopy(critic)
        obs = torch.zeros(8, 3)
        with torch.no_grad():
            actions, log_probs = actor.sample_action(obs)
        data = dict(obs=obs, actions=actions, old_log_probs=log_probs,
                    advantages=torch.zeros(8), returns=torch.ones(8))
        histories = []
        for policy, value, coefficient in ((reference_actor, reference_critic, 0.0),
                                            (actor, critic, 0.001)):
            histories.append(train_ppo_epochs(
                policy, value, torch.optim.SGD(policy.parameters(), lr=0.1),
                torch.optim.SGD(value.parameters(), lr=0.1), data, 0.2, 0.5,
                num_epochs=1, mini_batch_size=8, entropy_coef=coefficient))
        initial = reference_actor.evaluate_entropy(obs)
        updated = actor.evaluate_entropy(obs)
        self.assertTrue(torch.all(updated['continuous'] > initial['continuous']))
        self.assertTrue(torch.all(updated['binary'] > initial['binary']))
        for before, after in zip(reference_critic.parameters(), critic.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(histories[0][0]['actor_loss'], 0.0)
        self.assertGreater(histories[1][0]['entropy_bonus'], 0.0)
        self.assertAlmostEqual(histories[1][0]['actor_loss'],
                               -histories[1][0]['entropy_bonus'])

    def test_cli_old_checkpoint_and_explicit_override(self):
        # La CLI transmet ensuite le checkpoint au trainer : seul le décodage
        # de la configuration est exercé ici, sans serveur ni chargement réseau.
        for saved, options, expected in (({}, [], 0.0),
                                         ({'entropy_coef': 0.002}, [], 0.002),
                                         ({'entropy_coef': 0.002}, ['--entropy-coef', '0'], 0.0),
                                         ({}, ['--entropy-coef', '0.001'], 0.001)):
            with self.subTest(saved=saved, options=options), \
                 patch('football_rl.ppo_unity.load_checkpoint', return_value={'config': saved}), \
                 patch('football_rl.ppo_unity.TrainingServer'), \
                 patch('football_rl.ppo_unity.PPOUnityTrainer') as trainer:
                main(['--resume', 'unused.pt', '--no-tensorboard', *options])
                self.assertEqual(trainer.call_args.args[0].entropy_coef, expected)

    def test_invalid_coefficient_is_rejected(self):
        for value in (-0.1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PPOTrainerConfig(entropy_coef=value)

    def test_trainer_writes_entropy_for_uneven_minibatches(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                trainer = PPOUnityTrainer(PPOTrainerConfig(
                    device=device, log_dir=directory, entropy_coef=0.001, mini_batch_size=4))
                trainer.actor = PPOActorNetwork(3).to(device)
                critic = PPOValueNetwork(3).to(device)
                obs = torch.randn(9, 3, device=device)
                with torch.no_grad():
                    actions, log_probs = trainer.actor.sample_action(obs)
                trainer.last_history = train_ppo_epochs(
                    trainer.actor, critic,
                    torch.optim.Adam(trainer.actor.parameters(), lr=1e-4),
                    torch.optim.Adam(critic.parameters(), lr=1e-4),
                    dict(obs=obs, actions=actions, old_log_probs=log_probs,
                         advantages=torch.randn(9, device=device), returns=torch.ones(9, device=device)),
                    0.2, 0.5, num_epochs=1, mini_batch_size=4, entropy_coef=0.001)
                trainer.buffer = SimpleNamespace(get_data=lambda: {'obs': obs.cpu()})
                trainer.trained_transitions = 9
                trainer._log_tensorboard(9, perf_counter())
                trainer.close()
                events = EventAccumulator(directory).Reload()
                expected = trainer.actor.evaluate_entropy(obs)
                for name, values in expected.items():
                    point = events.Scalars('exploration/entropy_' + name)[-1]
                    self.assertEqual(point.step, 9)
                    self.assertAlmostEqual(point.value, values.mean().item(), places=5)
                self.assertAlmostEqual(events.Scalars('exploration/entropy_coef')[-1].value, 0.001)
                loss = events.Scalars('loss/actor')[-1].value
                policy = events.Scalars('loss/actor_policy_loss')[-1].value
                bonus = events.Scalars('loss/entropy_bonus')[-1].value
                self.assertAlmostEqual(loss, policy - bonus, places=6)


if __name__ == '__main__':
    unittest.main()
