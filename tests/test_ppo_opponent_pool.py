"""Routage multi-adversaires, confirmations et bilans séparés."""

from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from football_rl.ppo_logging import PPOTensorBoardLogger
from football_rl.ppo_opponent import FrozenOpponent, FrozenOpponentPool
from football_rl.ppo_unity import main


def opponent(name):
    # Vrai mécanisme d'accusé, export remplacé : les tests réseau ne nécessitent
    # pas de poids volumineux ni de serveur Unity.
    result = FrozenOpponent.__new__(FrozenOpponent)
    result.behavior = name
    result.schema = (1, ('x',), (('move', 'continuous'),), 5)
    result.version_counter = 0
    result.connection = None
    result.pending_version = None
    result.ready = False
    result.export = Mock(return_value=('model', 'decoder'))
    return result


def behavior(name):
    return dict(name=name, obs_size=1, obs_names=['x'], action_size=1,
                actions=[dict(name='move', type='continuous')], decision_period=5,
                model_version=0, agents=[dict(id=1, record_experience=False)])


class OpponentPoolTest(unittest.TestCase):
    def setUp(self):
        self.old, self.recent = opponent('Old'), opponent('Recent')
        self.trainer = Mock(behavior='Foot')
        self.pool = FrozenOpponentPool([self.old, self.recent], self.trainer)
        self.connection = SimpleNamespace(behaviors=[behavior(n) for n in ['Foot', 'Old', 'Recent']], send_model=Mock())

    def ack(self, name, version=1, ok=True):
        self.pool.on_model_ack(self.connection, dict(behavior=name, version=version, ok=ok))

    def test_waits_for_both_acks_and_never_trains_opponents(self):
        self.pool.on_connect(self.connection)
        self.pool.on_connect(self.connection)
        self.assertEqual(self.connection.send_model.call_count, 2)
        self.pool.on_batch(self.connection, SimpleNamespace(behavior='Foot'))
        self.trainer.on_batch.assert_not_called()
        self.ack('Recent')
        self.ack('Old', version=999)
        self.trainer.on_connect.assert_not_called()
        self.ack('Old')
        self.trainer.on_connect.assert_called_once_with(self.connection)
        self.ack('Old')
        self.trainer.on_connect.assert_called_once()
        self.ack('Foot', version=42)
        self.trainer.on_model_ack.assert_called_once()
        self.pool.on_batch(self.connection, SimpleNamespace(behavior='Old'))
        self.trainer.on_batch.assert_not_called()
        self.pool.on_batch(self.connection, SimpleNamespace(behavior='Foot'))
        self.trainer.on_batch.assert_called_once()

    def test_disconnect_republishes_all_snapshots(self):
        self.pool.on_connect(self.connection)
        self.ack('Old'); self.ack('Recent')
        self.pool.on_disconnect(self.connection)
        self.assertFalse(self.pool.ready)
        self.pool.on_connect(self.connection)
        self.assertEqual(self.connection.send_model.call_count, 4)
        self.assertEqual(self.old.pending_version, 2)

    def test_missing_or_bad_schema_never_starts_learner(self):
        self.connection.behaviors.pop()
        self.pool.on_connect(self.connection)
        self.connection.send_model.assert_not_called()
        invalid = behavior('Recent'); invalid['obs_size'] = 2
        self.connection.behaviors.append(invalid)
        with self.assertRaises(ValueError): self.pool.on_connect(self.connection)
        self.connection.send_model.assert_not_called()
        self.trainer.on_connect.assert_not_called()

    def test_recording_opponent_and_rejected_ack_are_rejected(self):
        self.connection.behaviors[-1]['agents'][0]['record_experience'] = True
        with self.assertRaises(ValueError): self.pool.on_connect(self.connection)
        self.connection.behaviors[-1]['agents'][0]['record_experience'] = False
        self.pool.on_connect(self.connection)
        with self.assertRaises(ValueError): self.ack('Old', ok=False)
        self.assertFalse(self.pool.ready)
        self.trainer.on_connect.assert_not_called()

    def test_empty_pool_preserves_solo_and_bot(self):
        pool = FrozenOpponentPool([], self.trainer)
        pool.on_connect(self.connection)
        self.trainer.on_connect.assert_called_once()
        self.assertTrue(pool.ready)

    def test_cli_loads_two_snapshots_and_keeps_legacy_option(self):
        for options, expected in [(['--opponent', 'Old', 'old.pt', '--opponent', 'Recent', 'recent.pt'], 2),
                                  (['--opponent-checkpoint', 'old.pt'], 1)]:
            with patch('football_rl.ppo_unity.FrozenOpponent') as factory, \
                 patch('football_rl.ppo_unity.FrozenOpponentPool'), \
                 patch('football_rl.ppo_unity.TrainingServer'), \
                 patch('football_rl.ppo_unity.PPOUnityTrainer'):
                main(['--no-tensorboard', *options])
                self.assertEqual(factory.call_count, expected)

    def test_duplicate_cli_names_are_rejected(self):
        with self.assertRaises(SystemExit), patch('football_rl.ppo_unity.FrozenOpponent') as factory:
            main(['--no-tensorboard', '--opponent', 'Old', 'a.pt', '--opponent', 'Old', 'b.pt'])
        factory.assert_not_called()

    def test_metrics_use_each_opponents_own_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = PPOTensorBoardLogger(directory)
            with patch.object(logger, '_write') as write:
                logger.log_opponent_episodes(100, [
                    dict(opponent_group=g, outcome=o, length=10, **{'return': 0})
                    for g,o in [('bot','goal_scored'),('bot','goal_conceded'),
                                ('bot','match_over'),('snapshot_old','timeout')]])
                scalars = {k:v for call in write.call_args_list for k,v in call.args[1].items()}
                self.assertEqual(scalars['opponents/bot/sport/count'], 2)
                self.assertEqual(scalars['opponents/bot/sport/goal_scored_fraction'], 0.5)
                self.assertEqual(scalars['opponents/snapshot_old/sport/timeout_fraction'], 1)
                self.assertNotIn('opponents/snapshot_recent/sport/count', scalars)
            logger.close()
