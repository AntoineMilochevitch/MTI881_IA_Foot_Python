"""Vérifications des réseaux PPO, avec le tacle toujours activé.

Depuis la racine, dans l'environnement virtuel du projet :
    python -m unittest discover -s tests -p "test_ppo_network.py" -v

Ou directement :
    python tests/test_ppo_network.py

Ces tests vérifient des calculs et leurs gradients, sans entraîner les
réseaux ni lancer Unity. Les observations synthétiques ne sont pas des
états de match. Chaque test utilise des réseaux neufs et une graine fixe.
"""

import math
import unittest

import torch
from torch import nn

from football_rl.ppo_network import PPOActorNetwork, PPOValueNetwork
from football_rl.types import ObservationLayout


class TestPPONetworks(unittest.TestCase):
    """Contrats de formes, distributions et réévaluation pour PPO."""

    def setUp(self) -> None:
        # Restaurer le générateur après chaque test évite d'affecter les autres.
        rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, rng_state)
        torch.manual_seed(881)

        self.observation_size = ObservationLayout(2, 3).size
        self.actor = PPOActorNetwork(self.observation_size)
        self.critic = PPOValueNetwork(self.observation_size)

    def observations(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.observation_size)

    def assert_finite(self, tensor: torch.Tensor) -> None:
        self.assertTrue(bool(torch.isfinite(tensor).all()), "Valeur non finie")

    def assert_binary(self, tensor: torch.Tensor) -> None:
        self.assert_finite(tensor)
        self.assertTrue(bool(((tensor == 0) | (tensor == 1)).all()))

    def assert_bounded(self, tensor: torch.Tensor, low: float, high: float) -> None:
        self.assert_finite(tensor)
        # Les deux bornes doivent être respectées simultanément : ET, pas OU.
        self.assertTrue(bool(((tensor >= low) & (tensor <= high)).all()))

    def assert_gradients(self, module: nn.Module) -> None:
        for name, parameter in module.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad, "Gradient absent")
                self.assert_finite(parameter.grad)

    @staticmethod
    def constant_head(head: nn.Linear, value: float) -> None:
        """Fixe une tête pour vérifier une distribution connue analytiquement."""
        with torch.no_grad():
            head.weight.zero_()
            head.bias.fill_(value)

    def evaluate_recorded_action(
        self, observations: torch.Tensor, action: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.actor.evaluate_action(
            observations,
            raw_move=action["raw_move"],
            jump=action["jump"],
            tackle=action["tackle"],
            trigger=action["trigger"],
            intensity=action["intensity"],
        )

    # --- Critique et sorties brutes de l'acteur ---

    def test_critic_shapes_finite_values_and_gradients(self) -> None:
        """Le critique produit (B,) et reçoit des gradients, même pour B=1."""
        for batch_size in (1, 8):
            with self.subTest(batch_size=batch_size):
                self.critic.zero_grad(set_to_none=True)
                value = self.critic(self.observations(batch_size))["value"]
                self.assertEqual(value.shape, (batch_size,))
                self.assert_finite(value)
                value.square().mean().backward()
                self.assert_gradients(self.critic)

    def test_critic_output_is_not_bounded_to_probabilities(self) -> None:
        """Une valeur peut être négative ou supérieure à un."""
        observations = self.observations(8)
        for expected in (-2.0, 3.0):
            with self.subTest(value=expected):
                self.constant_head(self.critic.network[-1], expected)
                actual = self.critic(observations)["value"]
                torch.testing.assert_close(actual, torch.full((8,), expected))

    def test_actor_forward_shapes(self) -> None:
        """Les paramètres des distributions ont les dimensions convenues."""
        widths = {
            "mean": 3,
            "log_std": 3,
            "jump_logit": 1,
            "tackle_logit": 1,
            "shoot_trigger_logit": 1,
            "shoot_alpha_raw": 1,
            "shoot_beta_raw": 1,
        }
        for batch_size in (1, 8):
            outputs = self.actor(self.observations(batch_size))
            for name, width in widths.items():
                with self.subTest(batch_size=batch_size, output=name):
                    self.assertEqual(outputs[name].shape, (batch_size, width))
                    self.assert_finite(outputs[name])

    # --- Déplacement : gaussienne suivie de tanh ---

    def test_move_shapes_bounds_and_reevaluation(self) -> None:
        """Le déplacement respecte ses bornes et conserve sa log-densité."""
        for batch_size in (1, 8):
            with self.subTest(batch_size=batch_size):
                observations = self.observations(batch_size)
                with torch.no_grad():
                    sampled = self.actor.sample_move(observations)
                self.assertEqual(sampled["raw_move"].shape, (batch_size, 3))
                self.assertEqual(sampled["move_target"].shape, (batch_size, 3))
                self.assertEqual(sampled["log_prob"].shape, (batch_size,))
                self.assert_finite(sampled["raw_move"])
                self.assert_bounded(sampled["move_target"], -1.0, 1.0)
                torch.testing.assert_close(
                    sampled["move_target"], torch.tanh(sampled["raw_move"])
                )
                actual = self.actor.evaluate_move(observations, sampled["raw_move"])
                self.assertEqual(actual.shape, (batch_size,))
                self.assert_finite(actual)
                torch.testing.assert_close(actual, sampled["log_prob"])

    def test_move_sampling_explores_for_same_observation(self) -> None:
        """Les paramètres restent fixes tandis que les tirages changent."""
        observations = self.observations(1)
        with torch.no_grad():
            parameters_before = self.actor(observations)
            first = self.actor.sample_move(observations)
            second = self.actor.sample_move(observations)
            parameters_after = self.actor(observations)
        for name in ("mean", "log_std"):
            torch.testing.assert_close(parameters_before[name], parameters_after[name])
        self.assertFalse(torch.equal(first["raw_move"], second["raw_move"]))
        self.assertFalse(torch.equal(first["move_target"], second["move_target"]))

    def test_move_standard_normal_at_zero(self) -> None:
        """Repère indépendant : trois N(0,1) en zéro donnent -2,7568 environ."""
        self.constant_head(self.actor.mean, 0.0)
        self.constant_head(self.actor.log_std_head, 0.0)
        actual = self.actor.evaluate_move(self.observations(1), torch.zeros(1, 3))
        expected = torch.tensor([-1.5 * math.log(2.0 * math.pi)])
        torch.testing.assert_close(actual, expected)

    def test_move_density_away_from_zero(self) -> None:
        """La correction tanh est comparée à une formule indépendante hors zéro."""
        self.constant_head(self.actor.mean, 0.0)
        self.constant_head(self.actor.log_std_head, 0.0)
        raw_move = torch.tensor([[0.2, -0.3, 0.8]])
        # log(1 - tanh(u)^2) = -2 log(cosh(u)). Les valeurs ici sont modérées.
        expected = sum(
            -0.5 * (value * value + math.log(2.0 * math.pi))
            + 2.0 * math.log(math.cosh(value))
            for value in raw_move[0].tolist()
        )
        actual = self.actor.evaluate_move(self.observations(1), raw_move)
        torch.testing.assert_close(actual, torch.tensor([expected]))

    def test_move_log_std_clamping(self) -> None:
        """Les bornes de log_std sont appliquées avant le calcul de densité."""
        self.constant_head(self.actor.mean, 0.0)
        for raw_log_std, bounded_log_std in ((-10.0, -5.0), (10.0, 2.0)):
            with self.subTest(raw_log_std=raw_log_std):
                self.constant_head(self.actor.log_std_head, raw_log_std)
                actual = self.actor.evaluate_move(
                    self.observations(1), torch.zeros(1, 3)
                )
                expected = -3.0 * (0.5 * math.log(2.0 * math.pi) + bounded_log_std)
                torch.testing.assert_close(actual, torch.tensor([expected]))

    def test_move_saturation_keeps_log_prob_and_gradients_finite(self) -> None:
        """Des valeurs brutes de +/-20 ne provoquent pas log(0) après tanh."""
        raw_move = torch.tensor([[-20.0, 0.0, 20.0], [20.0, -20.0, 0.0]])
        actual = self.actor.evaluate_move(self.observations(2), raw_move)
        self.assertEqual(actual.shape, (2,))
        self.assert_finite(actual)
        (-actual.mean()).backward()
        self.assert_gradients(self.actor.mean)
        self.assert_gradients(self.actor.log_std_head)
        self.assert_gradients(self.actor.network)

    # --- Commandes binaires : saut et tacle ---

    def test_jump_and_tackle_shapes_values_and_reevaluation(self) -> None:
        """Chaque Bernoulli retourne des décisions binaires et des logs (B,)."""
        for name in ("jump", "tackle"):
            sample = getattr(self.actor, f"sample_{name}")
            evaluate = getattr(self.actor, f"evaluate_{name}")
            for batch_size in (1, 8):
                with self.subTest(command=name, batch_size=batch_size):
                    observations = self.observations(batch_size)
                    with torch.no_grad():
                        sampled = sample(observations)
                    target = sampled[f"{name}_target"]
                    self.assertEqual(target.shape, (batch_size,))
                    self.assert_binary(target)
                    self.assertEqual(sampled["log_prob"].shape, (batch_size,))
                    actual = evaluate(observations, target)
                    self.assertEqual(actual.shape, (batch_size,))
                    self.assert_finite(actual)
                    torch.testing.assert_close(actual, sampled["log_prob"])

    def test_jump_and_tackle_zero_logits_reference(self) -> None:
        """Un logit nul donne log(0,5), pour zéro comme pour un."""
        observations = self.observations(2)
        decisions = torch.tensor([0.0, 1.0])
        for name in ("jump", "tackle"):
            with self.subTest(command=name):
                self.constant_head(getattr(self.actor, f"{name}_head"), 0.0)
                evaluate = getattr(self.actor, f"evaluate_{name}")
                actual = evaluate(observations, decisions)
                torch.testing.assert_close(actual, torch.full((2,), math.log(0.5)))

    # --- Frappe : déclenchement Bernoulli et intensité Beta ---

    def test_shoot_shapes_bounds_and_reevaluation(self) -> None:
        """La commande est bornée et les données Beta sont conservées séparément."""
        for batch_size in (1, 8):
            with self.subTest(batch_size=batch_size):
                observations = self.observations(batch_size)
                with torch.no_grad():
                    sampled = self.actor.sample_shoot(observations)
                for name in ("trigger", "intensity", "shoot_target", "log_prob"):
                    self.assertEqual(sampled[name].shape, (batch_size,))
                    self.assert_finite(sampled[name])
                self.assert_binary(sampled["trigger"])
                self.assert_bounded(sampled["shoot_target"], 0.0, 1.0)
                self.assertTrue(bool(
                    ((sampled["intensity"] > 0) & (sampled["intensity"] < 1)).all()
                ))
                torch.testing.assert_close(
                    sampled["shoot_target"], sampled["trigger"] * sampled["intensity"]
                )
                actual = self.actor.evaluate_shoot(
                    observations, sampled["trigger"], sampled["intensity"]
                )
                self.assertEqual(actual.shape, (batch_size,))
                self.assert_finite(actual)
                torch.testing.assert_close(actual, sampled["log_prob"])

    def test_shoot_trigger_controls_zero_and_positive_commands(self) -> None:
        """Vérifie les deux cas sans attendre un résultat aléatoire du lot."""
        observations = self.observations(8)
        for trigger_logit, expected_trigger in ((-100.0, 0.0), (100.0, 1.0)):
            with self.subTest(trigger=expected_trigger):
                # En float32, ces logits donnent des probabilités arrondies à 0/1.
                self.constant_head(self.actor.shoot_trigger_head, trigger_logit)
                with torch.no_grad():
                    sampled = self.actor.sample_shoot(observations)
                torch.testing.assert_close(
                    sampled["trigger"], torch.full((8,), expected_trigger)
                )
                expected_target = (
                    torch.zeros(8) if expected_trigger == 0.0 else sampled["intensity"]
                )
                torch.testing.assert_close(
                    sampled["shoot_target"], expected_target, rtol=0, atol=0
                )
                self.assert_finite(sampled["log_prob"])

    def test_shoot_beta_reference(self) -> None:
        """Avec p=0,5 et Beta(2,2), les valeurs de référence sont calculables."""
        self.constant_head(self.actor.shoot_trigger_head, 0.0)
        # softplus(raw) + 1 = 2 implique raw = log(exp(1) - 1).
        raw_concentration = math.log(math.expm1(1.0))
        self.constant_head(self.actor.shoot_alpha_head, raw_concentration)
        self.constant_head(self.actor.shoot_beta_head, raw_concentration)
        trigger = torch.tensor([0.0, 0.0, 1.0, 1.0])
        intensity = torch.tensor([0.25, 0.75, 0.5, 0.25])
        actual = self.actor.evaluate_shoot(self.observations(4), trigger, intensity)
        # Beta(2,2) a pour densité 6*x*(1-x).
        expected = torch.tensor([
            math.log(0.5), math.log(0.5), math.log(0.75), math.log(0.5625)
        ])
        torch.testing.assert_close(actual, expected)

    def test_shoot_inactive_ignores_intensity_and_its_gradients(self) -> None:
        """Sans frappe, l'intensité n'affecte ni le log ni ses propres têtes."""
        observations = self.observations(8)
        trigger = torch.zeros(8)
        first = self.actor.evaluate_shoot(observations, trigger, torch.full((8,), 0.2))
        second = self.actor.evaluate_shoot(observations, trigger, torch.full((8,), 0.8))
        self.assert_finite(first)
        torch.testing.assert_close(first, second)

        self.actor.zero_grad(set_to_none=True)
        (-first.mean()).backward()
        self.assert_gradients(self.actor.shoot_trigger_head)
        for name in ("shoot_alpha_head", "shoot_beta_head"):
            for parameter in getattr(self.actor, name).parameters():
                with self.subTest(head=name):
                    # Un chemin exclu peut donner un gradient absent ou nul.
                    if parameter.grad is not None:
                        self.assert_finite(parameter.grad)
                        self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    # --- Action complète : contrat, somme des logs, ratio et gradients ---

    def test_action_shapes_sum_reevaluation_and_initial_ratio(self) -> None:
        """Chaque action complète possède une seule log-probabilité, ratio initial 1."""
        for batch_size in (1, 8):
            with self.subTest(batch_size=batch_size):
                observations = self.observations(batch_size)
                with torch.no_grad():
                    sampled = self.actor.sample_action(observations)
                expected_shapes = {
                    "move_target": (batch_size, 3),
                    "jump": (batch_size,),
                    "tackle": (batch_size,),
                    "shoot": (batch_size, 1),
                    "raw_move": (batch_size, 3),
                    "trigger": (batch_size,),
                    "intensity": (batch_size,),
                    "log_prob": (batch_size,),
                }
                for name, shape in expected_shapes.items():
                    self.assertEqual(sampled[name].shape, shape, name)
                    self.assert_finite(sampled[name])
                    self.assertFalse(sampled[name].requires_grad, name)
                self.assert_bounded(sampled["move_target"], -1.0, 1.0)
                self.assert_bounded(sampled["shoot"], 0.0, 1.0)
                self.assert_binary(sampled["jump"])
                self.assert_binary(sampled["tackle"])
                self.assert_binary(sampled["trigger"])
                torch.testing.assert_close(
                    sampled["move_target"], torch.tanh(sampled["raw_move"])
                )
                torch.testing.assert_close(
                    sampled["shoot"],
                    (sampled["trigger"] * sampled["intensity"]).unsqueeze(-1),
                )

                # Réévaluer les mêmes commandes, sans en tirer de nouvelles.
                component_sum = (
                    self.actor.evaluate_move(observations, sampled["raw_move"])
                    + self.actor.evaluate_jump(observations, sampled["jump"])
                    + self.actor.evaluate_tackle(observations, sampled["tackle"])
                    + self.actor.evaluate_shoot(
                        observations, sampled["trigger"], sampled["intensity"]
                    )
                )
                actual = self.evaluate_recorded_action(observations, sampled)
                self.assertEqual(actual.shape, (batch_size,))
                self.assert_finite(actual)
                self.assertTrue(actual.requires_grad)
                torch.testing.assert_close(actual, component_sum)
                torch.testing.assert_close(actual, sampled["log_prob"])
                ratio = torch.exp(actual - sampled["log_prob"])
                torch.testing.assert_close(ratio, torch.ones(batch_size))

    def test_evaluate_action_neither_samples_nor_changes_recorded_data(self) -> None:
        """Réévaluer ne consomme pas d'aléatoire et préserve les données collectées."""
        observations = self.observations(8)
        with torch.no_grad():
            sampled = self.actor.sample_action(observations)
        saved = {name: value.clone() for name, value in sampled.items()}
        rng_before = torch.get_rng_state().clone()
        first = self.evaluate_recorded_action(observations, sampled)
        second = self.evaluate_recorded_action(observations, sampled)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        torch.testing.assert_close(first, second)
        for name, value in sampled.items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)

    def test_action_ratio_changes_with_current_policy_only(self) -> None:
        """Changer uniquement le saut modifie le ratio sans changer l'ancien log."""
        self.constant_head(self.actor.jump_head, 0.0)  # Ancienne probabilité : 0,5.
        observations = self.observations(8)
        with torch.no_grad():
            sampled = self.actor.sample_action(observations)
        old_log_prob = sampled["log_prob"].clone()
        self.constant_head(self.actor.jump_head, math.log(3.0))  # Nouvelle : 0,75.
        actual = self.evaluate_recorded_action(observations, sampled)
        ratio = torch.exp(actual - sampled["log_prob"])
        # 0,75/0,5 si saut ; 0,25/0,5 sinon. Les autres termes s'annulent.
        expected_ratio = torch.where(sampled["jump"] == 1.0, 1.5, 0.5)
        torch.testing.assert_close(ratio, expected_ratio)
        torch.testing.assert_close(sampled["log_prob"], old_log_prob, rtol=0, atol=0)
        self.assertFalse(sampled["log_prob"].requires_grad)

    def test_action_backward_reaches_all_heads_without_updating_weights(self) -> None:
        """La réévaluation transmet des gradients aux sept têtes et au tronc."""
        observations = self.observations(8)
        # Actions fixes valides, avec des frappes, pour exercer toutes les têtes.
        action = {
            "raw_move": torch.tensor([[0.3, -0.2, 0.4]]).repeat(8, 1),
            "jump": torch.ones(8),
            "tackle": torch.zeros(8),
            "trigger": torch.ones(8),
            "intensity": torch.full((8,), 0.35),
        }
        weights_before = {
            name: parameter.detach().clone()
            for name, parameter in self.actor.named_parameters()
        }
        self.actor.zero_grad(set_to_none=True)
        actual = self.evaluate_recorded_action(observations, action)
        self.assertTrue(actual.requires_grad)
        self.assert_finite(actual)
        # Perte de diagnostic uniquement ; il ne s'agit pas de la perte PPO.
        (-actual.mean()).backward()
        self.assert_gradients(self.actor)
        for name in (
            "mean", "log_std_head", "jump_head", "tackle_head",
            "shoot_trigger_head", "shoot_alpha_head", "shoot_beta_head",
        ):
            with self.subTest(head=name):
                magnitude = sum(
                    parameter.grad.abs().sum().item()
                    for parameter in getattr(self.actor, name).parameters()
                )
                self.assertGreater(magnitude, 0.0)
        for name, parameter in self.actor.named_parameters():
            torch.testing.assert_close(parameter, weights_before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.critic.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
