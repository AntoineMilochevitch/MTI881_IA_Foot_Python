"""Collecte Unity avec l'acteur PPO initial, sans optimisation des poids.

Le serveur et les échanges restent fournis par football_rl.unity et iafoot.
Ce module ne contient que la construction de la politique propre à PPO.
"""

from typing import Any

import torch
from iafoot import DecoderExport, ModelExport

from football_rl.ppo_export import export_actor
from football_rl.ppo_network import PPOActorNetwork
from football_rl.unity import main as run_collector


def build_policy(behavior: dict[str, Any], seed: int) -> tuple[ModelExport, DecoderExport]:
    """Initialise et exporte l'acteur selon la dimension annoncée par Unity."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = PPOActorNetwork(observation_size=int(behavior["obs_size"]))
        return export_actor(actor)


if __name__ == "__main__":
    run_collector(policy_factory=build_policy, description=__doc__)
