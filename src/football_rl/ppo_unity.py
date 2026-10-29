"""Entraîne PPO à partir des transitions Unity, ou vérifie la collecte seule."""

import argparse
from dataclasses import fields
import logging
import sys
from typing import Any

import torch
from iafoot import DecoderExport, ModelExport, TrainingServer

from football_rl.ppo_export import export_actor
from football_rl.ppo_network import PPOActorNetwork
from football_rl.ppo_trainer import PPOTrainerConfig, PPOUnityTrainer


def build_policy(behavior: dict[str, Any], seed: int) -> tuple[ModelExport, DecoderExport]:
    """Fabrique conservée pour le collecteur générique, sans apprentissage."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = PPOActorNetwork(observation_size=int(behavior["obs_size"]))
        return export_actor(actor)


def build_parser() -> argparse.ArgumentParser:
    defaults = PPOTrainerConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--behavior", default=defaults.behavior)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device, help="cpu ou cuda[:index]")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size,
                        help="transitions par message envoyé par Unity")
    parser.add_argument("--time-scale", type=float, default=defaults.time_scale)
    parser.add_argument("--rollout-size", type=int, default=defaults.rollout_size,
                        help="transitions de la même politique avant une mise à jour PPO")
    parser.add_argument("--mini-batch-size", type=int, default=defaults.mini_batch_size,
                        help="transitions par pas des optimiseurs")
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--actor-lr", type=float, default=defaults.actor_lr)
    parser.add_argument("--critic-lr", type=float, default=defaults.critic_lr)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", dest="lambda_", type=float, default=defaults.lambda_)
    parser.add_argument("--clip-epsilon", type=float, default=defaults.clip_epsilon)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    kl_group = parser.add_mutually_exclusive_group()
    kl_group.add_argument("--target-kl", type=float, default=defaults.target_kl)
    kl_group.add_argument("--no-target-kl", dest="target_kl", action="store_const", const=None)
    parser.add_argument("--parity-atol", type=float, default=defaults.parity_atol,
                        help="écart absolu maximal accepté entre logp Python et Unity")
    parser.add_argument("--collect-only", action="store_true",
                        help="vérifie les données et log-probabilités sans optimiser")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port doit être entre 1 et 65535.")
    try:
        config = PPOTrainerConfig(**{
            field.name: getattr(args, field.name) for field in fields(PPOTrainerConfig)
        })
    except (ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    # Le RNG Python sert aux permutations des mini-lots ; Unity a son propre RNG.
    torch.manual_seed(config.seed)
    trainer = PPOUnityTrainer(config)
    server = TrainingServer(args.host, args.port, debug=args.verbose)
    server.on_connect = trainer.on_connect
    server.on_batch = trainer.on_batch
    server.on_model_ack = trainer.on_model_ack
    server.on_disconnect = trainer.on_disconnect
    mode = "diagnostic sans optimisation" if config.collect_only else "entraînement"
    logging.getLogger(__name__).info(
        "PPO : %s, lot Unity=%d, rollout=%d, mini-lot=%d, époques=%d.",
        mode, config.batch_size, config.rollout_size, config.mini_batch_size, config.num_epochs,
    )
    try:
        server.serve_forever()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
