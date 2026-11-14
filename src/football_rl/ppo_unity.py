"""Entraîne PPO à partir des transitions Unity, ou vérifie la collecte seule."""

import argparse
from dataclasses import fields
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

import torch
from iafoot import Batch, DecoderExport, ModelExport, TrainingServer, UnityConnection

from football_rl.ppo_checkpoint import load_checkpoint
from football_rl.ppo_export import export_actor
from football_rl.ppo_network import PPOActorNetwork
from football_rl.ppo_opponent import FrozenOpponent, FrozenOpponentPool
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
    parser.add_argument("--rollout-size", type=int, default=None,
                        help="transitions de la même politique avant une mise à jour PPO")
    parser.add_argument("--mini-batch-size", type=int, default=None,
                        help="transitions par pas des optimiseurs")
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--actor-lr", type=float, default=None,
                        help="learning rate acteur ; remplace aussi celui du checkpoint")
    parser.add_argument("--critic-lr", type=float, default=defaults.critic_lr)
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="bonus d'entropie acteur (défaut : 0 ; reprise : valeur sauvegardée)")
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    # None permet de distinguer une option absente d'un zéro explicitement
    # demandé, afin de ne pas changer silencieusement la récompense à la reprise.
    parser.add_argument("--ball-approach-shaping", type=float, default=None,
                        help="coefficient du potentiel joueur-ballon (défaut : 0 ; reprise : valeur sauvegardée)")
    parser.add_argument("--gae-lambda", dest="lambda_", type=float, default=defaults.lambda_)
    parser.add_argument("--clip-epsilon", type=float, default=defaults.clip_epsilon)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    kl_group = parser.add_mutually_exclusive_group()
    kl_group.add_argument("--target-kl", type=float, default=defaults.target_kl)
    kl_group.add_argument("--no-target-kl", dest="target_kl", action="store_const", const=None)
    parser.add_argument("--parity-atol", type=float, default=None,
                        help="écart absolu maximal accepté entre logp Python et Unity")
    parser.add_argument("--collect-only", action="store_true",
                        help="vérifie les données et log-probabilités sans optimiser")
    parser.add_argument("--evaluate", action="store_true",
                        help="évalue un checkpoint sans entraîner")
    parser.add_argument("--evaluation-episodes", type=int, default=defaults.evaluation_episodes,
                        help="nombre d'épisodes sportifs à évaluer (but marqué, encaissé ou timeout)")
    parser.add_argument("--diagnostic-no-jump-tackle", action="store_true",
                        help="désactive tacle et saut pendant une évaluation")
    parser.add_argument("--log-dir", default="runs",
                        help="dossier racine des logs TensorBoard")
    parser.add_argument("--run-name", default=None,
                        help="nom de l'expérience (défaut : ppo_AAAAMMJJ-HHMMSS)")
    parser.add_argument("--resume", default=None,
                        help="chemin du checkpoint à reprendre")
    parser.add_argument("--opponent", action="append", nargs=2, default=[],
                        metavar=("BEHAVIOR", "CHECKPOINT"),
                        help="snapshot fig?e ; r?p?ter cette option pour plusieurs adversaires")
    parser.add_argument("--opponent-checkpoint", default=None,
                        help="chemin de la snapshot de l'adversaire")
    parser.add_argument("--opponent-behavior", default="FootSnapshot",
                        help="comportement Unity de l'adversaire")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="dossier des nouvelles sauvegardes")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="désactive le suivi TensorBoard")
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
    if args.evaluate and args.collect_only:
        parser.error("--evaluate est incompatible avec --collect-only.")
    if args.evaluate and args.resume is None:
        parser.error("--evaluate exige --resume.")
    if args.diagnostic_no_jump_tackle and not args.evaluate:
        parser.error("--diagnostic-no-jump-tackle exige --evaluate.")
    if args.evaluation_episodes <= 0:
        parser.error("--evaluation-episodes doit être un entier strictement positif.")

    run_name = args.run_name or datetime.now().strftime("ppo_%Y%m%d-%H%M%S")
    checkpoint_dir = args.checkpoint_dir or str(Path("checkpoints") / run_name)
    resume_checkpoint = None
    if args.resume is not None:
        try:
            resume_checkpoint = load_checkpoint(args.resume)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            parser.error(f"Impossible de charger le checkpoint : {error}")

    if args.no_tensorboard:
        args.log_dir = None
    else:
        args.log_dir = str(Path(args.log_dir) / run_name)

    try:
        if resume_checkpoint is None:
            config_values = {
                field.name: getattr(args, field.name) for field in fields(PPOTrainerConfig)
            }
            config_values["ball_approach_shaping"] = (
                args.ball_approach_shaping if args.ball_approach_shaping is not None else 0.0
            )
        else:
            saved_config = resume_checkpoint["config"]
            if not isinstance(saved_config, dict):
                raise ValueError("La configuration du checkpoint doit être un dictionnaire.")
            saved_shaping = saved_config.get("ball_approach_shaping", 0.0)
            if args.ball_approach_shaping is not None and args.ball_approach_shaping != saved_shaping:
                raise ValueError(
                    "--resume conserve le shaping sauvegardé. Pour changer "
                    "--ball-approach-shaping, démarrer un nouvel entraînement sans --resume."
                )
            config_values = saved_config.copy()
            config_values.update({
                "device": args.device,
                "log_dir": args.log_dir,
                "collect_only": args.collect_only,
                "evaluate": args.evaluate,
                "evaluation_episodes": args.evaluation_episodes,
            })
        # Une option absente conserve le checkpoint. Une option explicite
        # remplace son réglage et sera enregistrée dans les prochaines sauvegardes.
        defaults = PPOTrainerConfig()
        for name in ("actor_lr", "rollout_size", "mini_batch_size", "parity_atol", "entropy_coef"):
            value = getattr(args, name)
            if value is not None:
                config_values[name] = value
            elif resume_checkpoint is None:
                config_values[name] = getattr(defaults, name)
        config = PPOTrainerConfig(**config_values)
    except (ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))

    if args.opponent and args.opponent_checkpoint is not None:
        parser.error("Utiliser --opponent ou --opponent-checkpoint, pas les deux.")
    opponent_specs = args.opponent or (
        [(args.opponent_behavior, args.opponent_checkpoint)] if args.opponent_checkpoint else []
    )
    names = [name for name, _ in opponent_specs]
    if len(set(names)) != len(names) or any(
        not name.strip() or name == "*" or name == config.behavior for name in names
    ):
        parser.error("Les comportements adverses doivent ?tre distincts entre eux et de l'apprenant.")
    try:
        opponents = [FrozenOpponent(path, behavior=name) for name, path in opponent_specs]
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Impossible de charger une snapshot adverse : {error}")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    # Le RNG Python sert aux permutations des mini-lots ; Unity a son propre RNG.
    torch.manual_seed(config.seed)
    trainer = PPOUnityTrainer(
        config,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=resume_checkpoint,
        diagnostic_no_jump_tackle=args.diagnostic_no_jump_tackle,
    )
    server = TrainingServer(args.host, args.port, debug=args.verbose)
    logger = logging.getLogger(__name__)

    # Le pool vide conserve aussi le fonctionnement solo / bot / self-play.
    pool = FrozenOpponentPool(opponents, trainer)
    server.on_connect = pool.on_connect
    server.on_batch = pool.on_batch
    server.on_model_ack = pool.on_model_ack
    server.on_disconnect = pool.on_disconnect
    mode = (
        "évaluation"
        if config.evaluate
        else "diagnostic sans optimisation"
        if config.collect_only
        else "entraînement"
    )
    logging.getLogger(__name__).info(
        "PPO : %s, lot Unity=%d, rollout=%d, mini-lot=%d, époques=%d.",
        mode, config.batch_size, config.rollout_size, config.mini_batch_size, config.num_epochs,
    )
    if resume_checkpoint is not None:
        logging.getLogger(__name__).info(
            "Reprise : réglages sauvegardés utilisés, sauf les options de session "
            "et les remplacements explicites actor-lr, rollout-size, mini-batch-size, parity-atol, entropy-coef."
        )
    logger.info(
        "Réglages effectifs : LR acteur=%g, LR critique=%g, cible KL=%s, tolérance logp=%g.",
        config.actor_lr, config.critic_lr, config.target_kl, config.parity_atol,
    )
    logger.info("Coefficient d'entropie acteur : %g.", config.entropy_coef)
    if config.ball_approach_shaping > 0:
        logging.getLogger(__name__).info(
            "Shaping joueur-ballon : alpha=%g, gamma=%g. %s "
            "Pour remplacer la pénalité de distance Unity, régler Distance To Ball Per Second à 0 "
            "sur les FootAgent entraînés (ce réglage n'est pas transmis par le pont).",
            config.ball_approach_shaping, config.gamma,
            "Évaluation : récompenses Unity uniquement." if config.evaluate
            else "Appliqué aux transitions avant le GAE.",
        )
    if args.diagnostic_no_jump_tackle:
        logging.getLogger(__name__).info(
            "Diagnostic : tacle et saut désactivés dans l'acteur en mémoire."
        )

    if config.log_dir is not None and not config.collect_only:
        logging.getLogger(__name__).info("TensorBoard : %s", config.log_dir)

    try:
        server.serve_forever()
    finally:
        try:
            server.stop()
        finally:
            trainer.close()


if __name__ == "__main__":
    main()
