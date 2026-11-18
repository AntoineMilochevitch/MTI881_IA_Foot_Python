"""Validation du DQN sur CartPole-v1 avant de passer au jeu Unity.

Le but est de vérifier que l'algorithme apprend sur un problème connu : si le
DQN n'apprend pas dans Unity, on saura alors que le problème vient de
l'environnement ou des récompenses, pas de l'algorithme.

Le script utilise ``DQNLearner`` (``football_rl.dqn``), exactement le même cœur
que l'entraînement Unity (``football_rl.dqn_unity``) : cible de Bellman, Double
DQN, Huber, clipping, réseau cible (copie ou Polyak), ε, ``train_after`` (un pas
de gradient toutes les ``--train-freq`` transitions), ``DQNNetwork`` et ``ReplayBuffer``.
Seule la collecte diffère : ici l'ε-greedy est fait en Python, dans Unity par le
décodeur ``discrete``.

Mesures, par graine :
- meilleure éval et éval finale (politique gloutonne, ``--eval-episodes`` épisodes) ;
- moyenne des évals sur la seconde moitié de l'entraînement (stabilité) ;
- part des évals de la seconde moitié au-dessus du seuil de 475 ;
- score du meilleur checkpoint réévalué sur 100 épisodes (critère officiel).

Les valeurs par défaut sont la configuration la plus stable trouvée (Double DQN,
τ = 0.005, lr 2.5e-4, MLP 256, lot 128, tampon 100k, ε final 0.01) : sur 5 graines,
meilleur checkpoint à 500/500 pour toutes, éval finale 496.6 ± 6.9.

Lancement depuis la racine du dépôt Python :
    $env:PYTHONPATH = "src"; python scripts/dqn_cartpole.py --double --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from football_rl.dqn import DQNConfig, DQNLearner
from football_rl.dqn_network import DQNNetwork

SOLVED_THRESHOLD = 475.0


@dataclass(slots=True)
class RunSummary:
    seed: int
    best_eval: float
    final_eval: float
    second_half_mean: float
    second_half_solved_frac: float
    best_checkpoint_100: float
    first_solved_step: int | None
    duration_s: float


def greedy_action(net: DQNNetwork, obs: np.ndarray, device: torch.device) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return int(net(obs_t)["q_values"].argmax(dim=1).item())


def evaluate(net: DQNNetwork, device: torch.device, seed: int, episodes: int) -> float:
    """Retour moyen de la politique gloutonne (sans exploration)."""
    env = gym.make("CartPole-v1")
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(greedy_action(net, obs, device))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def run(seed: int, args: argparse.Namespace, base_config: DQNConfig) -> RunSummary:
    config = DQNConfig(**{**asdict(base_config), "seed": seed})
    device = torch.device("cpu")  # un petit MLP va plus vite sur CPU qu'avec les transferts GPU

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    num_actions = int(env.action_space.n)
    learner = DQNLearner(env.observation_space.shape[0], num_actions, config, device)

    writer = None
    if args.logdir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f"{args.logdir}/cartpole_{args.tag}_seed{seed}")

    recent_returns: deque[float] = deque(maxlen=20)
    episode_return, episodes = 0.0, 0
    loss, mean_q = float("nan"), float("nan")
    evals: list[tuple[int, float]] = []
    best_eval, best_state = -1.0, None
    first_solved_step: int | None = None
    eval_seed = 10_000 + seed * 100
    start = time.perf_counter()

    for step in range(1, args.steps + 1):
        learner.env_steps = step  # ε suit le nombre de pas d'environnement
        action = learner.act(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Seul terminated coupe le bootstrap : une coupure à 500 pas n'est pas un échec.
        learner.store(obs, action, float(reward), next_obs, terminated)
        episode_return += float(reward)

        if terminated or truncated:
            recent_returns.append(episode_return)
            episodes += 1
            if writer:
                writer.add_scalar("train/episode_return", episode_return, step)
            episode_return = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        result = learner.train_after(1)
        if result["steps"]:
            loss, mean_q = result["loss"], result["mean_q"]
            if writer and step % 100 == 0:
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/mean_q", mean_q, step)
                writer.add_scalar("train/epsilon", learner.epsilon, step)

        if step % args.eval_every == 0:
            score = evaluate(learner.online_net, device, eval_seed, args.eval_episodes)
            evals.append((step, score))
            if score >= best_eval:  # à égalité, on garde le plus récent
                best_eval, best_state = score, copy.deepcopy(learner.online_net.state_dict())
            if first_solved_step is None and score >= SOLVED_THRESHOLD:
                first_solved_step = step
            if writer:
                writer.add_scalar("eval/greedy_return", score, step)
            if not args.quiet:
                train_mean = np.mean(recent_returns) if recent_returns else 0.0
                print(
                    f"[{args.tag} seed {seed}] pas {step:>6} | ép {episodes:>4} | ε {learner.epsilon:.3f} | "
                    f"retour entr. {train_mean:6.1f} | éval {score:6.1f} | "
                    f"perte {loss:.4f} | Q moyen {mean_q:6.2f}",
                    flush=True,
                )

    env.close()
    if writer:
        writer.close()

    second_half = [score for step, score in evals if step > args.steps // 2]
    best_net = DQNNetwork(learner.online_net.observation_size, num_actions, config.hidden_size)
    best_net.load_state_dict(best_state)
    # Graines d'éval jamais vues pendant l'entraînement, pour ne pas surestimer le checkpoint.
    best_checkpoint_100 = evaluate(best_net, device, seed=50_000 + seed * 1_000, episodes=100)

    return RunSummary(
        seed=seed,
        best_eval=best_eval,
        final_eval=evals[-1][1],
        second_half_mean=float(np.mean(second_half)),
        second_half_solved_frac=float(np.mean([s >= SOLVED_THRESHOLD for s in second_half])),
        best_checkpoint_100=best_checkpoint_100,
        first_solved_step=first_solved_step,
        duration_s=time.perf_counter() - start,
    )


def main() -> None:
    # Sous Windows, une sortie redirigée est en cp1252, qui ne connaît pas ε, τ ou ≥.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="dqn", help="nom de la configuration (logs et résumé)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--double", action="store_true", help="active le Double DQN")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--target-update", type=int, default=500, help="en pas de gradient (si tau = 1)")
    parser.add_argument("--tau", type=float, default=0.005, help="< 1 : mise à jour douce du réseau cible")
    parser.add_argument("--eps-decay", type=int, default=10_000, help="en pas d'environnement")
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--train-freq", type=int, default=1, help="pas d'environnement par pas de gradient")
    parser.add_argument("--eval-every", type=int, default=2_500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--logdir", default=None, help="dossier TensorBoard (désactivé si absent)")
    parser.add_argument("--json", type=Path, default=None, help="ajoute le résumé à ce fichier JSON Lines")
    parser.add_argument("--quiet", action="store_true", help="n'affiche que le résumé")
    args = parser.parse_args()

    torch.set_num_threads(1)
    config = DQNConfig(
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=args.eps_end,
        epsilon_decay_steps=args.eps_decay,
        buffer_capacity=args.buffer,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        hidden_size=args.hidden,
        double_dqn=args.double,
        learning_starts=args.learning_starts,
        tau=args.tau,
        train_freq=args.train_freq,
    )
    print(f"[{args.tag}] CartPole-v1 : {config}", flush=True)

    summaries = [run(seed, args, config) for seed in args.seeds]

    print(f"\n[{args.tag}] Résumé (seuil 475)")
    for s in summaries:
        solved = f"1er succès au pas {s.first_solved_step}" if s.first_solved_step else "jamais résolu"
        print(f"  seed {s.seed} : meilleure {s.best_eval:5.1f} | finale {s.final_eval:5.1f} | "
              f"moy. 2e moitié {s.second_half_mean:5.1f} | ≥475 en 2e moitié {s.second_half_solved_frac:4.0%} | "
              f"meilleur checkpoint (100 ép.) {s.best_checkpoint_100:5.1f} | {solved} | {s.duration_s:.0f} s")

    def stat(values: list[float]) -> str:
        return f"{np.mean(values):5.1f} ± {np.std(values):4.1f}"

    print(f"  MOYENNE ± écart-type sur {len(summaries)} graines : "
          f"finale {stat([s.final_eval for s in summaries])} | "
          f"2e moitié {stat([s.second_half_mean for s in summaries])} | "
          f"meilleur checkpoint {stat([s.best_checkpoint_100 for s in summaries])}")

    if args.json:
        with args.json.open("a", encoding="utf-8") as f:
            for s in summaries:
                f.write(json.dumps({"tag": args.tag, "config": asdict(config), **asdict(s)}) + "\n")


if __name__ == "__main__":
    main()
