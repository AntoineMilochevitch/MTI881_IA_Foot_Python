import math

import torch
from football_rl.ppo_network import PPOActorNetwork, PPOValueNetwork
from football_rl.ppo_buffer import PPORolloutBuffer
from football_rl.ppo_return import compute_gae, compute_reference_values


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalise les avantages pour avoir une moyenne de 0 et un écart-type de 1.

    advantages :
        avantages à normaliser
    """
    
    if advantages.ndim != 1 or advantages.size(0) <= 0 or advantages.is_floating_point() is False:
        raise ValueError("Les avantages doivent être un tenseur 1D.")
    
    mean = advantages.mean()
    std = advantages.std(correction=0)  # correction=0 pour l'écart-type de la population
    return (advantages - mean) / (std + 1e-8)


def prepare_training_data(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Prépare les données d'entraînement à partir des données collectées.

    data :
        dictionnaire contenant les données collectées et les avantages
    """
    return {
        "obs": data["obs"],
        "actions": data["actions"],
        "old_log_probs": data["old_log_probs"],
        "advantages": _normalize_advantages(data["advantages"]),
        "returns": data["returns"],
    }
    
def iterate_mini_batches(training_data: dict[str, torch.Tensor], mini_batch_size: int):
    """Itère sur les mini-batches de données d'entraînement.

    training_data :
        dictionnaire contenant les données d'entraînement
    mini_batch_size :
        taille des mini-batches    
    """
    
    if mini_batch_size <= 0:
        raise ValueError("La taille du mini-batch doit être un entier positif.")
    
    if len(training_data["obs"]) <= 0:
        raise ValueError("Les données d'entraînement sont vides.")  
    
    n = training_data["obs"].shape[0]
    for key, tensor in training_data.items():
        if tensor.ndim == 0 or tensor.shape[0] != n:
            raise ValueError(
                f"Le tenseur « {key} » doit avoir une première dimension de taille {n}."
            )

    permutation = torch.randperm(training_data["obs"].shape[0])
    for i in range(0, len(permutation), mini_batch_size):
        indices = permutation[i:i + mini_batch_size]
        yield {key: tensor[indices] for key, tensor in training_data.items()}
        
def compute_actor_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float
) -> torch.Tensor:
    """Calcule la perte de l'acteur en utilisant la méthode PPO.

    new_log_probs :
        log-probabilités des actions sous la nouvelle politique
    old_log_probs :
        log-probabilités des actions sous l'ancienne politique
    advantages :
        avantages normalisés
    clip_epsilon :
        paramètre de clipping pour PPO
    """
    
    tensors = {
        "new_log_probs": new_log_probs,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
    }
    if any(not tensor.is_floating_point() for tensor in tensors.values()):
        raise ValueError("Les trois tenseurs doivent être de type flottant.")

    if any(tensor.ndim != 1 or tensor.shape[0] == 0 for tensor in tensors.values()):
        raise ValueError("Les trois tenseurs doivent être non vides et avoir la forme (B,).")

    if not (new_log_probs.shape == old_log_probs.shape == advantages.shape):
        raise ValueError("Les trois tenseurs doivent avoir la même forme.")

    if not (new_log_probs.device == old_log_probs.device == advantages.device):
        raise ValueError("Les trois tenseurs doivent être sur le même périphérique.")

    if not (0 < clip_epsilon < 1):
        raise ValueError("Le paramètre clip_epsilon doit être un nombre positif entre 0 et 1.")
    
    log_prob_difference = new_log_probs - old_log_probs
    ratio = torch.exp(log_prob_difference)
    loss_unclipped = ratio * advantages

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss_clipped = clipped_ratio * advantages
    
    return -torch.mean(torch.min(loss_unclipped, loss_clipped))

def compute_critic_loss(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor
) -> torch.Tensor:
    """Calcule la perte du critique en utilisant l'erreur quadratique moyenne.

    predicted_values :
        valeurs prédites par le critique
    target_values :
        valeurs cibles pour l'entraînement du critique
    """
    
    if not (predicted_values.is_floating_point() and target_values.is_floating_point()):
        raise ValueError("Les deux tenseurs doivent être de type flottant.")

    if not (predicted_values.ndim == target_values.ndim == 1):
        raise ValueError("Les deux tenseurs doivent être des vecteurs 1D.")

    if not (predicted_values.numel() > 0 and target_values.numel() > 0):
        raise ValueError("Les deux tenseurs doivent être non vides.")
    
    if not (predicted_values.shape[0] == target_values.shape[0]):
        raise ValueError("Les deux tenseurs doivent avoir la même taille.")

    if not (predicted_values.device == target_values.device):
        raise ValueError("Les deux tenseurs doivent être sur le même périphérique.")
    
    return torch.mean((predicted_values - target_values) ** 2)

def update_minibatch(
    actor: PPOActorNetwork,
    critic: PPOValueNetwork,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    mini_batch: dict[str, torch.Tensor],
    clip_epsilon: float,
    max_grad_norm: float
) -> dict[str, float]:
    """Met à jour les réseaux de l'acteur et du critique pour un mini-batch donné.

    actor :
        réseau de l'acteur
    critic :
        réseau du critique
    actor_optimizer :
        optimiseur pour le réseau de l'acteur
    critic_optimizer :
        optimiseur pour le réseau du critique
    mini_batch :
        dictionnaire contenant les données du mini-batch
    clip_epsilon :
        paramètre de clipping pour PPO
    max_grad_norm :
        norme maximale des gradients pour le clipping
    """
    
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    new_log_probs = actor.evaluate_action(mini_batch["obs"], mini_batch["actions"])
    
    predicted_values = critic(mini_batch["obs"])["value"]
    
    actor_loss = compute_actor_loss(
        new_log_probs=new_log_probs,
        old_log_probs=mini_batch["old_log_probs"],
        advantages=mini_batch["advantages"],
        clip_epsilon=clip_epsilon
    )
    
    critic_loss = compute_critic_loss(
        predicted_values=predicted_values,
        target_values=mini_batch["returns"]
    )

    # Arrêter avant optimizer.step si un lot réel fait diverger les calculs.
    if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
        raise ValueError("Les pertes PPO doivent être finies.")
    
    actor_loss.backward()
    critic_loss.backward()
    
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        critic.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    
    actor_optimizer.step()
    critic_optimizer.step()
    
    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "actor_grad_norm": actor_grad_norm.detach().item(),
        "critic_grad_norm": critic_grad_norm.detach().item() 
    }
    
def train_ppo_epochs(
    actor: PPOActorNetwork,
    critic: PPOValueNetwork,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    training_data: dict[str, torch.Tensor],
    clip_epsilon: float,
    max_grad_norm: float,
    num_epochs: int,
    mini_batch_size: int,
    target_kl: float | None = None
) -> list[dict[str, float | int | bool]]:
    """Entraîne les réseaux de l'acteur et du critique sur plusieurs époques.

    actor :
        réseau de l'acteur
    critic :
        réseau du critique
    actor_optimizer :
        optimiseur pour le réseau de l'acteur
    critic_optimizer :
        optimiseur pour le réseau du critique
    training_data :
        dictionnaire contenant les données d'entraînement
    clip_epsilon :
        paramètre de clipping pour PPO
    max_grad_norm :
        norme maximale des gradients pour le clipping
    num_epochs :
        nombre d'époques d'entraînement
    mini_batch_size :
        taille des mini-batches
    target_kl :
        seuil facultatif d'arrêt anticipé sur la divergence KL
    """
    
    if not isinstance(num_epochs, int) or not isinstance(mini_batch_size, int):
        raise TypeError("Le nombre d'époques et la taille du mini-batch doivent être des entiers.")
    
    if num_epochs <= 0:
        raise ValueError("Le nombre d'époques doit être un entier positif.")
    
    if mini_batch_size <= 0:
        raise ValueError("La taille du mini-batch doit être un entier positif.")

    if target_kl is not None and (
        isinstance(target_kl, bool)
        or not isinstance(target_kl, (int, float))
        or not math.isfinite(target_kl)
        or target_kl <= 0
    ):
        raise ValueError("target_kl doit être un nombre fini strictement positif ou None.")
    
    if len(training_data["obs"]) <= 0:
        raise ValueError("Les données d'entraînement sont vides.")
    
    history = []
    
    for epoch in range(num_epochs):
        critic_sum_loss = 0.0
        actor_sum_loss = 0.0
        critic_grad_norm_sum = 0.0
        actor_grad_norm_sum = 0.0
        transition_count = 0
        update_count = 0
        for mini_batch in iterate_mini_batches(training_data, mini_batch_size):
            loss_info = update_minibatch(
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                mini_batch=mini_batch,
                clip_epsilon=clip_epsilon,
                max_grad_norm=max_grad_norm
            )
            batch_size = len(mini_batch["obs"])
            critic_sum_loss += batch_size * loss_info["critic_loss"]
            actor_sum_loss += batch_size * loss_info["actor_loss"]
            critic_grad_norm_sum += loss_info["critic_grad_norm"]
            actor_grad_norm_sum += loss_info["actor_grad_norm"]
            transition_count += batch_size
            update_count += 1

        policy_metrics = evaluate_policy_on_rollout(
            actor=actor,
            training_data=training_data,
            clip_epsilon=clip_epsilon,
            evaluation_batch_size=mini_batch_size
        )
        kl_limit_reached = (
            target_kl is not None
            and policy_metrics["approx_kl"] > target_kl
        )

        history.append({
            "epoch": epoch + 1,
            "actor_loss": actor_sum_loss / transition_count,
            "critic_loss": critic_sum_loss / transition_count,
            "actor_grad_norm": actor_grad_norm_sum / update_count,
            "critic_grad_norm": critic_grad_norm_sum / update_count,
            "num_samples": transition_count,
            "num_updates": update_count,
            "approx_kl": policy_metrics["approx_kl"],
            "clip_fraction": policy_metrics["clip_fraction"],
            "ratio_mean": policy_metrics["ratio_mean"],
            "kl_limit_reached": kl_limit_reached
        })

        if kl_limit_reached:
            break
    
    return history

def compute_policy_metrics(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_epsilon: float
) -> dict[str, float]:
    """Calcule les métriques de la politique pour évaluer l'entraînement.

    new_log_probs :
        log-probabilités des actions sous la nouvelle politique
    old_log_probs :
        log-probabilités des actions sous l'ancienne politique
    clip_epsilon :
        paramètre de clipping pour PPO
    """
    
    if not (new_log_probs.is_floating_point() and old_log_probs.is_floating_point()):
        raise ValueError("Les deux tenseurs doivent être de type flottant.")

    if not (new_log_probs.ndim == old_log_probs.ndim == 1):
        raise ValueError("Les deux tenseurs doivent être des vecteurs 1D.")

    if not (new_log_probs.numel() > 0 and old_log_probs.numel() > 0):
        raise ValueError("Les deux tenseurs doivent être non vides.")
    
    if not (new_log_probs.shape[0] == old_log_probs.shape[0]):
        raise ValueError("Les deux tenseurs doivent avoir la même taille.")

    if not (new_log_probs.device == old_log_probs.device):
        raise ValueError("Les deux tenseurs doivent être sur le même périphérique.")
    
    if not (0 < clip_epsilon < 1):
        raise ValueError("Le paramètre clip_epsilon doit être un nombre positif entre 0 et 1.")
    
    if not torch.isfinite(new_log_probs).all() or not torch.isfinite(old_log_probs).all():
        raise ValueError("Les deux tenseurs doivent contenir uniquement des valeurs finies.")
    
    
    with torch.no_grad():
        dif_log_probs = new_log_probs - old_log_probs
    
        ratio = torch.exp(dif_log_probs)
        approx_kl = torch.mean(torch.expm1(dif_log_probs) - dif_log_probs).item()
        clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_epsilon).float()).item()
        ratio_mean = torch.mean(ratio).item()
    
    return {
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "ratio_mean": ratio_mean
    }
    
def evaluate_policy_on_rollout(
    actor: PPOActorNetwork,
    training_data: dict[str, torch.Tensor],
    clip_epsilon: float,
    evaluation_batch_size: int
) -> dict[str, float]:
    """Évalue la politique sur un rollout en utilisant les métriques PPO.

    actor :
        réseau de l'acteur
    training_data :
        dictionnaire contenant les données d'entraînement
    clip_epsilon :
        paramètre de clipping pour PPO
    evaluation_batch_size :
        taille des mini-batches pour l'évaluation
    """

    required_keys = ("obs", "actions", "old_log_probs")
    if any(key not in training_data for key in required_keys):
        raise ValueError("Les données doivent contenir obs, actions et old_log_probs.")

    observations = training_data["obs"]
    actions = training_data["actions"]
    old_log_probs = training_data["old_log_probs"]
    sample_count = observations.shape[0]
    if sample_count <= 0:
        raise ValueError("Les données d'évaluation sont vides.")
    if not (actions.shape[0] == old_log_probs.shape[0] == sample_count):
        raise ValueError("Les données d'évaluation doivent avoir la même taille.")
    if not isinstance(evaluation_batch_size, int) or isinstance(evaluation_batch_size, bool):
        raise TypeError("La taille du mini-batch d'évaluation doit être un entier.")
    if evaluation_batch_size <= 0:
        raise ValueError("La taille du mini-batch d'évaluation doit être positive.")

    metric_sums = {
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "ratio_mean": 0.0
    }
    for start in range(0, sample_count, evaluation_batch_size):
        end = min(start + evaluation_batch_size, sample_count)
        with torch.no_grad():
            new_log_probs = actor.evaluate_action(
                observations[start:end],
                actions[start:end]
            )
            metrics = compute_policy_metrics(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs[start:end],
                clip_epsilon=clip_epsilon
            )
        batch_size = end - start
        for name, value in metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"La métrique {name} n'est pas finie.")
            metric_sums[name] += batch_size * value

    return {
        name: value / sample_count
        for name, value in metric_sums.items()
    }


def train_ppo(
    actor: PPOActorNetwork,
    critic: PPOValueNetwork,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    buffer: PPORolloutBuffer,
    gamma: float,
    lambda_: float,
    clip_epsilon: float,
    max_grad_norm: float,
    num_epochs: int,
    mini_batch_size: int,
    target_kl: float | None = None
) -> list[dict[str, float | int | bool]]:
    """Prépare une collecte puis entraîne PPO pendant plusieurs époques."""
    if len(buffer) <= 0:
        raise ValueError("Le buffer PPO est vide.")

    if buffer.observation_size != actor.observation_size:
        raise ValueError("La dimension du buffer et celle de l'acteur diffèrent.")
    if buffer.observation_size != critic.observation_size:
        raise ValueError("La dimension du buffer et celle du critique diffèrent.")

    actor_device = next(actor.parameters()).device
    critic_device = next(critic.parameters()).device
    if actor_device != critic_device:
        raise ValueError("L'acteur et le critique doivent être sur le même périphérique.")

    for name, value in (("gamma", gamma), ("lambda_", lambda_)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} doit être un nombre fini compris entre 0 et 1.")

    data = buffer.get_data()
    segments = buffer.build_segment()
    reference_values = compute_reference_values(critic, data)
    gae = compute_gae(
        data=data,
        reference_values=reference_values["reference_values"],
        next_reference_values=reference_values["next_reference_values"],
        gamma=gamma,
        lambda_=lambda_,
        segments=segments
    )

    prepared_data = prepare_training_data({
        "obs": data["obs"],
        "actions": data["actions"],
        "old_log_probs": data["old_log_probs"],
        "advantages": gae["advantages"],
        "returns": gae["returns"]
    })
    prepared_data = {
        name: tensor.to(actor_device)
        for name, tensor in prepared_data.items()
    }

    return train_ppo_epochs(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        training_data=prepared_data,
        clip_epsilon=clip_epsilon,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        mini_batch_size=mini_batch_size,
        target_kl=target_kl
    )
            
