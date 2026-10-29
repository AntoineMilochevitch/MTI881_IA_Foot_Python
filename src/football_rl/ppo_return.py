from football_rl.ppo_network import PPOValueNetwork
import torch

def _check_data(data: dict[str, torch.Tensor], observation_size: int) -> bool:
    """Vérifie que data respecte le bon format :
        - contient obs et next_obs
        - obs et next_obs sont de forme (N, D)
        - contient au moins une transition
        - obs et next_obs ont la même forme
        - D correspond à observation_size
    """
    if "obs" not in data or "next_obs" not in data:
        return False

    obs = data["obs"]
    next_obs = data["next_obs"]

    if obs.ndim != 2 or next_obs.ndim != 2:
        return False

    if obs.size(0) <= 0 or next_obs.size(0) <= 0:
        return False

    if obs.shape != next_obs.shape:
        return False

    return obs.size(1) == observation_size
        
    
def compute_reference_values(
    critics: PPOValueNetwork, data: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Calcule les valeurs de référence du critique pour les obsversations et les obsvervations suivantes.
    
    critics :
        réseau de valeur du critique
    data :
        dictionnaire contenant les observations et les actions
    """
    if not _check_data(data, critics.observation_size):
        raise ValueError("data doit contenir des observations et des next_observations de forme (N, D) non vides.")
    
    device = next(critics.parameters()).device
    obs = data["obs"].to(device)
    next_obs = data["next_obs"].to(device)
    
    with torch.no_grad():
        reference_values = critics(obs)["value"]
        next_reference_values = critics(next_obs)["value"]
        
    return {
        "reference_values": reference_values.cpu(),
        "next_reference_values": next_reference_values.cpu()
    }

def compute_gae(
    data: dict[str, torch.Tensor],
    reference_values: torch.Tensor,
    next_reference_values: torch.Tensor,
    gamma: float,
    lambda_: float,
    segments: list[list[int]]
) -> dict[str, torch.Tensor]:
    """Calcule l'avantage généralisé (GAE) pour les transitions données.
    
    data :
        dictionnaire contenant les récompenses, les indicateurs de terminaison et de troncature
    reference_values :
        valeurs de référence du critique pour les observations
    next_reference_values :
        valeurs de référence du critique pour les observations suivantes
    gamma :
        facteur d'actualisation
    lambda_ :
        paramètre GAE
    segments :
        listes des indices des transitions pour chaque segment
    """
    rewards = data["rewards"]
    terminated = data["terminated"]
    truncated = data["truncated"]
    
    delta = rewards + gamma * next_reference_values * (1 - (terminated).float()) - reference_values
    advantages = torch.zeros_like(delta)
    
    for segment in segments:
        if not segment:
            continue
        
        last_advantage = 0.0
        
        for t in reversed(segment):
            mask = 1.0 - (terminated[t] | truncated[t]).float()
            advantages[t] = last_advantage = delta[t] + gamma * lambda_ * mask * last_advantage
    
    returns = advantages + reference_values
    return {
        "advantages": advantages,
        "returns": returns
    }