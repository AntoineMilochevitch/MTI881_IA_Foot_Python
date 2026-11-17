from football_rl.ppo_network import PPOActorNetwork, PPOValueNetwork
import torch
from pathlib import Path

_FORMAT_VERSION = 1
_REQUIRED_KEYS = (
    "format_version",
    "actor_state_dict",
    "critic_state_dict",
    "actor_optimizer_state_dict",
    "critic_optimizer_state_dict",
    "progress",
    "schema",
    "config",
    "torch_rng_state",
)
_PROGRESS_COUNTERS = (
    "version_counter",
    "update_count",
    "trained_transitions",
    "ignored_transitions",
)


def build_checkpoint(
    actor: PPOActorNetwork, 
    critic: PPOValueNetwork, 
    actor_optimizer: torch.optim.Optimizer, 
    critic_optimizer: torch.optim.Optimizer,
    progress: dict[str, int],
    schema: tuple,
    config: dict[str, object]
    ) -> dict[str, object]:
    config = config.copy()
    config["device"] = str(config["device"])
    return {
        "format_version": _FORMAT_VERSION,
        # Métadonnées indépendantes de la CLI. Les anciens checkpoints n'ont
        # pas ce champ : leurs dimensions seront lues directement dans les poids.
        "architecture": {"actor_hidden_size": actor.hidden_size,
                         "critic_hidden_size": critic.hidden_size},
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "progress": progress.copy(),
        "schema": schema,
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
    }
    
def save_checkpoint(checkpoint: dict[str, object], path: str) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(f"{target_path.name}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    
def load_checkpoint(path: str) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint: expected a dictionary")

    missing_keys = [key for key in _REQUIRED_KEYS if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            f"Invalid checkpoint: missing required entries: {', '.join(missing_keys)}"
        )

    format_version = checkpoint["format_version"]
    if isinstance(format_version, bool) or format_version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: {format_version}"
        )

    progress = checkpoint["progress"]
    if not isinstance(progress, dict):
        raise ValueError("Invalid checkpoint: progress must be a dictionary")
    for counter in _PROGRESS_COUNTERS:
        if counter not in progress:
            raise ValueError(
                f"Invalid checkpoint: missing progress counter '{counter}'"
            )
        value = progress[counter]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Invalid checkpoint: progress counter '{counter}' "
                f"must be a non-negative integer, got {value!r}"
            )

    return checkpoint



def checkpoint_hidden_size(checkpoint: dict[str, object], network: str) -> int:
    """Retrouve la largeur à construire AVANT de charger les poids et Adam.

    Aucun agrandissement des anciens poids : un modèle 128 reste un modèle 128.
    Le nouveau modèle 256 et une ancienne snapshot 128 peuvent ainsi coexister.
    """
    if network not in ("actor", "critic"):
        raise ValueError("network doit valoir actor ou critic.")
    weight = checkpoint[f"{network}_state_dict"]["network.0.weight"]
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] <= 0:
        raise ValueError(f"Première couche du réseau {network} invalide.")
    width = weight.shape[0]
    architecture = checkpoint.get("architecture", {})
    if not isinstance(architecture, dict):
        raise ValueError("Les métadonnées architecture doivent être un dictionnaire.")
    declared = architecture.get(f"{network}_hidden_size", width)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared != width:
        raise ValueError(f"Architecture {network} incompatible avec les poids sauvegardés.")
    return width


def restore_networks(checkpoint: dict[str, object], actor: PPOActorNetwork, critic: PPOValueNetwork) -> None:
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    
def restore_optimizers(checkpoint: dict[str, object], actor_optimizer: torch.optim.Optimizer, critic_optimizer: torch.optim.Optimizer) -> None:
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

def validate_schema(checkpoint: dict[str, object], expected_schema: tuple) -> None:
    if checkpoint["schema"] != expected_schema:
        raise ValueError(f"Checkpoint schema {checkpoint['schema']} does not match expected schema {expected_schema}")
