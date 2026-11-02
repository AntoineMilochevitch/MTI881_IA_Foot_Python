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



def restore_networks(checkpoint: dict[str, object], actor: PPOActorNetwork, critic: PPOValueNetwork) -> None:
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    
def restore_optimizers(checkpoint: dict[str, object], actor_optimizer: torch.optim.Optimizer, critic_optimizer: torch.optim.Optimizer) -> None:
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

def validate_schema(checkpoint: dict[str, object], expected_schema: tuple) -> None:
    if checkpoint["schema"] != expected_schema:
        raise ValueError(f"Checkpoint schema {checkpoint['schema']} does not match expected schema {expected_schema}")