"""
Buffer pour stocker les transitions pour l'algo PPO.
Il stocke :
- observations
- actions
- rewards
- next_observations
- terminated
- truncated
- old_log_probs
- agent_id
- episode_id
- step
- model_version
"""

import torch
from iafoot.server import Batch
import numpy as np


class PPORolloutBuffer:
    def __init__(self, session_id: str, behavior: str, policy_version: int, observation_size: int):
        self.session_id = session_id
        self.behavior = behavior
        self.policy_version = policy_version 
        self.observation_size = observation_size
        
        self.chunks: list[dict[str, torch.Tensor]] = []
        self.count = 0
        
    
    @staticmethod
    def _validate_shapes(batch: Batch, observation_size: int) -> None:
        """Vérifie que les champs décrivent le même nombre de transitions."""
        if batch.next_obs is None:
            raise ValueError("next_obs est requis pour le buffer PPO.")

        if batch.obs.ndim != 2:
            raise ValueError("obs doit avoir la forme (N, D).")

        n = batch.obs.shape[0]

        expected_shapes = {
            "obs": (n, observation_size),
            "actions": (n, 7),
            "next_obs": (n, observation_size),
        }

        vector_fields = (
            "rewards",
            "terminated",
            "truncated",
            "log_probs",
            "agent_id",
            "episode_id",
            "step",
            "model_version",
        )

        for name in vector_fields:
            expected_shapes[name] = (n,)

        for name, expected_shape in expected_shapes.items():
            array = getattr(batch, name)

            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} : forme attendue {expected_shape}, "
                    f"forme reçue {array.shape}"
                )
             
    @staticmethod
    def _version_mask(batch: Batch, policy_version: int) -> np.ndarray:
        """Sélectionne les transitions produites par la politique attendue."""
        return batch.model_version == policy_version
    
    @staticmethod
    def _to_cpu_tensor(array: np.ndarray, mask: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        """Copie les lignes sélectionnées dans un tenseur CPU."""
        selected = array[mask]

        return torch.tensor(
            selected,
            dtype=dtype,
            device="cpu",
        )

    def add_batch(self, batch: Batch, batch_id: int, session_id: str) -> int:
        if self.session_id != session_id:
            raise ValueError(f"Session ID mismatch: expected {self.session_id}, got {session_id}")
        
        if batch.behavior != self.behavior:
            raise ValueError(f"Behavior mismatch: expected {self.behavior}, got {batch.behavior}")

        if batch.batch_id != batch_id:
            raise ValueError(f"Batch ID mismatch: expected {batch_id}, got {batch.batch_id}")

        if batch.next_obs is None:
            raise ValueError("next_obs is required for the PPO rollout buffer")
        
        self._validate_shapes(batch, self.observation_size)
        mask = self._version_mask(batch, self.policy_version)
        mask_count = int(mask.sum())
        
        if (mask_count == 0):
            return 0
        
        chunk = {
            "obs": self._to_cpu_tensor(batch.obs, mask, torch.float32),
            "next_obs": self._to_cpu_tensor(batch.next_obs, mask, torch.float32),
            "old_log_probs": self._to_cpu_tensor(batch.log_probs, mask, torch.float32),
            "actions": self._to_cpu_tensor(batch.actions, mask, torch.float32),
            "rewards": self._to_cpu_tensor(batch.rewards, mask, torch.float32),
            "terminated": self._to_cpu_tensor(batch.terminated, mask, torch.bool),
            "truncated": self._to_cpu_tensor(batch.truncated, mask, torch.bool),
            "agent_id": self._to_cpu_tensor(batch.agent_id, mask, torch.int32),
            "episode_id": self._to_cpu_tensor(batch.episode_id, mask, torch.int32),
            "step": self._to_cpu_tensor(batch.step, mask, torch.int32),
            "model_version": self._to_cpu_tensor(batch.model_version, mask, torch.int32),
        }
        
        self.chunks.append(chunk)
        self.count += mask_count
        
        return mask_count
            

    def __len__(self) -> int:
        return self.count  
    
    def get_data(self)-> dict[str, torch.Tensor]:
        """Retourne les données du buffer sous forme de dictionnaire."""
        if not self.chunks:
            return {}

        keys = self.chunks[0].keys()
        return {
            key: torch.cat([chunk[key] for chunk in self.chunks], dim=0)
            for key in keys
        }
    
    def clear(self):
        """Vide le buffer."""
        self.chunks = []
        self.count = 0
        
    def build_segment(self) -> list[list[int]]:
        """Construit un segment de transitions à partir des chunks accumulés."""
        
        data = self.get_data()
        if not data:
            return []
        """clé : (agent_id, episode_id) et valeur : liste des indices de transitions"""
        segment_dict: dict[tuple[int, int], list[int]] = {}
        
        for idx in range(len(data["agent_id"])):
            key = (int(data["agent_id"][idx]), int(data["episode_id"][idx]))
            if key not in segment_dict:
                segment_dict[key] = []
            segment_dict[key].append(idx)
            
        """trier les indices de transitions pour chaque (agent_id, episode_id) par ordre croissant de step"""
        for key in segment_dict:
            segment_dict[key].sort(key=lambda idx: data["step"][idx].item())
        
        """vérifier que les indices consécutif ne sont pas égaux"""
        for key, indices in segment_dict.items():
            for i in range(len(indices) - 1):
                if data["step"][indices[i + 1]] == data["step"][indices[i]]:
                    raise ValueError(f"Indices consécutifs égaux pour {key} : {data['step'][indices[i]]}")
        
        """découper en segments continus"""
        segments: list[list[int]] = []
        for key, indices in segment_dict.items():
            current_segment: list[int] = []
            for i in range (len(indices)):
                idx = indices[i]
                current_segment.append(idx)
                
                if data["terminated"][idx] or data["truncated"][idx]:
                    segments.append(current_segment)
                    current_segment = []
                    
                elif i < len(indices) - 1 and data["step"][indices[i + 1]] != data["step"][idx] + 1:
                    segments.append(current_segment)
                    current_segment = []
                    
            if current_segment:
                segments.append(current_segment)
                
        return segments
                
        