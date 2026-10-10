from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch

_explainer_registry: dict[str, dict] = {}


class BaseExplainer(ABC):
    name: str
    display_name: str
    description: str = ""
    requires_sequential: bool = False
    estimated_compute_time_seconds: float = 5.0

    @abstractmethod
    def explain(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int
    ) -> np.ndarray:
        """Return attribution map as numpy array with shape (H, W), values in [0, 1]."""
        ...

    def is_compatible(self, model_name: str) -> bool:
        return True

    def get_params_schema(self) -> list[dict]:
        return []


def register_explainer(
    name: str,
    display_name: str,
    description: str = "",
    requires_sequential: bool = False,
    estimated_compute_time_seconds: float = 5.0,
):
    def decorator(cls):
        instance = cls()
        _explainer_registry[name] = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "requires_sequential": requires_sequential,
            "estimated_compute_time_seconds": estimated_compute_time_seconds,
            "params": instance.get_params_schema(),
            "instance": instance,
        }
        return cls
    return decorator


def list_explainers() -> list[dict]:
    return [
        {k: v for k, v in info.items() if k != "instance"}
        for info in _explainer_registry.values()
    ]


def get_explainer(name: str) -> Optional[BaseExplainer]:
    info = _explainer_registry.get(name)
    if info is None:
        return None
    return info["instance"]


def get_explainer_info(name: str) -> Optional[dict]:
    info = _explainer_registry.get(name)
    if info is None:
        return None
    return {k: v for k, v in info.items() if k != "instance"}
