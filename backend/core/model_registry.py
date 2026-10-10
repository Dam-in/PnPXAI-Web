import torch
from typing import Optional

_model_registry: dict[str, dict] = {}
_loaded_models: dict[str, torch.nn.Module] = {}


def register_model(name: str, display_name: str, architecture: str,
                   description: str, lrp_compatible: bool):
    def decorator(loader_fn):
        _model_registry[name] = {
            "name": name,
            "display_name": display_name,
            "architecture": architecture,
            "description": description,
            "lrp_compatible": lrp_compatible,
            "loader": loader_fn,
        }
        return loader_fn
    return decorator


def list_models() -> list[dict]:
    return [
        {k: v for k, v in info.items() if k != "loader"}
        for info in _model_registry.values()
    ]


def get_model(name: str) -> Optional[torch.nn.Module]:
    if name not in _model_registry:
        return None
    if name not in _loaded_models:
        loader = _model_registry[name]["loader"]
        model = loader()
        model.eval()
        _loaded_models[name] = model
    return _loaded_models[name]


def get_model_info(name: str) -> Optional[dict]:
    info = _model_registry.get(name)
    if info is None:
        return None
    return {k: v for k, v in info.items() if k != "loader"}


def is_lrp_compatible(model_name: str) -> bool:
    info = _model_registry.get(model_name)
    if info is None:
        return False
    return info["lrp_compatible"]
