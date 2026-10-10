import numpy as np
import torch
from captum.attr import Lime as CaptumLime

from backend.core.explainer_registry import BaseExplainer, register_explainer


@register_explainer(
    name="lime",
    display_name="LIME",
    description="Local Interpretable Model-agnostic Explanations. Perturbation-based, slower but model-agnostic.",
    estimated_compute_time_seconds=25.0,
)
class LIMEExplainer(BaseExplainer):

    def explain(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int,
        n_samples: int = 64,
    ) -> np.ndarray:
        lime = CaptumLime(model)
        input_tensor = input_tensor.requires_grad_(True)

        attribution = lime.attribute(
            input_tensor,
            target=target_class,
            n_samples=n_samples,
        )

        attr_np = attribution.squeeze(0).detach().cpu().numpy()
        attr_np = np.mean(np.abs(attr_np), axis=0)
        attr_max = attr_np.max()
        if attr_max > 0:
            attr_np = attr_np / attr_max
        return attr_np

    def get_params_schema(self) -> list[dict]:
        return [
            {"name": "n_samples", "type": "int", "default": 64, "description": "Number of perturbed samples (higher = more accurate but slower)"},
        ]
