import numpy as np
import torch
from captum.attr import IntegratedGradients as CaptumIG

from backend.core.explainer_registry import BaseExplainer, register_explainer


@register_explainer(
    name="integrated_gradients",
    display_name="Integrated Gradients",
    description="Attributes importance by integrating gradients along a path from a baseline to the input.",
    estimated_compute_time_seconds=5.0,
)
class IntegratedGradientsExplainer(BaseExplainer):

    def explain(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int
    ) -> np.ndarray:
        ig = CaptumIG(model)
        input_tensor = input_tensor.requires_grad_(True)
        attribution = ig.attribute(input_tensor, target=target_class, n_steps=50)

        # Aggregate across channels and normalize to [0, 1]
        attr_np = attribution.squeeze(0).detach().cpu().numpy()
        attr_np = np.mean(np.abs(attr_np), axis=0)  # (H, W)
        attr_max = attr_np.max()
        if attr_max > 0:
            attr_np = attr_np / attr_max
        return attr_np

    def get_params_schema(self) -> list[dict]:
        return [
            {"name": "n_steps", "type": "int", "default": 50, "description": "Number of interpolation steps"},
        ]
