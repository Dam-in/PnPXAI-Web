import numpy as np
import torch
from captum.attr import KernelShap as CaptumKernelShap

from backend.core.explainer_registry import BaseExplainer, register_explainer


@register_explainer(
    name="kernel_shap",
    display_name="KernelSHAP",
    description="Kernel SHAP uses weighted linear regression to estimate Shapley values. Perturbation-based, slower but theoretically grounded.",
    estimated_compute_time_seconds=25.0,
)
class KernelSHAPExplainer(BaseExplainer):

    def explain(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int,
        n_samples: int = 50,
    ) -> np.ndarray:
        ks = CaptumKernelShap(model)
        attribution = ks.attribute(
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
            {"name": "n_samples", "type": "int", "default": 50, "description": "Number of perturbed samples (higher = more accurate but slower)"},
        ]
