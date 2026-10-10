import numpy as np
import torch
from captum.attr import LRP as CaptumLRP

from backend.core.explainer_registry import BaseExplainer, register_explainer
from backend.core.model_registry import is_lrp_compatible


@register_explainer(
    name="lrp",
    display_name="LRP",
    description="Layer-wise Relevance Propagation. Decomposes prediction into pixel-level contributions. Only compatible with VGG16 (nn.Sequential).",
    requires_sequential=True,
    estimated_compute_time_seconds=2.0,
)
class LRPExplainer(BaseExplainer):

    def is_compatible(self, model_name: str) -> bool:
        return is_lrp_compatible(model_name)

    def explain(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int
    ) -> np.ndarray:
        lrp = CaptumLRP(model)
        input_tensor = input_tensor.requires_grad_(True)
        attribution = lrp.attribute(input_tensor, target=target_class)

        attr_np = attribution.squeeze(0).detach().cpu().numpy()
        attr_np = np.mean(np.abs(attr_np), axis=0)
        attr_max = attr_np.max()
        if attr_max > 0:
            attr_np = attr_np / attr_max
        return attr_np

    def get_params_schema(self) -> list[dict]:
        return []
