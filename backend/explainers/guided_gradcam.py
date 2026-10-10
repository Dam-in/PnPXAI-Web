import numpy as np
import torch
from captum.attr import GuidedGradCam as CaptumGGC

from backend.core.explainer_registry import BaseExplainer, register_explainer

# Map model names to their last convolutional layer
_TARGET_LAYERS = {
    "resnet50": "layer4",
    "vgg16": "features.29",
    "densenet121": "features.denseblock4",
}


def _get_target_layer(model: torch.nn.Module, model_name: str):
    layer_path = _TARGET_LAYERS.get(model_name, "")
    parts = layer_path.split(".")
    layer = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    return layer


@register_explainer(
    name="guided_gradcam",
    display_name="Guided Grad-CAM",
    description="Combines Guided Backpropagation with Grad-CAM for high-resolution, class-discriminative visualizations.",
    estimated_compute_time_seconds=3.0,
)
class GuidedGradCamExplainer(BaseExplainer):

    def explain(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int,
        model_name: str = "resnet50",
    ) -> np.ndarray:
        target_layer = _get_target_layer(model, model_name)
        ggc = CaptumGGC(model, target_layer)
        input_tensor = input_tensor.requires_grad_(True)
        attribution = ggc.attribute(input_tensor, target=target_class)

        attr_np = attribution.squeeze(0).detach().cpu().numpy()
        attr_np = np.mean(np.abs(attr_np), axis=0)
        attr_max = attr_np.max()
        if attr_max > 0:
            attr_np = attr_np / attr_max
        return attr_np

    def get_params_schema(self) -> list[dict]:
        return [
            {"name": "target_layer", "type": "str", "default": "auto", "description": "Target convolutional layer (auto = last conv layer)"},
        ]
