import torch
import torchvision.models as models
from backend.core.model_registry import register_model


@register_model(
    name="resnet50",
    display_name="ResNet-50",
    architecture="Residual Network",
    description="50-layer deep residual network. Most widely used baseline in XAI research.",
    lrp_compatible=False,
)
def load_resnet50() -> torch.nn.Module:
    return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)


@register_model(
    name="vgg16",
    display_name="VGG-16",
    architecture="Sequential CNN",
    description="16-layer sequential CNN. Simple architecture ideal for layer-wise XAI methods like LRP.",
    lrp_compatible=True,
)
def load_vgg16() -> torch.nn.Module:
    return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)


@register_model(
    name="densenet121",
    display_name="DenseNet-121",
    architecture="Dense Connections",
    description="121-layer densely connected network. Different connectivity pattern for comparison.",
    lrp_compatible=False,
)
def load_densenet121() -> torch.nn.Module:
    return models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
