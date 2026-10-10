import io
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_IMAGE_SIZE = 1024

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_and_validate_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
    except Exception:
        raise ValueError("Invalid image file. Please upload a valid image (JPEG, PNG, etc.).")

    # Convert to RGB (handles RGBA, CMYK, grayscale, palette)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if too large
    w, h = image.size
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    return image


def preprocess_image(image: Image.Image) -> torch.Tensor:
    tensor = _preprocess(image)
    return tensor.unsqueeze(0)  # Add batch dimension: (1, 3, 224, 224)


def get_original_image_array(image: Image.Image) -> np.ndarray:
    resized = image.copy()
    resized = resized.resize((224, 224), Image.LANCZOS)
    return np.array(resized)
