import os
import sys
import pytest
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def sample_image():
    """Create a simple 224x224 RGB test image."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_bytes(sample_image):
    """Return sample image as bytes (JPEG)."""
    import io
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    return buf.getvalue()
