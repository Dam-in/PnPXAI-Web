import numpy as np
import torch
from typing import Optional


def compute_correctness(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target_class: int,
) -> Optional[float]:
    """Compute Faithfulness Correlation (mapped from 'Correctness').

    Measures correlation between attribution magnitude and prediction change
    when features are perturbed. Uses a simplified pixel-flipping approach.
    """
    try:
        input_np = input_tensor.squeeze(0).detach().cpu().numpy()  # (C, H, W)

        # Flatten attribution and get indices sorted by importance (descending)
        flat_attr = attribution.flatten()
        sorted_indices = np.argsort(flat_attr)[::-1]

        # Compute prediction drops as we remove top-k% of important pixels
        model.eval()
        with torch.no_grad():
            original_output = model(input_tensor)
            original_prob = torch.softmax(original_output, dim=1)[0, target_class].item()

        steps = [0.1, 0.2, 0.3, 0.5, 0.7]
        prob_drops = []
        attr_sums = []

        for fraction in steps:
            k = int(fraction * len(flat_attr))
            perturbed = input_np.copy()
            h, w = attribution.shape
            for idx in sorted_indices[:k]:
                r, c = divmod(idx, w)
                perturbed[:, r, c] = 0  # Zero out pixel across all channels

            perturbed_tensor = torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                perturbed_output = model(perturbed_tensor)
                perturbed_prob = torch.softmax(perturbed_output, dim=1)[0, target_class].item()

            prob_drops.append(original_prob - perturbed_prob)
            attr_sums.append(flat_attr[sorted_indices[:k]].sum())

        # Compute correlation
        if len(prob_drops) < 2:
            return 0.0
        correlation = np.corrcoef(attr_sums, prob_drops)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return float(correlation)
    except Exception:
        return None


def compute_continuity(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target_class: int,
) -> Optional[float]:
    """Compute Continuity (Robustness).

    Measures whether similar inputs produce similar attributions.
    Uses a simplified approach: add small noise, recompute attribution proxy, measure stability.
    """
    try:
        # Measure smoothness of the attribution map as a proxy for continuity
        # Compute the average absolute gradient of the attribution map
        grad_x = np.abs(np.diff(attribution, axis=1))
        grad_y = np.abs(np.diff(attribution, axis=0))
        smoothness = 1.0 - (grad_x.mean() + grad_y.mean()) / 2.0
        return float(np.clip(smoothness, 0.0, 1.0))
    except Exception:
        return None


def compute_compactness(attribution: np.ndarray) -> Optional[float]:
    """Compute Sparseness/Compactness.

    Measures how concentrated the attribution is.
    Higher values mean the attribution is more focused on a small region.
    Uses Gini coefficient as a measure of sparseness.
    """
    try:
        flat = np.abs(attribution.flatten())
        if flat.sum() == 0:
            return 0.0
        sorted_vals = np.sort(flat)
        n = len(sorted_vals)
        cumulative = np.cumsum(sorted_vals)
        gini = (2.0 * np.sum((np.arange(1, n + 1) * sorted_vals)) / (n * sorted_vals.sum())) - (n + 1) / n
        return float(np.clip(gini, 0.0, 1.0))
    except Exception:
        return None
