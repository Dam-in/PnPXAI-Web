import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def render_text_attribution(
    tokens: list[str],
    attribution: np.ndarray,
    output_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize attribution to [0, 1]
    attr = np.abs(attribution.flatten())
    if len(attr) > len(tokens):
        attr = attr[:len(tokens)]
    elif len(attr) < len(tokens):
        attr = np.pad(attr, (0, len(tokens) - len(attr)))

    attr_max = attr.max()
    if attr_max > 0:
        attr = attr / attr_max

    # Render as horizontal bar chart with token labels
    fig, ax = plt.subplots(figsize=(6, max(2, len(tokens) * 0.4)), dpi=100)

    cmap = plt.cm.YlOrRd
    colors = [cmap(v) for v in attr]

    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, attr, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_title("Token Attribution", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return output_path
