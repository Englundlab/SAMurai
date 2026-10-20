"""Stable instance colors shared by the viewer and RGB TIFF export."""
MASK_PALETTE = ("#72b7b2", "#8caad4", "#b5a2d2", "#95c5b2", "#a2bfda", "#c5b5cf")
MASK_SELECTED_COLOR = "#e8c477"


def instance_rgb(labels):
    import numpy as np
    ids, inverse = np.unique(labels, return_inverse=True)
    colors = np.zeros((len(ids), 3), dtype=np.uint8)
    for index, label in enumerate(ids):
        if label:
            color = MASK_PALETTE[(int(label) - 1) % len(MASK_PALETTE)]
            colors[index] = [int(color[i:i+2], 16) for i in (1, 3, 5)]
    return colors[inverse].reshape(*labels.shape, 3)
