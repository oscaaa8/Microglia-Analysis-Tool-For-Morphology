from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_skeleton_image(
    bwcb: np.ndarray,
    cell_body: np.ndarray,
    out_path: Path,
    *,
    figsize: tuple[float, float] = (8.0, 8.0),
) -> None:
    """
    Save a skeleton visualization similar to MATLAB getProp.m:
    - skeleton (bwcb) in grayscale
    - cell body highlighted (value 2)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bwcb = np.asarray(bwcb, dtype=bool)
    cell_body = np.asarray(cell_body, dtype=bool)

    tmp = bwcb.astype(np.float32)
    tmp[cell_body] = 2.0

    # Save the full field without cropping or aspect distortion.
    # This preserves the ROI's location within the original image space.
    plt.imsave(out_path, tmp, cmap="bone", vmin=0.0, vmax=2.0)
