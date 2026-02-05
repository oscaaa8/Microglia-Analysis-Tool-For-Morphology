from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from collections import deque
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class PropResult:
    n_branch_points: int
    total_area: float
    mean_branch_length: float
    branch_depth: float
    cell_body_size: float
    debug: Dict[str, np.ndarray]


def znorm(z: np.ndarray) -> np.ndarray:
    """
    Rank-normalize all values in z to [0, 1], matching MATLAB znorm.m.
    Each voxel gets a unique rank based on ascending order.
    """
    z = np.asarray(z)
    flat = z.ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = (np.arange(flat.size, dtype=np.float64) + 1.0) / float(flat.size)
    return ranks.reshape(z.shape)


def _gaussian_smooth_3d(
    volume: np.ndarray,
    sigma: float = 1.0,
    kernel_size: int = 10,
) -> np.ndarray:
    """
    Approximate MATLAB: imfilter(double(r), fspecial3('gaussian',[10 10 10],1),'same')
    """
    volume = np.asarray(volume, dtype=np.float64)
    radius = kernel_size // 2
    truncate = float(radius) / float(sigma) if sigma > 0 else 0.0
    return ndi.gaussian_filter(volume, sigma=sigma, truncate=truncate, mode="constant", cval=0.0)


def _branch_points_2d(skel: np.ndarray) -> np.ndarray:
    """
    Approximate MATLAB bwmorph(bw, 'branchpoints') in 2D (8-connectivity).
    A branch point is a skeleton pixel with >= 3 neighbors.
    """
    skel = np.asarray(skel, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return skel & (neighbor_count >= 3)


def _remove_small_objects_2d(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove connected components smaller than min_size (2D, 8-connectivity).
    Implemented here to avoid version-specific warnings in skimage.
    """
    if min_size <= 0:
        return mask.astype(bool)

    labeled = label(mask, connectivity=2)
    if labeled.max() == 0:
        return mask.astype(bool)

    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labeled]


def branch_depth(bw: np.ndarray, cell_body: np.ndarray) -> np.ndarray:
    """
    Port of MATLAB branchDepth.m.
    Computes a depth map by walking the skeleton from cell body centroid,
    incrementing depth on branch points.
    """
    bw = np.asarray(bw, dtype=bool)
    cell_body = np.asarray(cell_body, dtype=bool)

    if not np.any(cell_body):
        return np.full(bw.shape, np.nan, dtype=np.float64)

    branch_points = _branch_points_2d(bw)
    branch_points[cell_body] = False

    bw = bw.copy()
    bw[cell_body] = True

    props = regionprops(cell_body.astype(np.uint8))
    if not props:
        return np.full(bw.shape, np.nan, dtype=np.float64)

    start_r, start_c = props[0].centroid
    start = (int(round(start_r)), int(round(start_c)))

    depth_map = np.full(bw.shape, np.nan, dtype=np.float64)
    have_not_checked = bw.copy()

    queue = deque([(start[0], start[1], 0)])
    if start[0] < 0 or start[0] >= bw.shape[0] or start[1] < 0 or start[1] >= bw.shape[1]:
        return depth_map

    while queue:
        r, c, depth = queue.popleft()
        if not have_not_checked[r, c]:
            continue

        have_not_checked[r, c] = False
        current_depth = depth + 1 if branch_points[r, c] else depth
        depth_map[r, c] = current_depth

        r0 = max(r - 1, 0)
        r1 = min(r + 1, bw.shape[0] - 1)
        c0 = max(c - 1, 0)
        c1 = min(c + 1, bw.shape[1] - 1)

        neighborhood = have_not_checked[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any(neighborhood):
            continue

        rr, cc = np.where(neighborhood)
        for dr, dc in zip(rr, cc):
            nr = r0 + dr
            nc = c0 + dc
            queue.append((nr, nc, current_depth))

    return depth_map


def get_prop(
    roi_mask: np.ndarray,
    *,
    smooth_sigma: float = 1.0,
    smooth_kernel_size: int = 10,
    mip_thresh: float = 0.1,
    body_thresh: float = 0.9,
    min_branch_length: int = 20,
) -> PropResult:
    """
    Port of MATLAB getProp.m.

    Args:
      roi_mask: 2D or 3D boolean array. If 3D, a ROI volume. If 2D, treated as a single-slice ROI.
    """
    roi_mask = np.asarray(roi_mask, dtype=bool)
    # Normalize to (z, y, x) so we can safely max-project over z.
    if roi_mask.ndim == 2:
        roi_mask_3d = roi_mask[None, :, :]
    elif roi_mask.ndim == 3:
        roi_mask_3d = roi_mask
    else:
        raise ValueError("roi_mask must be 2D or 3D.")

    smth = _gaussian_smooth_3d(roi_mask_3d, sigma=smooth_sigma, kernel_size=smooth_kernel_size)
    # Max-project across z to produce a 2D (y, x) image.
    mps = np.nanmax(smth, axis=0)

    bw = mps > mip_thresh
    skel = skeletonize(bw)
    if min_branch_length > 0:
        skel = _remove_small_objects_2d(skel, min_size=min_branch_length)

    body_mask = mps > body_thresh
    labeled_body = label(body_mask, connectivity=2)
    props = regionprops(labeled_body)
    if props:
        largest = max(props, key=lambda p: p.area)
        cell_body = labeled_body == largest.label
    else:
        cell_body = np.zeros_like(body_mask, dtype=bool)

    branch_points = _branch_points_2d(skel)
    branch_points[cell_body] = False
    n_branch_points = int(np.nansum(branch_points))

    bwcb = skel.copy()
    bwcb[cell_body] = False

    depth_map = branch_depth(skel, cell_body)
    branch_depth_val = float(np.nanmax(depth_map)) if np.any(~np.isnan(depth_map)) else np.nan

    total_area = float(np.nansum(bwcb))
    mean_branch_length = (
        float(total_area / n_branch_points) if n_branch_points > 0 else np.nan
    )
    cell_body_size = float(np.nansum(cell_body))

    debug = {
        "mps": mps,
        "bw": bw,
        "skel": skel,
        "bwcb": bwcb,
        "cell_body": cell_body,
        "branch_points": branch_points,
        "depth_map": depth_map,
    }

    return PropResult(
        n_branch_points=n_branch_points,
        total_area=total_area,
        mean_branch_length=mean_branch_length,
        branch_depth=branch_depth_val,
        cell_body_size=cell_body_size,
        debug=debug,
    )
