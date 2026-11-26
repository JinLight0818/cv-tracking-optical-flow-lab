"""Evaluation metrics and helper primitives for object tracking and optical flow."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


def validate_box(box: Optional[Sequence[float]], frame_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """Return True if (x, y, w, h) is finite, positive-area, and overlaps frame bounds."""
    if box is None or len(box) != 4:
        return False
    x, y, w, h = box
    if not all(math.isfinite(float(v)) for v in (x, y, w, h)):
        return False
    if w <= 0.0 or h <= 0.0:
        return False
    if frame_shape is not None:
        fh, fw = frame_shape[:2]
        if x + w <= 0.0 or y + h <= 0.0 or x >= fw or y >= fh:
            return False
    return True


def iou(box_a: Optional[Sequence[float]], box_b: Optional[Sequence[float]]) -> float:
    """Compute Intersection over Union for two (x, y, w, h) bounding boxes."""
    if not validate_box(box_a) or not validate_box(box_b):
        return 0.0
    ax, ay, aw, ah = (float(v) for v in box_a)
    bx, by, bw, bh = (float(v) for v in box_b)
    inter_x = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_y = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_x * inter_y
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def epe(pred_vec: Tuple[float, float], gt_vec: Tuple[float, float]) -> float:
    """Compute Euclidean Endpoint Error (in pixels) between two 2D flow vectors."""
    return math.hypot(float(pred_vec[0]) - float(gt_vec[0]), float(pred_vec[1]) - float(gt_vec[1]))


def mean_epe(pairs: Iterable[Tuple[Tuple[float, float], Tuple[float, float]]]) -> float:
    """Compute mean Endpoint Error over an iterable of (pred, gt) vector pairs."""
    vals = [epe(p, g) for p, g in pairs]
    return sum(vals) / len(vals) if vals else float("nan")


def epe_map(pred_flow: np.ndarray, gt_flow: np.ndarray, valid_mask: np.ndarray) -> Tuple[float, int]:
    """Compute mean EPE across all True pixels in valid_mask."""
    mask = np.asarray(valid_mask, dtype=bool)
    mask &= np.isfinite(pred_flow).all(axis=2) & np.isfinite(gt_flow).all(axis=2)
    count = int(mask.sum())
    if count == 0:
        return float("nan"), 0
    diff = pred_flow[mask] - gt_flow[mask]
    dist = np.hypot(diff[:, 0], diff[:, 1])
    return float(dist.mean()), count


def read_flo(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Middlebury .flo file and return (flow_hw2, valid_mask_hw)."""
    with open(path, "rb") as f:
        tag, = struct.unpack("<f", f.read(4))
        if abs(tag - 202021.25) > 1e-3:
            raise ValueError(f"Invalid .flo header tag in {path}: {tag}")
        width, height = struct.unpack("<ii", f.read(8))
        raw = np.fromfile(f, dtype="<f4", count=width * height * 2)
    if raw.size != width * height * 2:
        raise ValueError(f"Truncated .flo file {path}: expected {width * height * 2} floats, got {raw.size}")
    flow = raw.reshape(height, width, 2).astype(np.float32)
    valid = (np.abs(flow[..., 0]) < 1e9) & (np.abs(flow[..., 1]) < 1e9) & np.isfinite(flow).all(axis=2)
    return flow, valid


def write_flo(path: Path, flow: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> None:
    """Write a 2-band float32 optical flow array to Middlebury .flo format."""
    h, w = flow.shape[:2]
    out = np.asarray(flow, dtype="<f4").copy()
    if valid_mask is not None:
        out[~np.asarray(valid_mask, dtype=bool)] = np.float32(1e10)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<fii", 202021.25, w, h))
        out.tofile(f)


def mask_to_bbox(mask: np.ndarray, obj_id: int) -> Optional[Tuple[int, int, int, int]]:
    """Compute tight visible-part bounding box (x, y, w, h) for obj_id in mask."""
    ys, xs = np.where(mask == obj_id)
    if xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
