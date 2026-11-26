"""Evaluation metrics and helper primitives for object tracking and optical flow."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
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


def compute_flow_regions(img_bgr: np.ndarray, gt_flow: np.ndarray, gt_valid: np.ndarray) -> Dict[str, np.ndarray]:
    """Partition valid pixels into spatial/texture/object-motion regions for EPE analysis."""
    h, w = gt_valid.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)

    border = np.zeros((h, w), dtype=bool)
    b = min(24, h // 8, w // 8)
    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True
    border_valid = gt_valid & border
    interior_valid = gt_valid & (~border)

    if interior_valid.any():
        med_grad = float(np.median(grad_mag[interior_valid]))
    else:
        med_grad = 0.0
    low_tex = interior_valid & (grad_mag <= med_grad)
    high_tex = interior_valid & (grad_mag > med_grad)

    # Motion discontinuity / object boundary detection via flow gradient magnitude
    fu = np.where(gt_valid, gt_flow[..., 0], 0.0)
    fv = np.where(gt_valid, gt_flow[..., 1], 0.0)
    fgrad = np.hypot(cv2.Sobel(fu, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(fu, cv2.CV_32F, 0, 1, ksize=3))
    fgrad += np.hypot(cv2.Sobel(fv, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(fv, cv2.CV_32F, 0, 1, ksize=3))
    boundary_raw = (fgrad > 1.2) | (~gt_valid)
    kernel = np.ones((11, 11), dtype=np.uint8)
    boundary_band = cv2.dilate(boundary_raw.astype(np.uint8), kernel).astype(bool) & gt_valid

    flow_mag = np.hypot(fu, fv)
    # Identify foreground moving region when flow has both static/slow and fast pixels
    if gt_valid.any():
        p20 = float(np.percentile(flow_mag[gt_valid], 20))
        p90 = float(np.percentile(flow_mag[gt_valid], 90))
        if p90 - p20 > 1.0:
            thresh = p20 + 0.4 * (p90 - p20)
            inside_obj = gt_valid & (~boundary_band) & (flow_mag >= thresh)
            background = gt_valid & (~boundary_band) & (flow_mag < thresh)
        else:
            inside_obj = np.zeros((h, w), dtype=bool)
            background = gt_valid & (~boundary_band)
    else:
        inside_obj = np.zeros((h, w), dtype=bool)
        background = np.zeros((h, w), dtype=bool)

    return {
        "all_valid": gt_valid,
        "border_band": border_valid,
        "low_texture": low_tex,
        "high_texture": high_tex,
        "object_boundary": boundary_band,
        "inside_object": inside_obj,
        "background": background,
    }


def compute_recovery_stats(
    rows: Sequence[Dict],
    reappear_frame: Optional[int],
    iou_threshold: float = 0.5,
    consecutive: int = 3,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Compute (start_offset, confirmed_offset, first_ok_frame, confirmed_frame) after reappear_frame.

    start_offset = first frame of the 3-consecutive IoU>=0.5 streak minus reappear_frame.
    confirmed_offset = 3rd frame of that streak minus reappear_frame.
    Returns (None, None, None, None) if not recovered within clip or reappear_frame is None.
    """
    if reappear_frame is None:
        return None, None, None, None
    streak = 0
    for r in rows:
        fi = int(r["frame"])
        if fi < reappear_frame:
            continue
        if int(r["gt_present"]) == 1 and float(r["iou"]) >= iou_threshold:
            streak += 1
            if streak >= consecutive:
                confirmed_frame = fi
                first_ok_frame = fi - consecutive + 1
                return (
                    first_ok_frame - reappear_frame,
                    confirmed_frame - reappear_frame,
                    first_ok_frame,
                    confirmed_frame,
                )
        else:
            streak = 0
    return None, None, None, None


def read_flo(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Middlebury/Sintel .flo file and return (flow_hw2, valid_mask_hw)."""
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
