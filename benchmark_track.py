"""Single-thread, multi-repeat object tracking benchmark across DAVIS and controlled CC0 data."""

from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    ALL_TRACKERS,
    CSV_DIR,
    DATA_DIR,
    IOU_THRESHOLD,
    RECOVERY_CONSECUTIVE,
    TIMING_REPEATS,
    enforce_single_thread,
    ensure_dirs,
)
from metrics import iou, validate_box
from tracker_binding_check import resolve_tracker_factory, safe_tracker_init


@dataclass
class SequenceSpec:
    name: str
    gt_csv: Path
    img_dir: Path
    reappear_frame: Optional[int]
    description: str


SEQUENCES = [
    SequenceSpec(
        name="ctrl_bicycle_occ",
        gt_csv=CSV_DIR / "ctrl_bicycle_occ_gt.csv",
        img_dir=DATA_DIR / "controlled" / "bicycle_occ_seq",
        reappear_frame=23,  # Frame 18..22 are 100% occluded; frame 23 is first visible reappearance
        description="CC0 photo controlled sequence: moving target + scale + distractor + 5-frame full occlusion",
    ),
    SequenceSpec(
        name="ctrl_bicycle_cam_shift",
        gt_csv=CSV_DIR / "ctrl_bicycle_cam_shift_gt.csv",
        img_dir=DATA_DIR / "controlled" / "bicycle_cam_shift_seq",
        reappear_frame=None,
        description="CC0 photo controlled sequence: whole-frame camera pan shift (-6, -1 px/frame)",
    ),
    SequenceSpec(
        name="davis_bmx_trees_bike",
        gt_csv=CSV_DIR / "davis_bmx_trees_bike_gt.csv",
        img_dir=DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees",
        reappear_frame=62,  # Minimum visible area (426 px) behind tree trunk occurs at frame 61
        description="DAVIS 2017 bmx-trees (obj_id=1 bike): visible-part GT, heavy tree occlusion & scale shrink",
    ),
    SequenceSpec(
        name="davis_bmx_trees_rider",
        gt_csv=CSV_DIR / "davis_bmx_trees_bike_gt.csv".replace("bike", "rider"),
        img_dir=DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees",
        reappear_frame=69,  # Minimum visible area (429 px) occurs at frame 68
        description="DAVIS 2017 bmx-trees (obj_id=2 rider): visible-part GT, heavy occlusion & camera pan",
    ),
    SequenceSpec(
        name="davis_car_shadow",
        gt_csv=CSV_DIR / "davis_car_shadow_gt.csv",
        img_dir=DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "car-shadow",
        reappear_frame=None,
        description="DAVIS 2017 car-shadow (obj_id=1 car): ~3.5x area shrink + shadow illumination change",
    ),
]


def load_sequence(spec: SequenceSpec) -> Tuple[List[np.ndarray], List[Dict]]:
    """Load frames and GT rows from disk outside any timed section."""
    frames: List[np.ndarray] = []
    gt_rows: List[Dict] = []
    with open(spec.gt_csv, newline="") as f:
        for row in csv.DictReader(f):
            img_path = spec.img_dir / row["image_file"]
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Missing frame: {img_path}")
            frames.append(img)
            present = int(row.get("present", 1))
            if "vis_x" in row:
                box = (float(row["vis_x"]), float(row["vis_y"]), float(row["vis_w"]), float(row["vis_h"]))
            else:
                box = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
            gt_rows.append({"frame": int(row["frame"]), "present": present, "box": box})
    return frames, gt_rows


def _run_once(
    tracker_name: str,
    frames: List[np.ndarray],
    init_roi: Tuple[int, int, int, int],
) -> Tuple[str, bool, List[Tuple[bool, Tuple[float, float, float, float]]], float]:
    """Run one full pass of tracker_name; return (binding, init_ok, predictions, update_ms_per_frame)."""
    prefer_legacy = tracker_name in ("BOOSTING", "TLD", "MEDIANFLOW")
    factory, binding = resolve_tracker_factory(tracker_name, prefer_legacy=prefer_legacy)
    if factory is None:
        return binding, False, [(False, (0.0, 0.0, 0.0, 0.0)) for _ in frames], float("nan")

    tracker = factory()
    init_ok, _ = safe_tracker_init(tracker, frames[0], init_roi)
    if not init_ok:
        return binding, False, [(False, (0.0, 0.0, 0.0, 0.0)) for _ in frames], float("nan")

    preds: List[Tuple[bool, Tuple[float, float, float, float]]] = [(True, tuple(float(v) for v in init_roi))]
    t0 = time.perf_counter()
    for frame in frames[1:]:
        ok, box = tracker.update(frame)
        preds.append((bool(ok), tuple(float(v) for v in box)))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_per_frame = elapsed_ms / max(1, len(frames) - 1)
    return binding, True, preds, ms_per_frame


def evaluate_tracker_on_sequence(spec: SequenceSpec, tracker_name: str) -> Tuple[Dict, List[Dict]]:
    """Evaluate a tracker on one sequence across TIMING_REPEATS runs."""
    frames, gt_rows = load_sequence(spec)
    fh, fw = frames[0].shape[:2]
    init_roi = tuple(int(round(v)) for v in gt_rows[0]["box"])

    run_times: List[float] = []
    binding = ""
    init_ok = False
    preds: List[Tuple[bool, Tuple[float, float, float, float]]] = []
    for _ in range(TIMING_REPEATS):
        binding, init_ok, preds, ms_pf = _run_once(tracker_name, frames, init_roi)
        if np.isfinite(ms_pf):
            run_times.append(ms_pf)

    median_ms = statistics.median(run_times) if run_times else float("nan")
    fps = 1000.0 / median_ms if median_ms > 0 and np.isfinite(median_ms) else 0.0

    rows: List[Dict] = []
    eval_ious: List[float] = []
    failures = 0
    drift_success_true = 0
    absent_total = 0
    absent_fp = 0
    no_gt_frames = 0

    for idx in range(len(frames)):
        gt = gt_rows[idx]
        gt_present = gt["present"] == 1 and validate_box(gt["box"], frames[idx].shape)
        if gt["present"] == 1 and not gt_present:
            no_gt_frames += 1

        raw_ok, box = preds[idx]
        valid_pred = raw_ok and validate_box(box, frames[idx].shape)
        if gt_present:
            iou_v = iou(box, gt["box"]) if valid_pred else 0.0
        else:
            iou_v = 0.0

        # Frame 0 is initialization; strictly exclude frame 0 from all evaluation metrics
        if idx >= 1:
            if gt_present:
                eval_ious.append(iou_v)
                if not valid_pred:
                    failures += 1
                elif iou_v < 0.2:
                    drift_success_true += 1
            elif gt["present"] == 0:
                absent_total += 1
                if valid_pred:
                    absent_fp += 1

        rows.append({
            "sequence": spec.name,
            "frame": idx,
            "tracker": tracker_name,
            "binding": binding,
            "gt_present": int(gt_present),
            "success": int(raw_ok),
            "valid_pred": int(valid_pred),
            "iou": round(iou_v, 4),
            "box_x": round(box[0], 2),
            "box_y": round(box[1], 2),
            "box_w": round(box[2], 2),
            "box_h": round(box[3], 2),
            "gt_x": round(gt["box"][0], 2),
            "gt_y": round(gt["box"][1], 2),
            "gt_w": round(gt["box"][2], 2),
            "gt_h": round(gt["box"][3], 2),
        })

    # Recovery frames after occlusion: count frames from reappear_frame until 3 consecutive IoU >= 0.5
    recovery_str = "N/A"
    if spec.reappear_frame is not None:
        streak = 0
        found_at: Optional[int] = None
        for r in rows:
            if r["frame"] < spec.reappear_frame:
                continue
            if r["gt_present"] == 1 and r["iou"] >= IOU_THRESHOLD:
                streak += 1
                if streak >= RECOVERY_CONSECUTIVE:
                    found_at = r["frame"] - RECOVERY_CONSECUTIVE + 1 - spec.reappear_frame
                    break
            else:
                streak = 0
        recovery_str = str(found_at) if found_at is not None else "эргэж олоогүй"

    n_eval = len(eval_ious)
    mean_iou = sum(eval_ious) / n_eval if n_eval else 0.0
    iou_ge_05_pct = 100.0 * sum(1 for v in eval_ious if v >= IOU_THRESHOLD) / n_eval if n_eval else 0.0
    failure_pct = 100.0 * failures / n_eval if n_eval else 0.0
    absent_fp_pct = 100.0 * absent_fp / absent_total if absent_total else 0.0

    summary = {
        "sequence": spec.name,
        "resolution": f"{fw}x{fh}",
        "total_frames": len(frames),
        "eval_frames": n_eval,
        "absent_frames": absent_total,
        "no_gt_frames": no_gt_frames,
        "tracker": tracker_name,
        "binding": binding,
        "init_ok": int(init_ok),
        "mean_iou": round(mean_iou, 4),
        "iou_ge_05_pct": round(iou_ge_05_pct, 2),
        "failure_pct": round(failure_pct, 2),
        "drift_success_true": drift_success_true,
        "absent_fp_frames": absent_fp,
        "absent_fp_pct": round(absent_fp_pct, 2),
        "reappear_frame": spec.reappear_frame if spec.reappear_frame is not None else "-",
        "recovery_frames": recovery_str,
        "median_ms": round(median_ms, 3) if np.isfinite(median_ms) else "nan",
        "fps": round(fps, 2),
        "opencv_version": cv2.__version__,
    }
    return summary, rows


def main() -> None:
    enforce_single_thread()
    ensure_dirs()

    summaries: List[Dict] = []
    failure_examples: List[Dict] = []

    for spec in SEQUENCES:
        if not spec.gt_csv.exists():
            print(f"[SKIP] Missing GT: {spec.gt_csv}")
            continue
        print(f"=== Benchmarking tracking on {spec.name} ===")
        for tracker_name in ALL_TRACKERS:
            summary, rows = evaluate_tracker_on_sequence(spec, tracker_name)
            summaries.append(summary)

            per_csv = CSV_DIR / f"track_{spec.name}_{tracker_name}.csv"
            with open(per_csv, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)

            # Collect concrete failure / success=True drift examples
            drift_rows = [r for r in rows if r["frame"] >= 1 and r["gt_present"] == 1 and r["valid_pred"] == 1 and r["iou"] < 0.2]
            if drift_rows:
                first_drift = drift_rows[0]
                worst_drift = min(drift_rows, key=lambda r: r["iou"])
                failure_examples.append({
                    "sequence": spec.name,
                    "tracker": tracker_name,
                    "failure_type": "success=True_drift",
                    "first_frame": first_drift["frame"],
                    "first_iou": first_drift["iou"],
                    "worst_frame": worst_drift["frame"],
                    "worst_iou": worst_drift["iou"],
                    "drift_count": len(drift_rows),
                })

            print(
                f"  {tracker_name:10s} | meanIoU={summary['mean_iou']:.4f} "
                f"| IoU>=0.5={summary['iou_ge_05_pct']:6.2f}% "
                f"| fail={summary['failure_pct']:5.2f}% "
                f"| drift(T)={summary['drift_success_true']:2d} "
                f"| rec={summary['recovery_frames']:>13s} "
                f"| {summary['median_ms']:6.3f} ms ({summary['fps']:7.2f} FPS)"
            )

    summary_csv = CSV_DIR / "track_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        wr.writeheader()
        wr.writerows(summaries)

    fail_csv = CSV_DIR / "track_failures.csv"
    if failure_examples:
        with open(fail_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(failure_examples[0].keys()))
            wr.writeheader()
            wr.writerows(failure_examples)

    print("Saved tracking summary:", summary_csv)
    print("Saved failure examples:", fail_csv)


if __name__ == "__main__":
    main()
