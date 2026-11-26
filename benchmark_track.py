"""Single-thread object tracking benchmark: core, occlusion ladder, stress tests, ROI sensitivity, and LaSOT generalization."""

from __future__ import annotations

import csv
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    ALL_TRACKERS,
    CSV_DIR,
    DATA_DIR,
    IOU_THRESHOLD,
    RECOVERY_CONSECUTIVE,
    TIMING_REPEATS,
    WARMUP_RUNS,
    enforce_single_thread,
    ensure_dirs,
)
from metrics import compute_recovery_stats, iou, validate_box
from trackers import resolve_tracker_factory, safe_tracker_init


@dataclass
class SequenceSpec:
    name: str
    gt_csv: Path
    img_dir: Path
    reappear_frame: Optional[int]
    recovery_kind: str  # "full_occlusion", "heavy_occlusion", "none"
    description: str


CORE_SEQUENCES = [
    SequenceSpec(
        name="ctrl_bicycle_occ",
        gt_csv=CSV_DIR / "ctrl_bicycle_occ_gt.csv",
        img_dir=DATA_DIR / "controlled" / "bicycle_occ_seq",
        reappear_frame=23,
        recovery_kind="full_occlusion",
        description="CC0 photo controlled: moving target + scale + distractor + 5-frame full occlusion",
    ),
    SequenceSpec(
        name="ctrl_bicycle_cam_shift",
        gt_csv=CSV_DIR / "ctrl_bicycle_cam_shift_gt.csv",
        img_dir=DATA_DIR / "controlled" / "bicycle_cam_shift_seq",
        reappear_frame=None,
        recovery_kind="none",
        description="CC0 photo controlled: whole-frame camera pan shift (-6, -1 px/frame)",
    ),
    SequenceSpec(
        name="davis_bmx_trees_bike",
        gt_csv=CSV_DIR / "davis_bmx_trees_bike_gt.csv",
        img_dir=DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees",
        reappear_frame=62,
        recovery_kind="heavy_occlusion",
        description="DAVIS 2017 bmx-trees (obj_id=1 bike): visible-part GT, heavy tree occlusion",
    ),
    SequenceSpec(
        name="davis_bmx_trees_rider",
        gt_csv=CSV_DIR / "davis_bmx_trees_rider_gt.csv",
        img_dir=DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees",
        reappear_frame=69,
        recovery_kind="heavy_occlusion",
        description="DAVIS 2017 bmx-trees (obj_id=2 rider): visible-part GT, heavy occlusion & pan",
    ),
    SequenceSpec(
        name="davis_car_shadow",
        gt_csv=CSV_DIR / "davis_car_shadow_gt.csv",
        img_dir=DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "car-shadow",
        reappear_frame=None,
        recovery_kind="none",
        description="DAVIS 2017 car-shadow (obj_id=1 car): ~3.5x area shrink + shadow illumination change",
    ),
]

LADDER_SEQUENCES = [
    SequenceSpec("ctrl_occ_ladder_3f", CSV_DIR / "ctrl_occ_ladder_3f_gt.csv", DATA_DIR / "controlled" / "occ_ladder_3f", 29, "full_occlusion", "3-frame full occlusion"),
    SequenceSpec("ctrl_occ_ladder_10f", CSV_DIR / "ctrl_occ_ladder_10f_gt.csv", DATA_DIR / "controlled" / "occ_ladder_10f", 33, "full_occlusion", "11-frame full occlusion"),
    SequenceSpec("ctrl_occ_ladder_20f", CSV_DIR / "ctrl_occ_ladder_20f_gt.csv", DATA_DIR / "controlled" / "occ_ladder_20f", 39, "full_occlusion", "22-frame full occlusion"),
    SequenceSpec("ctrl_occ_ladder_10f_distractor", CSV_DIR / "ctrl_occ_ladder_10f_distractor_gt.csv", DATA_DIR / "controlled" / "occ_ladder_10f_distractor", 33, "full_occlusion", "11-frame full occlusion + similar bicycle distractor"),
]

LASOT_SEQUENCES = [
    SequenceSpec(
        name="lasot_bicycle11",
        gt_csv=CSV_DIR / "lasot_bicycle11_gt.csv",
        img_dir=DATA_DIR / "lasot" / "bicycle-11" / "img",
        reappear_frame=79,
        recovery_kind="full_occlusion",
        description="Held-out LaSOT bicycle-11 (120 frames, real full occlusion frames 74-78)",
    ),
    SequenceSpec(
        name="davis_scooter_black",
        gt_csv=CSV_DIR / "davis_scooter_black_gt.csv",
        img_dir=DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "scooter-black",
        reappear_frame=None,
        recovery_kind="none",
        description="Held-out DAVIS scooter-black (43 frames, visible-part GT)",
    ),
]


def load_sequence(spec: SequenceSpec) -> Tuple[List[Path], List[np.ndarray], List[Dict]]:
    """Load frame paths, BGR images, and GT rows (visible, optional amodal & distractor)."""
    img_paths: List[Path] = []
    frames: List[np.ndarray] = []
    gt_rows: List[Dict] = []
    with open(spec.gt_csv, newline="") as f:
        for row in csv.DictReader(f):
            p = spec.img_dir / row["image_file"]
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(f"Missing frame: {p}")
            img_paths.append(p)
            frames.append(img)
            present = int(row.get("present", 1))
            vis_box = (
                (float(row["vis_x"]), float(row["vis_y"]), float(row["vis_w"]), float(row["vis_h"]))
                if "vis_x" in row
                else (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
            )
            amodal_box = (
                (float(row["amodal_x"]), float(row["amodal_y"]), float(row["amodal_w"]), float(row["amodal_h"]))
                if "amodal_x" in row
                else None
            )
            dist_box = (
                (float(row["dist_x"]), float(row["dist_y"]), float(row["dist_w"]), float(row["dist_h"]))
                if "dist_x" in row
                else None
            )
            gt_rows.append({
                "frame": int(row["frame"]),
                "present": present,
                "box": vis_box,
                "amodal_box": amodal_box,
                "dist_box": dist_box,
            })
    return img_paths, frames, gt_rows


def _p95(values: List[float]) -> float:
    return float(np.percentile(values, 95)) if values else float("nan")


def _run_once_algo(
    tracker_name: str,
    frames: List[np.ndarray],
    init_roi: Tuple[int, int, int, int],
) -> Tuple[str, bool, List[Tuple[bool, Tuple[float, float, float, float]]], float, Dict]:
    prefer_legacy = tracker_name in ("BOOSTING", "TLD", "MEDIANFLOW")
    factory, binding = resolve_tracker_factory(tracker_name, prefer_legacy=prefer_legacy)
    if factory is None:
        return binding, False, [(False, (0.0, 0.0, 0.0, 0.0)) for _ in frames], float("nan"), {}

    tracker = factory()
    init_ok, _ = safe_tracker_init(tracker, frames[0], init_roi)
    if not init_ok:
        return binding, False, [(False, (0.0, 0.0, 0.0, 0.0)) for _ in frames], float("nan"), {}

    preds: List[Tuple[bool, Tuple[float, float, float, float]]] = [(True, tuple(float(v) for v in init_roi))]
    t0 = time.perf_counter()
    for frame in frames[1:]:
        ok, box = tracker.update(frame)
        preds.append((bool(ok), tuple(float(v) for v in box)))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_pf = elapsed_ms / max(1, len(frames) - 1)
    telemetry = {
        "redetect_calls": getattr(tracker, "redetect_calls", 0),
        "redetect_ms_total": getattr(tracker, "redetect_ms_total", 0.0),
        "reinits_count": getattr(tracker, "reinits_count", 0),
    }
    return binding, True, preds, ms_pf, telemetry


def _measure_e2e_ms(
    tracker_name: str,
    img_paths: List[Path],
    init_roi: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    """Measure end-to-end ms/frame: imread + update + overlay drawing + VideoWriter MP4."""
    prefer_legacy = tracker_name in ("BOOSTING", "TLD", "MEDIANFLOW")
    factory, _ = resolve_tracker_factory(tracker_name, prefer_legacy=prefer_legacy)
    if factory is None:
        return float("nan"), float("nan")

    times: List[float] = []
    e2e_repeats = min(3, TIMING_REPEATS)
    with tempfile.TemporaryDirectory() as tmp:
        out_mp4 = Path(tmp) / f"{tracker_name}.mp4"
        for run_idx in range(WARMUP_RUNS + e2e_repeats):
            t0 = time.perf_counter()
            f0 = cv2.imread(str(img_paths[0]))
            h, w = f0.shape[:2]
            writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
            tr = factory()
            safe_tracker_init(tr, f0, init_roi)
            cv2.rectangle(f0, (init_roi[0], init_roi[1]), (init_roi[0] + init_roi[2], init_roi[1] + init_roi[3]), (0, 255, 0), 2)
            writer.write(f0)
            for p in img_paths[1:]:
                fr = cv2.imread(str(p))
                ok, box = tr.update(fr)
                if ok:
                    bx, by, bw, bh = (int(round(v)) for v in box)
                    cv2.rectangle(fr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(fr, tracker_name, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                writer.write(fr)
            writer.release()
            dt = (time.perf_counter() - t0) * 1000.0 / len(img_paths)
            if run_idx >= WARMUP_RUNS:
                times.append(dt)
    return statistics.median(times), _p95(times)


def evaluate_tracker(
    spec: SequenceSpec,
    tracker_name: str,
    img_paths: List[Path],
    frames: List[np.ndarray],
    gt_rows: List[Dict],
    init_roi_override: Optional[Tuple[int, int, int, int]] = None,
    measure_e2e: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Run warmup + TIMING_REPEATS passes and compute complete tracking metrics."""
    fh, fw = frames[0].shape[:2]
    init_roi = init_roi_override or tuple(int(round(v)) for v in gt_rows[0]["box"])

    # Warmup (discarded)
    for _ in range(WARMUP_RUNS):
        _run_once_algo(tracker_name, frames, init_roi)

    algo_times: List[float] = []
    binding, init_ok, preds, telemetry = "", False, [], {}
    for _ in range(TIMING_REPEATS):
        binding, init_ok, preds, ms_pf, telemetry = _run_once_algo(tracker_name, frames, init_roi)
        if np.isfinite(ms_pf):
            algo_times.append(ms_pf)

    algo_med = statistics.median(algo_times) if algo_times else float("nan")
    algo_p95 = _p95(algo_times)
    algo_fps = 1000.0 / algo_med if algo_med > 0 and np.isfinite(algo_med) else 0.0

    if measure_e2e and img_paths:
        e2e_med, e2e_p95 = _measure_e2e_ms(tracker_name, img_paths, init_roi)
        e2e_fps = 1000.0 / e2e_med if e2e_med > 0 and np.isfinite(e2e_med) else 0.0
    else:
        e2e_med, e2e_p95, e2e_fps = float("nan"), float("nan"), 0.0

    rows: List[Dict] = []
    eval_ious_vis: List[float] = []
    eval_ious_amodal: List[float] = []
    failures = 0
    drift_success_true = 0
    absent_total = 0
    absent_fp = 0
    no_gt_frames = 0
    distractor_jumps = 0

    for idx in range(len(frames)):
        gt = gt_rows[idx]
        gt_present = gt["present"] == 1 and validate_box(gt["box"], frames[idx].shape)
        if gt["present"] == 1 and not gt_present:
            no_gt_frames += 1

        raw_ok, box = preds[idx]
        valid_pred = raw_ok and validate_box(box, frames[idx].shape)
        iou_vis = iou(box, gt["box"]) if (gt_present and valid_pred) else 0.0
        has_amodal = gt["amodal_box"] is not None and validate_box(gt["amodal_box"], frames[idx].shape)
        iou_amo = iou(box, gt["amodal_box"]) if (has_amodal and valid_pred) else 0.0

        dist_iou = iou(box, gt["dist_box"]) if (gt["dist_box"] is not None and valid_pred) else 0.0

        if idx >= 1:
            if gt_present:
                eval_ious_vis.append(iou_vis)
                if not valid_pred:
                    failures += 1
                elif iou_vis < 0.2:
                    drift_success_true += 1
            elif gt["present"] == 0:
                absent_total += 1
                if valid_pred:
                    absent_fp += 1
            if has_amodal:
                eval_ious_amodal.append(iou_amo)
            if valid_pred and dist_iou >= 0.30 and iou_vis < 0.10:
                distractor_jumps += 1

        rows.append({
            "sequence": spec.name,
            "frame": idx,
            "tracker": tracker_name,
            "binding": binding,
            "gt_present": int(gt_present),
            "success": int(raw_ok),
            "valid_pred": int(valid_pred),
            "iou": round(iou_vis, 4),
            "iou_amodal": round(iou_amo, 4) if has_amodal else "-",
            "box_x": round(box[0], 2),
            "box_y": round(box[1], 2),
            "box_w": round(box[2], 2),
            "box_h": round(box[3], 2),
            "gt_x": round(gt["box"][0], 2),
            "gt_y": round(gt["box"][1], 2),
            "gt_w": round(gt["box"][2], 2),
            "gt_h": round(gt["box"][3], 2),
        })

    rec_start, rec_conf, first_ok_f, conf_f = compute_recovery_stats(
        rows, spec.reappear_frame, IOU_THRESHOLD, RECOVERY_CONSECUTIVE
    )

    n_eval = len(eval_ious_vis)
    mean_iou = sum(eval_ious_vis) / n_eval if n_eval else 0.0
    mean_iou_amodal = sum(eval_ious_amodal) / len(eval_ious_amodal) if eval_ious_amodal else float("nan")
    iou_ge_05_pct = 100.0 * sum(1 for v in eval_ious_vis if v >= IOU_THRESHOLD) / n_eval if n_eval else 0.0
    failure_pct = 100.0 * failures / n_eval if n_eval else 0.0
    absent_fp_pct_str = f"{100.0 * absent_fp / absent_total:.2f}" if absent_total > 0 else "N/A (0 кадр)"

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
        "mean_iou_amodal": round(mean_iou_amodal, 4) if np.isfinite(mean_iou_amodal) else "-",
        "iou_ge_05_pct": round(iou_ge_05_pct, 2),
        "failure_pct": round(failure_pct, 2),
        "drift_success_true": drift_success_true,
        "distractor_jumps": distractor_jumps,
        "absent_fp_frames": absent_fp if absent_total > 0 else "N/A",
        "absent_fp_pct": absent_fp_pct_str,
        "recovery_type": spec.recovery_kind,
        "reappear_frame": spec.reappear_frame if spec.reappear_frame is not None else "-",
        "recovery_start_frames": str(rec_start) if rec_start is not None else ("N/A" if spec.reappear_frame is None else "эргэж олоогүй"),
        "recovery_confirmed_frames": str(rec_conf) if rec_conf is not None else ("N/A" if spec.reappear_frame is None else "эргэж олоогүй"),
        "first_ok_frame": str(first_ok_f) if first_ok_f is not None else "-",
        "confirmed_frame": str(conf_f) if conf_f is not None else "-",
        "redetect_calls": telemetry.get("redetect_calls", 0),
        "redetect_extra_ms": round(telemetry.get("redetect_ms_total", 0.0), 2),
        "algo_median_ms": round(algo_med, 3) if np.isfinite(algo_med) else "nan",
        "algo_p95_ms": round(algo_p95, 3) if np.isfinite(algo_p95) else "nan",
        "algo_fps": round(algo_fps, 2),
        "e2e_median_ms": round(e2e_med, 3) if np.isfinite(e2e_med) else "-",
        "e2e_p95_ms": round(e2e_p95, 3) if np.isfinite(e2e_p95) else "-",
        "e2e_fps": round(e2e_fps, 2) if np.isfinite(e2e_med) else "-",
        "opencv_version": cv2.__version__,
    }
    return summary, rows


def _apply_motion_blur(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    k = np.zeros((ksize, ksize), dtype=np.float32)
    k[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(img, -1, k)


def run_stress_and_roi_sensitivity() -> Tuple[List[Dict], List[Dict]]:
    """Run single-factor stress tests (stride, blur, darken) and initial ROI perturbations."""
    stress_rows: List[Dict] = []
    roi_rows: List[Dict] = []
    stress_trackers = ["MEDIANFLOW", "KCF", "MIL", "TLD", "CSRT", "CSRT_REDETECT"]

    # 1. Single-factor stress tests on davis_bmx_trees_bike & davis_car_shadow
    for spec in [CORE_SEQUENCES[2], CORE_SEQUENCES[4]]:
        if not spec.gt_csv.exists():
            continue
        _, frames, gt_rows = load_sequence(spec)
        variants: List[Tuple[str, List[np.ndarray], List[Dict]]] = [
            ("baseline_1x", frames, gt_rows),
            ("stride_2x", frames[::2], gt_rows[::2]),
            ("stride_3x", frames[::3], gt_rows[::3]),
            ("blur_gauss_k15", [cv2.GaussianBlur(fr, (15, 15), 0) for fr in frames], gt_rows),
            ("blur_motion_15", [_apply_motion_blur(fr, 15) for fr in frames], gt_rows),
            ("darken_05x", [cv2.convertScaleAbs(fr, alpha=0.5, beta=-25) for fr in frames], gt_rows),
        ]
        for cond, v_frames, v_gt in variants:
            v_spec = SequenceSpec(spec.name, spec.gt_csv, spec.img_dir, None, "none", cond)
            for tr in stress_trackers:
                summ, _ = evaluate_tracker(v_spec, tr, [], v_frames, v_gt, measure_e2e=False)
                stress_rows.append({
                    "sequence": spec.name,
                    "condition": cond,
                    "tracker": tr,
                    "eval_frames": summ["eval_frames"],
                    "mean_iou": summ["mean_iou"],
                    "iou_ge_05_pct": summ["iou_ge_05_pct"],
                    "failure_pct": summ["failure_pct"],
                })

    # 2. Initial ROI perturbation sensitivity on ctrl_bicycle_occ and davis_bmx_trees_bike
    for spec in [CORE_SEQUENCES[0], CORE_SEQUENCES[2]]:
        if not spec.gt_csv.exists():
            continue
        _, frames, gt_rows = load_sequence(spec)
        fh, fw = frames[0].shape[:2]
        x0, y0, w0, h0 = (float(v) for v in gt_rows[0]["box"])
        perturbations = [
            ("exact_roi", (int(round(x0)), int(round(y0)), int(round(w0)), int(round(h0)))),
            ("shift_10pct", (min(fw - 8, int(round(x0 + 0.1 * w0))), min(fh - 8, int(round(y0 + 0.1 * h0))), int(round(w0)), int(round(h0)))),
            ("scale_plus_10pct", (max(0, int(round(x0 - 0.05 * w0))), max(0, int(round(y0 - 0.05 * h0))), int(round(w0 * 1.1)), int(round(h0 * 1.1)))),
            ("scale_minus_10pct", (int(round(x0 + 0.05 * w0)), int(round(y0 + 0.05 * h0)), max(8, int(round(w0 * 0.9))), max(8, int(round(h0 * 0.9))))),
        ]
        base_ious: Dict[str, float] = {}
        for pname, proi in perturbations:
            for tr in stress_trackers:
                summ, _ = evaluate_tracker(spec, tr, [], frames, gt_rows, init_roi_override=proi, measure_e2e=False)
                miou = float(summ["mean_iou"])
                if pname == "exact_roi":
                    base_ious[tr] = miou
                roi_rows.append({
                    "sequence": spec.name,
                    "perturbation": pname,
                    "tracker": tr,
                    "mean_iou": miou,
                    "delta_iou": round(miou - base_ious[tr], 4),
                    "iou_ge_05_pct": summ["iou_ge_05_pct"],
                })
    return stress_rows, roi_rows


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def main() -> None:
    enforce_single_thread()
    ensure_dirs()

    core_summaries: List[Dict] = []
    failure_examples: List[Dict] = []

    print("=== 1. Core Suite (5 evaluation cases across 4 sequences x 8 trackers) ===")
    for spec in CORE_SEQUENCES:
        if not spec.gt_csv.exists():
            continue
        img_paths, frames, gt_rows = load_sequence(spec)
        print(f"--- {spec.name} ---")
        for tr in ALL_TRACKERS:
            summ, rows = evaluate_tracker(spec, tr, img_paths, frames, gt_rows, measure_e2e=True)
            core_summaries.append(summ)
            _write_csv(CSV_DIR / f"track_{spec.name}_{tr}.csv", rows)

            drift_rows = [r for r in rows if r["frame"] >= 1 and r["gt_present"] == 1 and r["valid_pred"] == 1 and float(r["iou"]) < 0.2]
            if drift_rows:
                first_d, worst_d = drift_rows[0], min(drift_rows, key=lambda r: float(r["iou"]))
                failure_examples.append({
                    "sequence": spec.name,
                    "tracker": tr,
                    "failure_type": "success=True_drift",
                    "first_frame": first_d["frame"],
                    "first_iou": first_d["iou"],
                    "worst_frame": worst_d["frame"],
                    "worst_iou": worst_d["iou"],
                    "drift_count": len(drift_rows),
                })
            print(
                f"  {tr:13s} | IoU={summ['mean_iou']:.4f} | >=0.5={summ['iou_ge_05_pct']:6.2f}% "
                f"| rec={summ['recovery_start_frames']}/{summ['recovery_confirmed_frames']} "
                f"| algo={summ['algo_median_ms']:6.2f}ms (p95={summ['algo_p95_ms']}) | e2e={summ['e2e_median_ms']}ms"
            )

    _write_csv(CSV_DIR / "track_summary.csv", core_summaries)
    _write_csv(CSV_DIR / "track_failures.csv", failure_examples)

    print("\n=== 2. Occlusion Ladder (3f, 10f, 20f, 10f+distractor) ===")
    ladder_summaries: List[Dict] = []
    for spec in LADDER_SEQUENCES:
        if not spec.gt_csv.exists():
            continue
        img_paths, frames, gt_rows = load_sequence(spec)
        for tr in ALL_TRACKERS:
            summ, rows = evaluate_tracker(spec, tr, img_paths, frames, gt_rows, measure_e2e=False)
            ladder_summaries.append(summ)
            _write_csv(CSV_DIR / f"track_{spec.name}_{tr}.csv", rows)
        print(f"  {spec.name}: CSRT={ladder_summaries[-2]['mean_iou']}, CSRT_REDETECT={ladder_summaries[-1]['mean_iou']} (rec={ladder_summaries[-1]['recovery_start_frames']}/{ladder_summaries[-1]['recovery_confirmed_frames']}, jumps={ladder_summaries[-1]['distractor_jumps']}), TLD={ladder_summaries[-4]['mean_iou']} (jumps={ladder_summaries[-4]['distractor_jumps']})")
    _write_csv(CSV_DIR / "track_occlusion_ladder.csv", ladder_summaries)

    print("\n=== 3. Stress Tests & ROI Perturbation Sensitivity ===")
    stress_rows, roi_rows = run_stress_and_roi_sensitivity()
    _write_csv(CSV_DIR / "track_stress_tests.csv", stress_rows)
    _write_csv(CSV_DIR / "track_roi_sensitivity.csv", roi_rows)
    print(f"  Saved {len(stress_rows)} stress rows and {len(roi_rows)} ROI sensitivity rows.")

    print("\n=== 4. Held-out Generalization (LaSOT bicycle-11 120f & DAVIS scooter-black 43f) ===")
    lasot_summaries: List[Dict] = []
    for spec in LASOT_SEQUENCES:
        if not spec.gt_csv.exists():
            continue
        img_paths, frames, gt_rows = load_sequence(spec)
        for tr in ALL_TRACKERS:
            summ, rows = evaluate_tracker(spec, tr, img_paths, frames, gt_rows, measure_e2e=True)
            lasot_summaries.append(summ)
            _write_csv(CSV_DIR / f"track_{spec.name}_{tr}.csv", rows)
            print(f"  {tr:13s} | IoU={summ['mean_iou']:.4f} | >=0.5={summ['iou_ge_05_pct']:6.2f}% | algo={summ['algo_median_ms']:6.2f}ms")
    _write_csv(CSV_DIR / "track_generalization_lasot.csv", lasot_summaries)


if __name__ == "__main__":
    main()
