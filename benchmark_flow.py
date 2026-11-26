"""Single-thread optical flow benchmark: common subset, regions, DIS vs Farneback, LK IDs/drift, camera compensation."""

from __future__ import annotations

import csv
import math
import statistics
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    CSV_DIR,
    DATA_DIR,
    FARNEBACK_PDF,
    FB_THRESH_PX,
    LK_PDF,
    LK_SMALL,
    SHI_TOMASI_PDF,
    TIMING_REPEATS,
    WARMUP_RUNS,
    enforce_single_thread,
    ensure_dirs,
)
from metrics import compute_flow_regions, epe, epe_map, read_flo
from opt_flow import draw_hsv_fixed


def _bilinear_sample_flow(flow: np.ndarray, x: float, y: float) -> Optional[Tuple[float, float]]:
    h, w = flow.shape[:2]
    if not (0.0 <= x <= w - 1 and 0.0 <= y <= h - 1):
        return None
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(w - 1, x0 + 1), min(h - 1, y0 + 1)
    dx, dy = x - x0, y - y0
    v00, v10, v01, v11 = flow[y0, x0], flow[y0, x1], flow[y1, x0], flow[y1, x1]
    u = (1 - dx) * (1 - dy) * v00[0] + dx * (1 - dy) * v10[0] + (1 - dx) * dy * v01[0] + dx * dy * v11[0]
    v = (1 - dx) * (1 - dy) * v00[1] + dx * (1 - dy) * v10[1] + (1 - dx) * dy * v01[1] + dx * dy * v11[1]
    if not (math.isfinite(float(u)) and math.isfinite(float(v))):
        return None
    return (float(u), float(v))


def _p95(vals: List[float]) -> float:
    return float(np.percentile(vals, 95)) if vals else float("nan")


def _time_fn(fn) -> Tuple[object, float, float]:
    """Run WARMUP_RUNS + TIMING_REPEATS and return (last_result, median_ms, p95_ms)."""
    res = None
    for _ in range(WARMUP_RUNS):
        res = fn()
    times: List[float] = []
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        res = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return res, statistics.median(times), _p95(times)


def _time_dense_e2e(p0: Path, p1: Path, flow_fn) -> Tuple[float, float]:
    """Measure end-to-end ms/frame: imread x2 + gray + dense flow + HSV render + VideoWriter."""
    times: List[float] = []
    e2e_repeats = min(3, TIMING_REPEATS)
    with tempfile.TemporaryDirectory() as tmp:
        out_mp4 = Path(tmp) / "flow_e2e.mp4"
        for run_idx in range(WARMUP_RUNS + e2e_repeats):
            t0 = time.perf_counter()
            im0, im1 = cv2.imread(str(p0)), cv2.imread(str(p1))
            g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            fl = flow_fn(g0, g1)
            vis = draw_hsv_fixed(fl, max_mag=12.0)
            h, w = vis.shape[:2]
            wr = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
            wr.write(vis)
            wr.release()
            dt = (time.perf_counter() - t0) * 1000.0
            if run_idx >= WARMUP_RUNS:
                times.append(dt)
    return statistics.median(times), _p95(times)


def _load_flow_pairs() -> List[Tuple[str, Path, Path, np.ndarray, np.ndarray]]:
    """Load Middlebury, Controlled CC0, and held-out MPI Sintel image pairs with valid GT masks."""
    pairs = []
    for seq in ("RubberWhale", "Grove2"):
        p0 = DATA_DIR / "middlebury" / "other-data" / seq / "frame10.png"
        p1 = DATA_DIR / "middlebury" / "other-data" / seq / "frame11.png"
        flo, valid = read_flo(DATA_DIR / "middlebury" / "other-gt-flow" / seq / "flow10.flo")
        pairs.append((f"Middlebury:{seq}", p0, p1, flo, valid))

    for seq in ("obj_occ", "cam_shift", "cam_plus_obj"):
        p0 = DATA_DIR / "controlled" / "flow_pairs" / f"{seq}_f0.png"
        p1 = DATA_DIR / "controlled" / "flow_pairs" / f"{seq}_f1.png"
        flo, valid = read_flo(DATA_DIR / "controlled" / "flow_pairs" / f"{seq}_gt.flo")
        pairs.append((f"Controlled:{seq}", p0, p1, flo, valid))

    sintel_specs = [
        ("Sintel:alley_1_clean", "clean", "alley_1", "0001", "0002"),
        ("Sintel:ambush_5_final", "final", "ambush_5", "0010", "0011"),
    ]
    for label, ptype, seq, f0s, f1s in sintel_specs:
        p0 = DATA_DIR / "sintel" / "training" / ptype / seq / f"frame_{f0s}.png"
        p1 = DATA_DIR / "sintel" / "training" / ptype / seq / f"frame_{f1s}.png"
        flo_p = DATA_DIR / "sintel" / "training" / "flow" / seq / f"frame_{f0s}.flo"
        occ_p = DATA_DIR / "sintel" / "training" / "occlusions" / seq / f"frame_{f0s}.png"
        inv_p = DATA_DIR / "sintel" / "training" / "invalid" / seq / f"frame_{f0s}.png"
        if p0.exists() and flo_p.exists():
            flo, valid = read_flo(flo_p)
            if occ_p.exists():
                valid &= cv2.imread(str(occ_p), cv2.IMREAD_GRAYSCALE) < 128
            if inv_p.exists():
                valid &= cv2.imread(str(inv_p), cv2.IMREAD_GRAYSCALE) < 128
            pairs.append((label, p0, p1, flo, valid))
    return pairs


def eval_pair_common_and_dense(
    pair_name: str,
    p0: Path,
    p1: Path,
    gt_flow: np.ndarray,
    gt_valid: np.ndarray,
    max_corners: int = 10,
) -> Tuple[List[Dict], List[Dict]]:
    """Evaluate LK (win200, win15), Farneback, and DIS presets on common points & dense regions."""
    im0, im1 = cv2.imread(str(p0)), cv2.imread(str(p1))
    g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    h, w = g0.shape

    corner_cfg = dict(SHI_TOMASI_PDF, maxCorners=max_corners)
    pts0 = cv2.goodFeaturesToTrack(g0, mask=None, **corner_cfg)
    if pts0 is None:
        pts0 = np.empty((0, 1, 2), dtype=np.float32)

    init_pts = int(len(pts0))
    gt_ok_mask = np.zeros(init_pts, dtype=bool)
    for j in range(init_pts):
        ix, iy = int(round(float(pts0[j, 0, 0]))), int(round(float(pts0[j, 0, 1])))
        if 0 <= ix < w and 0 <= iy < h and bool(gt_valid[iy, ix]):
            gt_ok_mask[j] = True
    gt_valid_pts = int(gt_ok_mask.sum())

    # Run both LK configurations to determine their individual valid masks AND their intersection
    lk_outputs = {}
    for label, cfg in [("LK_PDF_win200", LK_PDF), ("LK_small_win15", LK_SMALL)]:
        (nxt, st, _), med_ms, p95_ms = _time_fn(lambda c=cfg: cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, **c))
        st_flat = st.ravel() == 1 if st is not None else np.zeros(init_pts, dtype=bool)
        in_b = np.zeros(init_pts, dtype=bool)
        if nxt is not None:
            for j in range(init_pts):
                nx, ny = float(nxt[j, 0, 0]), float(nxt[j, 0, 1])
                if math.isfinite(nx) and math.isfinite(ny) and 0.0 <= nx < w and 0.0 <= ny < h:
                    in_b[j] = True
        lk_valid = gt_ok_mask & st_flat & in_b
        lk_outputs[label] = (nxt, lk_valid, med_ms, p95_ms)

    common_mask = lk_outputs["LK_PDF_win200"][1] & lk_outputs["LK_small_win15"][1]
    common_pts = int(common_mask.sum())

    # Dense algorithms: Farneback + DIS (ULTRAFAST, FAST, MEDIUM)
    dense_specs = [
        ("Farneback_dense", lambda a, b: cv2.calcOpticalFlowFarneback(a, b, None, **FARNEBACK_PDF)),
        ("DIS_ULTRAFAST", lambda a, b: cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST).calc(a, b, None)),
        ("DIS_FAST", lambda a, b: cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(a, b, None)),
        ("DIS_MEDIUM", lambda a, b: cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a, b, None)),
    ]

    dense_flows = {}
    dense_timings = {}
    for dlabel, dfn in dense_specs:
        flow_res, med_ms, p95_ms = _time_fn(lambda fn=dfn: fn(g0, g1))
        e2e_med, e2e_p95 = _time_dense_e2e(p0, p1, dfn) if max_corners == 10 else (float("nan"), float("nan"))
        dense_flows[dlabel] = flow_res
        dense_timings[dlabel] = (med_ms, p95_ms, e2e_med, e2e_p95)

    summary_rows: List[Dict] = []

    def _pt_epe_for_mask(vecs_from: str, mask_arr: np.ndarray) -> float:
        errs = []
        for j in range(init_pts):
            if not mask_arr[j]:
                continue
            px, py = float(pts0[j, 0, 0]), float(pts0[j, 0, 1])
            ix, iy = int(round(px)), int(round(py))
            gv = (float(gt_flow[iy, ix, 0]), float(gt_flow[iy, ix, 1]))
            if vecs_from in lk_outputs:
                nxt_arr = lk_outputs[vecs_from][0]
                pv = (float(nxt_arr[j, 0, 0]) - px, float(nxt_arr[j, 0, 1]) - py)
            else:
                fl_arr = dense_flows[vecs_from]
                pv = _bilinear_sample_flow(fl_arr, px, py) or (float(fl_arr[iy, ix, 0]), float(fl_arr[iy, ix, 1]))
            errs.append(epe(pv, gv))
        return sum(errs) / len(errs) if errs else float("nan")

    for label in ("LK_PDF_win200", "LK_small_win15"):
        _, lmask, med_ms, p95_ms = lk_outputs[label]
        own_pts = int(lmask.sum())
        lost_pts = gt_valid_pts - own_pts
        loss_pct = 100.0 * lost_pts / gt_valid_pts if gt_valid_pts else 0.0
        common_epe = _pt_epe_for_mask(label, common_mask)
        own_epe = _pt_epe_for_mask(label, lmask)
        summary_rows.append({
            "dataset": pair_name,
            "resolution": f"{w}x{h}",
            "max_corners": max_corners,
            "method": label,
            "init_pts": init_pts,
            "gt_valid_pts": gt_valid_pts,
            "own_valid_pts": own_pts,
            "lost_pts": lost_pts,
            "point_loss_pct": round(loss_pct, 2),
            "common_pts": common_pts,
            "common_pts_epe": round(common_epe, 4) if np.isfinite(common_epe) else "-",
            "own_pts_epe": round(own_epe, 4) if np.isfinite(own_epe) else "-",
            "all_valid_px_epe": "-",
            "valid_px_count": "-",
            "algo_median_ms": round(med_ms, 3),
            "algo_p95_ms": round(p95_ms, 3),
            "algo_fps": round(1000.0 / med_ms, 2) if med_ms > 0 else 0.0,
            "e2e_median_ms": "-",
            "e2e_p95_ms": "-",
        })

    # Dense models evaluated on exact same `common_pts` AND full dense valid pixels
    regions = compute_flow_regions(im0, gt_flow, gt_valid) if max_corners == 10 else {}
    region_rows: List[Dict] = []

    for dlabel, _ in dense_specs:
        fl = dense_flows[dlabel]
        med_ms, p95_ms, e2e_med, e2e_p95 = dense_timings[dlabel]
        common_epe = _pt_epe_for_mask(dlabel, common_mask)
        all_epe, all_cnt = epe_map(fl, gt_flow, gt_valid)
        summary_rows.append({
            "dataset": pair_name,
            "resolution": f"{w}x{h}",
            "max_corners": max_corners,
            "method": dlabel,
            "init_pts": init_pts,
            "gt_valid_pts": gt_valid_pts,
            "own_valid_pts": gt_valid_pts,
            "lost_pts": 0,
            "point_loss_pct": 0.0,
            "common_pts": common_pts,
            "common_pts_epe": round(common_epe, 4) if np.isfinite(common_epe) else "-",
            "own_pts_epe": round(common_epe, 4) if np.isfinite(common_epe) else "-",
            "all_valid_px_epe": round(all_epe, 4),
            "valid_px_count": all_cnt,
            "algo_median_ms": round(med_ms, 3),
            "algo_p95_ms": round(p95_ms, 3),
            "algo_fps": round(1000.0 / med_ms, 2) if med_ms > 0 else 0.0,
            "e2e_median_ms": round(e2e_med, 3) if np.isfinite(e2e_med) else "-",
            "e2e_p95_ms": round(e2e_p95, 3) if np.isfinite(e2e_p95) else "-",
        })

        if regions:
            reg_entry = {
                "dataset": pair_name,
                "method": dlabel,
                "algo_median_ms": round(med_ms, 3),
                "algo_p95_ms": round(p95_ms, 3),
                "algo_fps": round(1000.0 / med_ms, 2) if med_ms > 0 else 0.0,
                "e2e_median_ms": round(e2e_med, 3) if np.isfinite(e2e_med) else "-",
            }
            for rname, rmask in regions.items():
                re, rc = epe_map(fl, gt_flow, rmask)
                reg_entry[f"{rname}_epe"] = round(re, 4) if np.isfinite(re) else "-"
                reg_entry[f"{rname}_px"] = rc
            region_rows.append(reg_entry)

    return summary_rows, region_rows


def _run_seq_lk_id_ablation(
    seq_name: str,
    frames: List[np.ndarray],
    gt_vec: Optional[Tuple[float, float]],
    lk_cfg: Dict,
    use_fb: bool,
    redetect_interval: int,
    variant_name: str,
) -> Tuple[Dict, List[Dict]]:
    """Run LK ablation tracking per-point IDs (`initial` vs `redetected`) & cumulative drift."""
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    h, w = g0.shape
    pts0 = cv2.goodFeaturesToTrack(g0, mask=None, **SHI_TOMASI_PDF)
    if pts0 is None:
        pts0 = np.empty((0, 1, 2), dtype=np.float32)

    def run_pass():
        prev_gray = g0
        # Each point: (pt_id, origin_type, start_x, start_y, start_frame, cur_x, cur_y)
        tracks = [
            (j, "initial", float(p[0]), float(p[1]), 0, float(p[0]), float(p[1]))
            for j, p in enumerate(pts0.reshape(-1, 2))
        ]
        next_id = len(tracks)
        init_n = len(tracks)
        step_epes: List[float] = []
        drift_log: List[Dict] = []
        t_algo = 0.0

        for idx in range(1, len(frames)):
            cur_gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            t0 = time.perf_counter()

            if tracks:
                p0 = np.float32([(t[5], t[6]) for t in tracks]).reshape(-1, 1, 2)
                p1, st_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, **lk_cfg)
                st_mask = st_f.ravel() == 1 if st_f is not None else np.zeros(len(tracks), dtype=bool)
                if use_fb and p1 is not None:
                    p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, None, **lk_cfg)
                    fb_err = np.abs(p0 - p0r).reshape(-1, 2).max(-1)
                    st_mask = st_mask & (st_b.ravel() == 1) & (fb_err < FB_THRESH_PX)

                new_tracks = []
                for j, (tr, ok) in enumerate(zip(tracks, st_mask)):
                    if not ok or p1 is None:
                        continue
                    nx, ny = float(p1[j, 0, 0]), float(p1[j, 0, 1])
                    if not (math.isfinite(nx) and math.isfinite(ny) and 0.0 <= nx < w and 0.0 <= ny < h):
                        continue
                    pid, orig, sx, sy, sf, px, py = tr
                    new_tracks.append((pid, orig, sx, sy, sf, nx, ny))
                    if gt_vec is not None:
                        step_epes.append(epe((nx - px, ny - py), gt_vec))
                tracks = new_tracks

            if redetect_interval > 0 and idx % redetect_interval == 0:
                mask = np.full_like(cur_gray, 255)
                for tr in tracks:
                    cv2.circle(mask, (int(round(tr[5])), int(round(tr[6]))), 5, 0, -1)
                extra = cv2.goodFeaturesToTrack(cur_gray, mask=mask, **SHI_TOMASI_PDF)
                if extra is not None:
                    for ex, ey in extra.reshape(-1, 2):
                        if len(tracks) < SHI_TOMASI_PDF["maxCorners"]:
                            tracks.append((next_id, "redetected", float(ex), float(ey), idx, float(ex), float(ey)))
                            next_id += 1

            t_algo += (time.perf_counter() - t0) * 1000.0
            if gt_vec is not None:
                init_drifts = [
                    epe((tr[5], tr[6]), (tr[2] + (idx - tr[4]) * gt_vec[0], tr[3] + (idx - tr[4]) * gt_vec[1]))
                    for tr in tracks if tr[1] == "initial"
                ]
                drift_log.append({
                    "sequence": seq_name,
                    "variant": variant_name,
                    "frame": idx,
                    "retained_initial_pts": len(init_drifts),
                    "total_active_pts": len(tracks),
                    "mean_cum_drift_px": round(sum(init_drifts) / len(init_drifts), 4) if init_drifts else float("nan"),
                })
            prev_gray = cur_gray

        retained_init = sum(1 for tr in tracks if tr[1] == "initial")
        redet_pts = sum(1 for tr in tracks if tr[1] == "redetected")
        return init_n, retained_init, redet_pts, len(tracks), step_epes, drift_log, t_algo / max(1, len(frames) - 1)

    for _ in range(WARMUP_RUNS):
        run_pass()

    times = []
    last_out = None
    for _ in range(TIMING_REPEATS):
        last_out = run_pass()
        times.append(last_out[6])

    init_n, retained_init, redet_pts, final_active, step_epes, drift_log, _ = last_out
    med_ms, p95_ms = statistics.median(times), _p95(times)
    final_cum_drift = drift_log[-1]["mean_cum_drift_px"] if drift_log else "-"

    summary = {
        "sequence": seq_name,
        "variant": variant_name,
        "winSize": str(lk_cfg["winSize"]),
        "fb_check": int(use_fb),
        "redetect_interval": redetect_interval,
        "init_pts": init_n,
        "retained_initial_pts": retained_init,
        "initial_retention_pct": round(100.0 * retained_init / init_n, 2) if init_n else 0.0,
        "redetected_pts": redet_pts,
        "final_active_pts": final_active,
        "step_mean_epe_px": round(sum(step_epes) / len(step_epes), 4) if (gt_vec is not None and step_epes) else "-",
        "final_cum_drift_px": final_cum_drift,
        "algo_median_ms": round(med_ms, 3),
        "algo_p95_ms": round(p95_ms, 3),
        "algo_fps": round(1000.0 / med_ms, 2) if med_ms > 0 else 0.0,
    }
    return summary, drift_log


def _detect_stratified_corners(gray: np.ndarray, rows: int = 4, cols: int = 6, per_cell: int = 8) -> np.ndarray:
    """Detect Shi-Tomasi corners evenly across spatial tiles so small moving foreground cannot dominate."""
    h, w = gray.shape
    pts_all: List[np.ndarray] = []
    ch, cw = max(16, h // rows), max(16, w // cols)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * ch, h if r == rows - 1 else (r + 1) * ch
            x0, x1 = c * cw, w if c == cols - 1 else (c + 1) * cw
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255
            p = cv2.goodFeaturesToTrack(gray, maxCorners=per_cell, qualityLevel=0.03, minDistance=10, mask=mask, blockSize=7)
            if p is not None and len(p) > 0:
                pts_all.append(p)
    if not pts_all:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.vstack(pts_all).astype(np.float32)


def compute_camera_motion_compensation(
    img0: np.ndarray,
    img1: np.ndarray,
    flow_method: str = "dis_medium",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate global camera affine flow from spatially stratified FB-filtered LK + RANSAC and subtract from dense flow."""
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    h, w = g0.shape
    if flow_method == "farneback":
        raw_flow = cv2.calcOpticalFlowFarneback(g0, g1, None, **FARNEBACK_PDF)
    else:
        raw_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g0, g1, None)

    pts0 = _detect_stratified_corners(g0)
    cam_flow = np.zeros_like(raw_flow)
    lk_bg = dict(winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    if len(pts0) >= 8:
        pts1, st_f, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, **lk_bg)
        pts0r, st_b, _ = cv2.calcOpticalFlowPyrLK(g1, g0, pts1, None, **lk_bg)
        fb = np.linalg.norm(pts0.reshape(-1, 2) - pts0r.reshape(-1, 2), axis=1)
        ok = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb <= 1.5)
        if ok.sum() >= 6:
            p0 = pts0[ok].reshape(-1, 2)
            p1 = pts1[ok].reshape(-1, 2)
            M, _inliers = cv2.estimateAffine2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000, confidence=0.995)
            if M is not None:
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                warped_x = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]
                warped_y = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]
                cam_flow[..., 0] = warped_x - xx
                cam_flow[..., 1] = warped_y - yy
    residual = raw_flow - cam_flow
    return raw_flow, cam_flow, residual


def run_camera_compensation_eval() -> List[Dict]:
    """Quantify camera ego-motion suppression on Farneback vs DIS_MEDIUM across 3 pairs."""
    from PIL import Image

    rows: List[Dict] = []

    f_cam0 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / "cam_shift_f0.png"))
    f_cam1 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / "cam_shift_f1.png"))
    f_mix0 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / "cam_plus_obj_f0.png"))
    f_mix1 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / "cam_plus_obj_f1.png"))
    mix_fg = np.zeros(f_mix0.shape[:2], dtype=bool)
    mix_fg[165:315, 185:265] = True

    bmx0 = cv2.imread(str(DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees" / "00010.jpg"))
    bmx1 = cv2.imread(str(DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees" / "00011.jpg"))
    msk_p = DATA_DIR / "davis" / "DAVIS" / "Annotations" / "480p" / "bmx-trees" / "00010.png"
    bmx_fg = (np.array(Image.open(msk_p)) > 0) if msk_p.exists() else None

    for flow_method in ("farneback", "dis_medium"):
        # 1. Pure camera shift
        raw, cam, res = compute_camera_motion_compensation(f_cam0, f_cam1, flow_method=flow_method)
        raw_m = float(np.hypot(raw[..., 0], raw[..., 1]).mean())
        res_m = float(np.hypot(res[..., 0], res[..., 1]).mean())
        rows.append({
            "pair": "Controlled:cam_shift",
            "flow_method": flow_method,
            "raw_bg_mag": round(raw_m, 4),
            "est_cam_mag": round(float(np.hypot(cam[..., 0], cam[..., 1]).mean()), 4),
            "residual_bg_mag": round(res_m, 4),
            "bg_reduction_pct": round(100.0 * (raw_m - res_m) / max(1e-6, raw_m), 2),
            "raw_fg_mag": "-",
            "residual_fg_mag": "-",
        })

        # 2. Pan + moving object
        raw, cam, res = compute_camera_motion_compensation(f_mix0, f_mix1, flow_method=flow_method)
        bg_mask = ~mix_fg
        raw_bg = float(np.hypot(raw[bg_mask, 0], raw[bg_mask, 1]).mean())
        res_bg = float(np.hypot(res[bg_mask, 0], res[bg_mask, 1]).mean())
        raw_fg = float(np.hypot(raw[mix_fg, 0], raw[mix_fg, 1]).mean())
        res_fg = float(np.hypot(res[mix_fg, 0], res[mix_fg, 1]).mean())
        rows.append({
            "pair": "Controlled:cam_plus_obj",
            "flow_method": flow_method,
            "raw_bg_mag": round(raw_bg, 4),
            "est_cam_mag": round(float(np.hypot(cam[bg_mask, 0], cam[bg_mask, 1]).mean()), 4),
            "residual_bg_mag": round(res_bg, 4),
            "bg_reduction_pct": round(100.0 * (raw_bg - res_bg) / max(1e-6, raw_bg), 2),
            "raw_fg_mag": round(raw_fg, 4),
            "residual_fg_mag": round(res_fg, 4),
        })

        # 3. DAVIS bmx-trees frames 10->11
        if bmx0 is not None and bmx_fg is not None:
            raw, cam, res = compute_camera_motion_compensation(bmx0, bmx1, flow_method=flow_method)
            bg = ~bmx_fg
            raw_bg = float(np.hypot(raw[bg, 0], raw[bg, 1]).mean())
            res_bg = float(np.hypot(res[bg, 0], res[bg, 1]).mean())
            raw_fg = float(np.hypot(raw[bmx_fg, 0], raw[bmx_fg, 1]).mean())
            res_fg = float(np.hypot(res[bmx_fg, 0], res[bmx_fg, 1]).mean())
            rows.append({
                "pair": "DAVIS:bmx_trees_f10",
                "flow_method": flow_method,
                "raw_bg_mag": round(raw_bg, 4),
                "est_cam_mag": round(float(np.hypot(cam[bg, 0], cam[bg, 1]).mean()), 4),
                "residual_bg_mag": round(res_bg, 4),
                "bg_reduction_pct": round(100.0 * (raw_bg - res_bg) / max(1e-6, raw_bg), 2),
                "raw_fg_mag": round(raw_fg, 4),
                "residual_fg_mag": round(res_fg, 4),
            })
    return rows


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

    pairs = _load_flow_pairs()
    summary_all: List[Dict] = []
    region_all: List[Dict] = []

    print("=== 1. Optical Flow Common-Subset & DIS vs Farneback Benchmark ===")
    for pair_name, p0, p1, flo, valid in pairs:
        for mc in (10, 100):
            s_rows, r_rows = eval_pair_common_and_dense(pair_name, p0, p1, flo, valid, max_corners=mc)
            summary_all.extend(s_rows)
            region_all.extend(r_rows)
            if mc == 10:
                for r in s_rows:
                    print(
                        f"  {pair_name:23s} | {r['method']:15s} | common={r['common_pts']:2d}/{r['gt_valid_pts']:2d} "
                        f"| loss={r['point_loss_pct']:5.1f}% | EPE_common={str(r['common_pts_epe']):>7s} "
                        f"| EPE_all={str(r['all_valid_px_epe']):>7s} | algo={r['algo_median_ms']:6.3f}ms"
                    )

    _write_csv(CSV_DIR / "flow_summary.csv", summary_all)
    _write_csv(CSV_DIR / "flow_dense_regions.csv", region_all)

    cam_frames = [cv2.imread(str(DATA_DIR / "controlled" / "bicycle_cam_shift_seq" / f"frame_{i:04d}.png")) for i in range(24)]
    occ_frames = [cv2.imread(str(DATA_DIR / "controlled" / "bicycle_occ_seq" / f"frame_{i:04d}.png")) for i in range(40)]
    davis_paths = sorted((DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees").glob("*.jpg"))
    davis_frames = [cv2.imread(str(p)) for p in davis_paths]

    ablations: List[Dict] = []
    drift_logs: List[Dict] = []
    configs = [
        ("LK_PDF_win200_noFB", LK_PDF, False, 0),
        ("LK_PDF_win200_FB", LK_PDF, True, 0),
        ("LK_PDF_win200_redetect5", LK_PDF, False, 5),
        ("LK_PDF_win200_FB_redetect5", LK_PDF, True, 5),
        ("LK_small_win15_noFB", LK_SMALL, False, 0),
        ("LK_small_win15_FB", LK_SMALL, True, 0),
    ]
    print("\n=== 2. LK Point-ID Retention & Cumulative Trajectory Drift ===")
    for seq_name, frames, gt_v in [
        ("Controlled:cam_shift_24f", cam_frames, (-6.0, -1.0)),
        ("Controlled:bicycle_occ_40f", occ_frames, None),
        ("DAVIS:bmx_trees_80f", davis_frames, None),
    ]:
        for vname, cfg, fb_flag, redet in configs:
            srow, dlog = _run_seq_lk_id_ablation(seq_name, frames, gt_v, cfg, fb_flag, redet, vname)
            ablations.append(srow)
            drift_logs.extend(dlog)
            print(
                f"  {seq_name:24s} | {vname:27s} | init_kept={srow['retained_initial_pts']:2d}/{srow['init_pts']:2d} "
                f"| redet={srow['redetected_pts']:2d} | active={srow['final_active_pts']:2d} | drift={srow['final_cum_drift_px']}"
            )

    _write_csv(CSV_DIR / "flow_ablation.csv", ablations)
    _write_csv(CSV_DIR / "flow_trajectory_drift.csv", drift_logs)

    print("\n=== 3. Camera Motion Compensation ===")
    comp_rows = run_camera_compensation_eval()
    _write_csv(CSV_DIR / "camera_compensation.csv", comp_rows)
    for r in comp_rows:
        print(f"  {r['pair']:23s} | raw_bg={r['raw_bg_mag']:.3f} -> res_bg={r['residual_bg_mag']:.3f} (-{r['bg_reduction_pct']}%) | fg={r['raw_fg_mag']}->{r['residual_fg_mag']}")


if __name__ == "__main__":
    main()
