"""Single-thread, multi-repeat optical flow benchmark and ablation experiments."""

from __future__ import annotations

import csv
import math
import statistics
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
    enforce_single_thread,
    ensure_dirs,
)
from metrics import epe, epe_map, read_flo


def _bilinear_sample_flow(flow: np.ndarray, x: float, y: float) -> Optional[Tuple[float, float]]:
    """Sample 2D vector at subpixel coordinate (x, y) via bilinear interpolation."""
    h, w = flow.shape[:2]
    if not (0.0 <= x <= w - 1 and 0.0 <= y <= h - 1):
        return None
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(w - 1, x0 + 1)
    y1 = min(h - 1, y0 + 1)
    dx = x - x0
    dy = y - y0
    v00 = flow[y0, x0]
    v10 = flow[y0, x1]
    v01 = flow[y1, x0]
    v11 = flow[y1, x1]
    u = (1 - dx) * (1 - dy) * v00[0] + dx * (1 - dy) * v10[0] + (1 - dx) * dy * v01[0] + dx * dy * v11[0]
    v = (1 - dx) * (1 - dy) * v00[1] + dx * (1 - dy) * v10[1] + (1 - dx) * dy * v01[1] + dx * dy * v11[1]
    if not (math.isfinite(float(u)) and math.isfinite(float(v))):
        return None
    return (float(u), float(v))


def _load_flow_pairs() -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load Middlebury and controlled CC0 two-frame pairs with GT flow and valid masks."""
    pairs = []
    for seq in ("RubberWhale", "Grove2"):
        img0 = cv2.imread(str(DATA_DIR / "middlebury" / "other-data" / seq / "frame10.png"))
        img1 = cv2.imread(str(DATA_DIR / "middlebury" / "other-data" / seq / "frame11.png"))
        flo, valid = read_flo(DATA_DIR / "middlebury" / "other-gt-flow" / seq / "flow10.flo")
        pairs.append((f"Middlebury:{seq}", img0, img1, flo, valid))

    for seq in ("obj_occ", "cam_shift"):
        img0 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / f"{seq}_f0.png"))
        img1 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / f"{seq}_f1.png"))
        flo, valid = read_flo(DATA_DIR / "controlled" / "flow_pairs" / f"{seq}_gt.flo")
        pairs.append((f"Controlled:{seq}", img0, img1, flo, valid))
    return pairs


def _time_fn(fn) -> Tuple[object, float]:
    """Run fn() TIMING_REPEATS times and return (last_result, median_ms)."""
    times = []
    res = None
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        res = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return res, statistics.median(times)


def eval_pair_lk_vs_farneback(
    pair_name: str,
    img0: np.ndarray,
    img1: np.ndarray,
    gt_flow: np.ndarray,
    gt_valid: np.ndarray,
    max_corners: int = 10,
) -> Tuple[List[Dict], List[Dict]]:
    """Compare LK (winSize=200 and winSize=15) and Farneback at identical Shi-Tomasi points."""
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    h, w = g0.shape

    corner_cfg = dict(SHI_TOMASI_PDF, maxCorners=max_corners)
    pts0 = cv2.goodFeaturesToTrack(g0, mask=None, **corner_cfg)
    if pts0 is None:
        pts0 = np.empty((0, 1, 2), dtype=np.float32)

    init_pts = int(len(pts0))
    gt_ok_mask = np.zeros(init_pts, dtype=bool)
    for j in range(init_pts):
        x, y = float(pts0[j, 0, 0]), float(pts0[j, 0, 1])
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h and bool(gt_valid[iy, ix]):
            gt_ok_mask[j] = True
    gt_valid_pts = int(gt_ok_mask.sum())

    # Run Farneback once (timed)
    fb_flow, fb_ms = _time_fn(lambda: cv2.calcOpticalFlowFarneback(g0, g1, None, **FARNEBACK_PDF))
    fb_all_epe, fb_valid_px = epe_map(fb_flow, gt_flow, gt_valid)

    summary_rows: List[Dict] = []
    point_rows: List[Dict] = []

    for label, lk_cfg in [("LK_PDF_win200", LK_PDF), ("LK_small_win15", LK_SMALL)]:
        (nxt, st, _err), lk_ms = _time_fn(lambda cfg=lk_cfg: cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, **cfg))
        st_flat = st.ravel() == 1 if st is not None else np.zeros(init_pts, dtype=bool)
        in_bounds = np.zeros(init_pts, dtype=bool)
        if nxt is not None:
            for j in range(init_pts):
                nx, ny = float(nxt[j, 0, 0]), float(nxt[j, 0, 1])
                if math.isfinite(nx) and math.isfinite(ny) and 0.0 <= nx < w and 0.0 <= ny < h:
                    in_bounds[j] = True

        lk_valid_mask = st_flat & in_bounds
        compared_mask = gt_ok_mask & lk_valid_mask
        lk_lost_pts = int((gt_ok_mask & (~lk_valid_mask)).sum())
        compared_pts = int(compared_mask.sum())

        lk_epes = []
        fb_pt_epes = []
        for j in range(init_pts):
            px, py = float(pts0[j, 0, 0]), float(pts0[j, 0, 1])
            ix, iy = int(round(px)), int(round(py))
            if not compared_mask[j]:
                continue
            gv = (float(gt_flow[iy, ix, 0]), float(gt_flow[iy, ix, 1]))
            lu = float(nxt[j, 0, 0]) - px
            lv = float(nxt[j, 0, 1]) - py
            fv = _bilinear_sample_flow(fb_flow, px, py) or (float(fb_flow[iy, ix, 0]), float(fb_flow[iy, ix, 1]))
            l_err = epe((lu, lv), gv)
            f_err = epe(fv, gv)
            lk_epes.append(l_err)
            fb_pt_epes.append(f_err)
            if max_corners == 10 and label == "LK_PDF_win200":
                point_rows.append({
                    "pair": pair_name,
                    "pt_idx": j,
                    "x": round(px, 2),
                    "y": round(py, 2),
                    "gt_u": round(gv[0], 3),
                    "gt_v": round(gv[1], 3),
                    "lk_u": round(lu, 3),
                    "lk_v": round(lv, 3),
                    "fb_u": round(fv[0], 3),
                    "fb_v": round(fv[1], 3),
                    "lk_epe": round(l_err, 4),
                    "fb_epe": round(f_err, 4),
                })

        loss_pct = 100.0 * lk_lost_pts / gt_valid_pts if gt_valid_pts else 0.0
        summary_rows.append({
            "dataset": pair_name,
            "resolution": f"{w}x{h}",
            "max_corners": max_corners,
            "method": label,
            "init_pts": init_pts,
            "gt_valid_pts": gt_valid_pts,
            "lost_pts": lk_lost_pts,
            "compared_pts": compared_pts,
            "point_loss_pct": round(loss_pct, 2),
            "mean_epe_at_pts": round(sum(lk_epes) / len(lk_epes), 4) if lk_epes else float("nan"),
            "mean_epe_all_px": "-",
            "valid_px_count": "-",
            "median_ms": round(lk_ms, 3),
            "fps": round(1000.0 / lk_ms, 2) if lk_ms > 0 else 0.0,
        })

        if label == "LK_PDF_win200":
            summary_rows.append({
                "dataset": pair_name,
                "resolution": f"{w}x{h}",
                "max_corners": max_corners,
                "method": "Farneback_dense",
                "init_pts": init_pts,
                "gt_valid_pts": gt_valid_pts,
                "lost_pts": 0,
                "compared_pts": compared_pts,
                "point_loss_pct": 0.0,
                "mean_epe_at_pts": round(sum(fb_pt_epes) / len(fb_pt_epes), 4) if fb_pt_epes else float("nan"),
                "mean_epe_all_px": round(fb_all_epe, 4),
                "valid_px_count": fb_valid_px,
                "median_ms": round(fb_ms, 3),
                "fps": round(1000.0 / fb_ms, 2) if fb_ms > 0 else 0.0,
            })

    return summary_rows, point_rows


def _run_seq_lk_ablation(
    seq_name: str,
    frames: List[np.ndarray],
    gt_vec: Optional[Tuple[float, float]],
    lk_cfg: Dict,
    use_fb: bool,
    redetect_interval: int,
    variant_name: str,
) -> Dict:
    """Run one controlled LK ablation variant over a multi-frame sequence."""
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    h, w = g0.shape
    pts0 = cv2.goodFeaturesToTrack(g0, mask=None, **SHI_TOMASI_PDF)
    if pts0 is None:
        pts0 = np.empty((0, 1, 2), dtype=np.float32)

    def run_pass():
        prev_gray = g0
        tracks = [(float(p[0]), float(p[1])) for p in pts0.reshape(-1, 2)]
        init_n = len(tracks)
        epes: List[float] = []
        retained_counts: List[int] = []
        t_algo = 0.0

        for idx in range(1, len(frames)):
            cur_gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            t0 = time.perf_counter()

            if tracks:
                p0 = np.float32(tracks).reshape(-1, 1, 2)
                p1, st_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, **lk_cfg)
                st_mask = st_f.ravel() == 1 if st_f is not None else np.zeros(len(tracks), dtype=bool)
                if use_fb and p1 is not None:
                    p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, None, **lk_cfg)
                    fb_err = np.abs(p0 - p0r).reshape(-1, 2).max(-1)
                    st_mask = st_mask & (st_b.ravel() == 1) & (fb_err < FB_THRESH_PX)

                new_tracks = []
                for j, ((px, py), ok) in enumerate(zip(tracks, st_mask)):
                    if not ok or p1 is None:
                        continue
                    nx, ny = float(p1[j, 0, 0]), float(p1[j, 0, 1])
                    if not (math.isfinite(nx) and math.isfinite(ny) and 0.0 <= nx < w and 0.0 <= ny < h):
                        continue
                    new_tracks.append((nx, ny))
                    if gt_vec is not None:
                        epes.append(epe((nx - px, ny - py), gt_vec))
                tracks = new_tracks

            if redetect_interval > 0 and idx % redetect_interval == 0:
                mask = np.full_like(cur_gray, 255)
                for tx, ty in tracks:
                    cv2.circle(mask, (int(round(tx)), int(round(ty))), 5, 0, -1)
                extra = cv2.goodFeaturesToTrack(cur_gray, mask=mask, **SHI_TOMASI_PDF)
                if extra is not None:
                    for ex, ey in extra.reshape(-1, 2):
                        if len(tracks) < SHI_TOMASI_PDF["maxCorners"]:
                            tracks.append((float(ex), float(ey)))

            t_algo += (time.perf_counter() - t0) * 1000.0
            retained_counts.append(len(tracks))
            prev_gray = cur_gray

        return init_n, retained_counts, epes, t_algo / max(1, len(frames) - 1)

    times = []
    last_out = None
    for _ in range(TIMING_REPEATS):
        last_out = run_pass()
        times.append(last_out[3])

    init_n, retained_counts, epes, _ = last_out
    med_ms = statistics.median(times)
    final_pts = retained_counts[-1] if retained_counts else 0
    return {
        "sequence": seq_name,
        "variant": variant_name,
        "winSize": str(lk_cfg["winSize"]),
        "fb_check": int(use_fb),
        "redetect_interval": redetect_interval,
        "init_pts": init_n,
        "final_pts": final_pts,
        "retention_pct": round(100.0 * final_pts / init_n, 2) if init_n else 0.0,
        "avg_active_pts": round(sum(retained_counts) / len(retained_counts), 2) if retained_counts else 0.0,
        "mean_epe_px": round(sum(epes) / len(epes), 4) if (gt_vec is not None and epes) else "-",
        "median_ms": round(med_ms, 3),
        "fps": round(1000.0 / med_ms, 2) if med_ms > 0 else 0.0,
    }


def main() -> None:
    enforce_single_thread()
    ensure_dirs()

    pairs = _load_flow_pairs()
    summary_all: List[Dict] = []
    points_all: List[Dict] = []

    print("=== Optical Flow Benchmark (Two-frame GT pairs) ===")
    for pair_name, img0, img1, flo, valid in pairs:
        for mc in (10, 100):
            s_rows, p_rows = eval_pair_lk_vs_farneback(pair_name, img0, img1, flo, valid, max_corners=mc)
            summary_all.extend(s_rows)
            points_all.extend(p_rows)
            if mc == 10:
                for r in s_rows:
                    print(
                        f"  {pair_name:22s} | {r['method']:15s} | pts={r['compared_pts']:2d}/{r['gt_valid_pts']:2d} "
                        f"| loss={r['point_loss_pct']:5.1f}% | EPE_pts={str(r['mean_epe_at_pts']):>7s} "
                        f"| EPE_all={str(r['mean_epe_all_px']):>7s} | {r['median_ms']:6.3f} ms"
                    )

    # Multi-frame ablation sequences:
    # 1) Controlled camera shift sequence (exact GT u=-6, v=-1)
    cam_frames = [
        cv2.imread(str(DATA_DIR / "controlled" / "bicycle_cam_shift_seq" / f"frame_{i:04d}.png"))
        for i in range(24)
    ]
    # 2) Controlled object+occlusion sequence (no full-frame flow GT, tests point loss & FB rejection)
    occ_frames = [
        cv2.imread(str(DATA_DIR / "controlled" / "bicycle_occ_seq" / f"frame_{i:04d}.png"))
        for i in range(40)
    ]
    # 3) Real DAVIS bmx-trees (qualitative real-video sequence without flow GT)
    davis_paths = sorted((DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees").glob("*.jpg"))
    davis_frames = [cv2.imread(str(p)) for p in davis_paths]

    ablations: List[Dict] = []
    configs = [
        ("LK_PDF_win200_noFB", LK_PDF, False, 0),
        ("LK_PDF_win200_FB", LK_PDF, True, 0),
        ("LK_PDF_win200_redetect5", LK_PDF, False, 5),
        ("LK_PDF_win200_FB_redetect5", LK_PDF, True, 5),
        ("LK_small_win15_noFB", LK_SMALL, False, 0),
        ("LK_small_win15_FB", LK_SMALL, True, 0),
    ]
    print("\n=== LK Ablations (FB check, Re-detection, winSize=(200,200) vs (15,15)) ===")
    for seq_name, frames, gt_v in [
        ("Controlled:cam_shift_24f", cam_frames, (-6.0, -1.0)),
        ("Controlled:bicycle_occ_40f", occ_frames, None),
        ("DAVIS:bmx_trees_80f", davis_frames, None),
    ]:
        for vname, cfg, fb_flag, redet in configs:
            row = _run_seq_lk_ablation(seq_name, frames, gt_v, cfg, fb_flag, redet, vname)
            ablations.append(row)
            print(
                f"  {seq_name:24s} | {vname:27s} | final={row['final_pts']:2d}/{row['init_pts']:2d} "
                f"({row['retention_pct']:5.1f}%) | EPE={str(row['mean_epe_px']):>7s} | {row['median_ms']:6.3f} ms"
            )

    with open(CSV_DIR / "flow_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary_all[0].keys()))
        wr.writeheader()
        wr.writerows(summary_all)

    with open(CSV_DIR / "flow_ablation.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(ablations[0].keys()))
        wr.writeheader()
        wr.writerows(ablations)

    if points_all:
        with open(CSV_DIR / "flow_points_detail.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(points_all[0].keys()))
            wr.writeheader()
            wr.writerows(points_all)

    print("Saved flow CSVs in", CSV_DIR)


if __name__ == "__main__":
    main()
