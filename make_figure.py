"""Generate storyboards, metric plots, demo MP4, and local qualitative figures from saved CSVs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    ALL_TRACKERS,
    CSV_DIR,
    DATA_DIR,
    FARNEBACK_PDF,
    FARNEBACK_VIS_MAX_MAG,
    FIGURES_DIR,
    LK_PDF,
    LK_SMALL,
    LOCAL_VIZ_DIR,
    SHI_TOMASI_PDF,
    ensure_dirs,
)
from metrics import read_flo
from opt_flow import draw_flow_arrows, draw_hsv_fixed, make_color_wheel


def _load_track_csv(seq_name: str, tracker: str) -> Dict[int, Dict]:
    path = CSV_DIR / f"track_{seq_name}_{tracker}.csv"
    with open(path, newline="") as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def _load_gt_csv(csv_path: Path) -> Dict[int, Dict]:
    with open(csv_path, newline="") as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def _draw_box_label(
    img: np.ndarray,
    box: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
    y_offset: int = -6,
) -> None:
    x, y, w, h = (int(round(v)) for v in box)
    if w <= 0 or h <= 0:
        return
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    ty = max(18, min(img.shape[0] - 8, y + y_offset if y_offset < 0 else y + h + y_offset))
    cv2.putText(img, label, (max(4, x), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _compute_lk_and_fb_panels(frames: List[np.ndarray], target_indices: List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Precompute LK cumulative tracks and fixed-scale Farneback HSV+arrows for selected frames."""
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **SHI_TOMASI_PDF)
    if pts is None:
        pts = np.empty((0, 1, 2), dtype=np.float32)
    mask_lk = np.zeros_like(frames[0])
    wheel = make_color_wheel(84, max_mag=FARNEBACK_VIS_MAX_MAG)

    lk_panels: Dict[int, np.ndarray] = {}
    fb_panels: Dict[int, np.ndarray] = {}

    if 0 in target_indices:
        f0 = frames[0].copy()
        for p in pts.reshape(-1, 2):
            cv2.circle(f0, (int(round(p[0])), int(round(p[1]))), 5, (0, 0, 255), -1)
        cv2.putText(f0, f"LK init pts={len(pts)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        lk_panels[0] = f0
        blank = np.zeros_like(frames[0])
        blank[6 : 6 + wheel.shape[0], 6 : 6 + wheel.shape[1]] = wheel
        cv2.putText(blank, "Frame 0 (no prev)", (98, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        fb_panels[0] = blank

    h, w = prev_gray.shape
    for idx in range(1, len(frames)):
        cur_gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        if len(pts) > 0:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **LK_PDF)
            keep = (st.ravel() == 1) & np.isfinite(nxt).all(axis=(1, 2))
            good_new = nxt[keep].reshape(-1, 2)
            good_prev = pts[keep].reshape(-1, 2)
            for (xn, yn), (xp, yp) in zip(good_new, good_prev):
                cv2.line(mask_lk, (int(round(xn)), int(round(yn))),
                         (int(round(xp)), int(round(yp))), (0, 255, 0), 2, cv2.LINE_AA)
            pts = good_new.reshape(-1, 1, 2).astype(np.float32)

        if idx in target_indices:
            lk_vis = cv2.add(frames[idx].copy(), mask_lk)
            for p in pts.reshape(-1, 2):
                cv2.circle(lk_vis, (int(round(p[0])), int(round(p[1]))), 5, (0, 0, 255), -1)
            cv2.putText(lk_vis, f"LK pts={len(pts)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            lk_panels[idx] = lk_vis

            flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, **FARNEBACK_PDF)
            hsv_bgr = draw_hsv_fixed(flow, max_mag=FARNEBACK_VIS_MAX_MAG)
            # Overlay sparse Farneback motion arrows directly onto HSV map for clarity
            step = 24
            yy, xx = np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(int)
            fx, fy = flow[yy, xx].T
            for x1, y1, u, v in zip(xx, yy, fx, fy):
                if math.hypot(float(u), float(v)) >= 0.8:
                    cv2.arrowedLine(hsv_bgr, (int(x1), int(y1)), (int(round(x1 + u * 1.5)), int(round(y1 + v * 1.5))),
                                    (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
            hsv_bgr[6 : 6 + wheel.shape[0], 6 : 6 + wheel.shape[1]] = wheel
            cv2.putText(hsv_bgr, f"Farneback max={FARNEBACK_VIS_MAX_MAG:.0f}px", (96, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
            fb_panels[idx] = hsv_bgr

        prev_gray = cur_gray

    return lk_panels, fb_panels


import math


def build_bicycle_storyboard() -> Tuple[Path, Path]:
    """Build 4-frame storyboards for Variant A (moving object + column) and Variant B (camera shift)."""
    # Variant A: bicycle_occ_seq
    seq_a = "ctrl_bicycle_occ"
    gt_a = _load_gt_csv(CSV_DIR / "ctrl_bicycle_occ_gt.csv")
    csrt_a = _load_track_csv(seq_a, "CSRT")
    kcf_a = _load_track_csv(seq_a, "KCF")
    tld_a = _load_track_csv(seq_a, "TLD")
    frames_a = [cv2.imread(str(DATA_DIR / "controlled" / "bicycle_occ_seq" / f"frame_{i:04d}.png")) for i in range(40)]
    idxs_a = [4, 13, 19, 32]
    lk_a, fb_a = _compute_lk_and_fb_panels(frames_a, idxs_a)

    rows_a = [[], [], [], []]
    for fi in idxs_a:
        # Row 1: GT
        p1 = frames_a[fi].copy()
        g = gt_a[fi]
        if int(g["present"]) == 1:
            _draw_box_label(p1, (float(g["vis_x"]), float(g["vis_y"]), float(g["vis_w"]), float(g["vis_h"])),
                            f"GT vis ({float(g['visible_fraction'])*100:.0f}%)", (0, 0, 255), 3)
        else:
            ax, ay, aw, ah = (int(float(g[k])) for k in ("amodal_x", "amodal_y", "amodal_w", "amodal_h"))
            cv2.rectangle(p1, (ax, ay), (ax + aw, ay + ah), (0, 140, 255), 1)
            cv2.putText(p1, "GT: FULL OCCLUSION (0%)", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(p1, f"A: Frame {fi}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        rows_a[0].append(p1)

        # Row 2: CSRT vs KCF vs TLD
        p2 = frames_a[fi].copy()
        rc, rk, rt = csrt_a[fi], kcf_a[fi], tld_a[fi]
        if int(rc["valid_pred"]):
            _draw_box_label(p2, (float(rc["box_x"]), float(rc["box_y"]), float(rc["box_w"]), float(rc["box_h"])),
                            f"CSRT IoU={float(rc['iou']):.2f}", (0, 255, 0), 2, -6)
        if int(rk["valid_pred"]):
            _draw_box_label(p2, (float(rk["box_x"]), float(rk["box_y"]), float(rk["box_w"]), float(rk["box_h"])),
                            f"KCF IoU={float(rk['iou']):.2f}", (255, 220, 0), 2, 18)
        if int(rt["valid_pred"]):
            _draw_box_label(p2, (float(rt["box_x"]), float(rt["box_y"]), float(rt["box_w"]), float(rt["box_h"])),
                            f"TLD IoU={float(rt['iou']):.2f}", (255, 0, 255), 2, 34)
        status_txt = f"CSRT:ok={rc['success']} KCF:ok={rk['success']} TLD:ok={rt['success']}"
        cv2.putText(p2, status_txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        rows_a[1].append(p2)

        # Row 3: LK points
        rows_a[2].append(lk_a[fi])
        # Row 4: Farneback HSV + arrows
        rows_a[3].append(fb_a[fi])

    canvas_a = np.vstack([np.hstack(r) for r in rows_a])
    # Resize cleanly so PNG stays crisp and lightweight for GitHub/Word
    canvas_a = cv2.resize(canvas_a, (1600, 1200), interpolation=cv2.INTER_AREA)
    out_a = FIGURES_DIR / "bicycle_storyboard_cc0.png"
    cv2.imwrite(str(out_a), canvas_a)

    # Variant B: bicycle_cam_shift_seq
    seq_b = "ctrl_bicycle_cam_shift"
    gt_b = _load_gt_csv(CSV_DIR / "ctrl_bicycle_cam_shift_gt.csv")
    csrt_b = _load_track_csv(seq_b, "CSRT")
    kcf_b = _load_track_csv(seq_b, "KCF")
    frames_b = [cv2.imread(str(DATA_DIR / "controlled" / "bicycle_cam_shift_seq" / f"frame_{i:04d}.png")) for i in range(24)]
    idxs_b = [1, 7, 14, 21]
    lk_b, fb_b = _compute_lk_and_fb_panels(frames_b, idxs_b)

    rows_b = [[], [], [], []]
    for fi in idxs_b:
        p1 = frames_b[fi].copy()
        g = gt_b[fi]
        _draw_box_label(p1, (float(g["x"]), float(g["y"]), float(g["w"]), float(g["h"])), "GT", (0, 0, 255), 3)
        cv2.putText(p1, f"B (Camera Shift): Frame {fi}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows_b[0].append(p1)

        p2 = frames_b[fi].copy()
        rc, rk = csrt_b[fi], kcf_b[fi]
        _draw_box_label(p2, (float(rc["box_x"]), float(rc["box_y"]), float(rc["box_w"]), float(rc["box_h"])),
                        f"CSRT IoU={float(rc['iou']):.2f}", (0, 255, 0), 2, -6)
        _draw_box_label(p2, (float(rk["box_x"]), float(rk["box_y"]), float(rk["box_w"]), float(rk["box_h"])),
                        f"KCF IoU={float(rk['iou']):.2f}", (255, 220, 0), 2, 18)
        rows_b[1].append(p2)
        rows_b[2].append(lk_b[fi])
        rows_b[3].append(fb_b[fi])

    canvas_b = np.vstack([np.hstack(r) for r in rows_b])
    canvas_b = cv2.resize(canvas_b, (1600, 1200), interpolation=cv2.INTER_AREA)
    out_b = FIGURES_DIR / "camera_shift_storyboard_cc0.png"
    cv2.imwrite(str(out_b), canvas_b)
    return out_a, out_b


def build_lk_vs_farneback_figure() -> Path:
    """Visualize sparse LK (winSize 200 vs 15) vs dense Farneback arrows and HSV on CC0 pairs."""
    pairs = [
        ("Variant A: Object + Occlusion", DATA_DIR / "controlled" / "flow_pairs" / "obj_occ_f0.png",
         DATA_DIR / "controlled" / "flow_pairs" / "obj_occ_f1.png"),
        ("Variant B: Whole-Frame Camera Shift", DATA_DIR / "controlled" / "flow_pairs" / "cam_shift_f0.png",
         DATA_DIR / "controlled" / "flow_pairs" / "cam_shift_f1.png"),
    ]
    wheel = make_color_wheel(84, max_mag=FARNEBACK_VIS_MAX_MAG)
    rows = []
    for title, p0_path, p1_path in pairs:
        im0 = cv2.imread(str(p0_path))
        im1 = cv2.imread(str(p1_path))
        g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(g0, mask=None, **SHI_TOMASI_PDF)
        p200, st200, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, **LK_PDF)
        p15, st15, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, **LK_SMALL)

        sparse_vis = im0.copy()
        for j in range(len(pts)):
            x0, y0 = float(pts[j, 0, 0]), float(pts[j, 0, 1])
            if st200[j, 0] == 1:
                x2, y2 = float(p200[j, 0, 0]), float(p200[j, 0, 1])
                cv2.arrowedLine(sparse_vis, (int(round(x0)), int(round(y0))),
                                (int(round(x0 + (x2 - x0) * 3.0)), int(round(y0 + (y2 - y0) * 3.0))),
                                (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.35)
            if st15[j, 0] == 1:
                x1, y1 = float(p15[j, 0, 0]), float(p15[j, 0, 1])
                cv2.arrowedLine(sparse_vis, (int(round(x0)), int(round(y0))),
                                (int(round(x0 + (x1 - x0) * 3.0)), int(round(y0 + (y1 - y0) * 3.0))),
                                (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.35)
            cv2.circle(sparse_vis, (int(round(x0)), int(round(y0))), 4, (255, 255, 0), -1)
        cv2.putText(sparse_vis, f"{title} | LK (3x vectors)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(sparse_vis, "Red=win200  Green=win15", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, **FARNEBACK_PDF)
        arrows = draw_flow_arrows(g0, flow, step=16)
        cv2.putText(arrows, "Farneback Dense Arrows (step=16)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        hsv_bgr = draw_hsv_fixed(flow, max_mag=FARNEBACK_VIS_MAX_MAG)
        hsv_bgr[6 : 6 + wheel.shape[0], 6 : 6 + wheel.shape[1]] = wheel
        cv2.putText(hsv_bgr, f"Farneback HSV (max={FARNEBACK_VIS_MAX_MAG:.0f}px)", (96, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        rows.append(np.hstack([sparse_vis, arrows, hsv_bgr]))

    canvas = cv2.resize(np.vstack(rows), (1500, 750), interpolation=cv2.INTER_AREA)
    out = FIGURES_DIR / "lk_vs_farneback_cc0.png"
    cv2.imwrite(str(out), canvas)
    return out


def build_metric_charts() -> Path:
    """Create 2x2 matplotlib chart summarizing tracking IoU, FPS, Flow EPE, and LK ablations."""
    with open(CSV_DIR / "track_summary.csv", newline="") as f:
        track_rows = list(csv.DictReader(f))
    with open(CSV_DIR / "flow_summary.csv", newline="") as f:
        flow_rows = [r for r in csv.DictReader(f) if int(r["max_corners"]) == 10]
    with open(CSV_DIR / "flow_ablation.csv", newline="") as f:
        ab_rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)

    # Panel A: Mean IoU across 4 key sequences
    seqs = ["ctrl_bicycle_occ", "davis_bmx_trees_bike", "davis_bmx_trees_rider", "davis_car_shadow"]
    seq_labels = ["CC0 Bike+Occ", "DAVIS bmx-bike", "DAVIS bmx-rider", "DAVIS car-shadow"]
    trackers = ALL_TRACKERS
    x = np.arange(len(seqs))
    bw = 0.11
    ax = axes[0, 0]
    for ti, tr in enumerate(trackers):
        vals = [
            float(next(r["mean_iou"] for r in track_rows if r["sequence"] == s and r["tracker"] == tr))
            for s in seqs
        ]
        ax.bar(x + (ti - 3) * bw, vals, width=bw, label=tr)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels, fontsize=8.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Mean IoU")
    ax.set_title("A. Tracking Accuracy (Mean IoU, frame 0 excluded)")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Single-thread median FPS on DAVIS 854x480 (excluding trivial STATIC no-op)
    ax = axes[0, 1]
    real_tr = [t for t in trackers if t != "STATIC"]
    fps_bmx = [float(next(r["fps"] for r in track_rows if r["sequence"] == "davis_bmx_trees_bike" and r["tracker"] == t)) for t in real_tr]
    fps_car = [float(next(r["fps"] for r in track_rows if r["sequence"] == "davis_car_shadow" and r["tracker"] == t)) for t in real_tr]
    xb = np.arange(len(real_tr))
    ax.bar(xb - 0.18, fps_bmx, width=0.35, label="bmx-trees (854x480)")
    ax.bar(xb + 0.18, fps_car, width=0.35, label="car-shadow (854x480)")
    ax.set_yscale("log")
    ax.set_xticks(xb)
    ax.set_xticklabels(real_tr, fontsize=8.5)
    ax.set_ylabel("FPS (log scale, 1 thread)")
    ax.set_title("B. Single-Thread Tracker Speed (Median of 3 Runs)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Optical Flow EPE (maxCorners=10)
    ax = axes[1, 0]
    datasets = ["Middlebury:RubberWhale", "Middlebury:Grove2", "Controlled:obj_occ", "Controlled:cam_shift"]
    ds_labels = ["RubberWhale", "Grove2", "CC0 obj_occ", "CC0 cam_shift"]
    methods = ["LK_PDF_win200", "LK_small_win15", "Farneback_dense"]
    xc = np.arange(len(datasets))
    for mi, m in enumerate(methods):
        vals = [float(next(r["mean_epe_at_pts"] for r in flow_rows if r["dataset"] == d and r["method"] == m)) for d in datasets]
        ax.bar(xc + (mi - 1.5) * 0.19, vals, width=0.18, label=f"{m} (at pts)")
    fb_all = [float(next(r["mean_epe_all_px"] for r in flow_rows if r["dataset"] == d and r["method"] == "Farneback_dense")) for d in datasets]
    ax.bar(xc + 1.5 * 0.19, fb_all, width=0.18, label="Farneback (all valid px)", hatch="//")
    ax.set_xticks(xc)
    ax.set_xticklabels(ds_labels, fontsize=8.5)
    ax.set_ylabel("Mean EPE (pixels, lower is better)")
    ax.set_title("C. Optical Flow Endpoint Error (Shi-Tomasi maxCorners=10)")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)

    # Panel D: LK Ablation Retention %
    ax = axes[1, 1]
    ab_seqs = ["Controlled:cam_shift_24f", "Controlled:bicycle_occ_40f", "DAVIS:bmx_trees_80f"]
    ab_labels = ["CC0 cam_shift (24f)", "CC0 obj_occ (40f)", "DAVIS bmx-trees (80f)"]
    variants = ["LK_PDF_win200_noFB", "LK_PDF_win200_FB", "LK_PDF_win200_redetect5", "LK_PDF_win200_FB_redetect5", "LK_small_win15_FB"]
    xd = np.arange(len(ab_seqs))
    bw2 = 0.15
    for vi, v in enumerate(variants):
        vals = [float(next(r["retention_pct"] for r in ab_rows if r["sequence"] == s and r["variant"] == v)) for s in ab_seqs]
        ax.bar(xd + (vi - 2) * bw2, vals, width=bw2, label=v.replace("LK_PDF_", "").replace("LK_", ""))
    ax.set_xticks(xd)
    ax.set_xticklabels(ab_labels, fontsize=8.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Final Active Points (%)")
    ax.set_title("D. LK Ablation: Forward-Backward Check & Re-detection")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "tracking_flow_metrics.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_demo_video() -> Path:
    """Render a compact CC0 2x2 demo video showing GT, CSRT/KCF/TLD, LK, and Farneback."""
    seq = "ctrl_bicycle_occ"
    gt = _load_gt_csv(CSV_DIR / "ctrl_bicycle_occ_gt.csv")
    csrt = _load_track_csv(seq, "CSRT")
    kcf = _load_track_csv(seq, "KCF")
    tld = _load_track_csv(seq, "TLD")
    frames = [cv2.imread(str(DATA_DIR / "controlled" / "bicycle_occ_seq" / f"frame_{i:04d}.png")) for i in range(40)]
    all_idx = list(range(len(frames)))
    lk_panels, fb_panels = _compute_lk_and_fb_panels(frames, all_idx)

    out_mp4 = FIGURES_DIR / "demo_bicycle_occlusion.mp4"
    pw, ph = 480, 360
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (pw * 2, ph * 2))
    try:
        for fi in all_idx:
            p_gt = frames[fi].copy()
            g = gt[fi]
            if int(g["present"]) == 1:
                _draw_box_label(p_gt, (float(g["vis_x"]), float(g["vis_y"]), float(g["vis_w"]), float(g["vis_h"])), "GT", (0, 0, 255), 3)
            else:
                cv2.putText(p_gt, "GT: FULL OCCLUSION", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(p_gt, f"CC0 Bicycle Demo | Frame {fi}/39", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            p_tr = frames[fi].copy()
            for r, lbl, col, yoff in [
                (csrt[fi], "CSRT", (0, 255, 0), -6),
                (kcf[fi], "KCF", (255, 220, 0), 18),
                (tld[fi], "TLD", (255, 0, 255), 34),
            ]:
                if int(r["valid_pred"]):
                    _draw_box_label(p_tr, (float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"])),
                                    f"{lbl} {float(r['iou']):.2f}", col, 2, yoff)
            cv2.putText(p_tr, "Trackers (CSRT / KCF / TLD)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            q1 = cv2.resize(p_gt, (pw, ph))
            q2 = cv2.resize(p_tr, (pw, ph))
            q3 = cv2.resize(lk_panels[fi], (pw, ph))
            q4 = cv2.resize(fb_panels[fi], (pw, ph))
            writer.write(np.vstack([np.hstack([q1, q2]), np.hstack([q3, q4])]))
    finally:
        writer.release()
    return out_mp4


def build_local_qualitative_viz() -> Tuple[Path, Path]:
    """Save local-only (gitignored) DAVIS failure & Middlebury GT vs Farneback visualizations."""
    # 1. DAVIS bmx-trees failure frames [10, 30, 61, 72]
    seq = "davis_bmx_trees_bike"
    gt = _load_gt_csv(CSV_DIR / "davis_bmx_trees_bike_gt.csv")
    csrt = _load_track_csv(seq, "CSRT")
    kcf = _load_track_csv(seq, "KCF")
    mil = _load_track_csv(seq, "MIL")
    tld = _load_track_csv(seq, "TLD")
    img_dir = DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees"
    panels = []
    for fi in [10, 30, 61, 72]:
        im = cv2.imread(str(img_dir / f"{fi:05d}.jpg"))
        g = gt[fi]
        _draw_box_label(im, (float(g["x"]), float(g["y"]), float(g["w"]), float(g["h"])), "GT", (0, 0, 255), 3, -6)
        for r, lbl, col, yoff in [
            (csrt[fi], "CSRT", (0, 255, 0), 16),
            (mil[fi], "MIL", (0, 200, 255), 32),
            (tld[fi], "TLD", (255, 0, 255), -22),
            (kcf[fi], "KCF", (255, 200, 0), 48),
        ]:
            if int(r["valid_pred"]):
                _draw_box_label(im, (float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"])),
                                f"{lbl}:{float(r['iou']):.2f}", col, 2, yoff)
        cv2.putText(im, f"DAVIS bmx-trees Frame {fi}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(im)
    davis_out = LOCAL_VIZ_DIR / "davis_bmx_failures.png"
    cv2.imwrite(str(davis_out), cv2.resize(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]), (1400, 788)))

    # 2. Middlebury RubberWhale & Grove2 GT vs Farneback HSV
    rows = []
    wheel = make_color_wheel(72, max_mag=6.0)
    for mseq in ("RubberWhale", "Grove2"):
        im0 = cv2.imread(str(DATA_DIR / "middlebury" / "other-data" / mseq / "frame10.png"))
        im1 = cv2.imread(str(DATA_DIR / "middlebury" / "other-data" / mseq / "frame11.png"))
        gt_flo, valid = read_flo(DATA_DIR / "middlebury" / "other-gt-flow" / mseq / "flow10.flo")
        g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        fb_flo = cv2.calcOpticalFlowFarneback(g0, g1, None, **FARNEBACK_PDF)
        gt_vis = draw_hsv_fixed(np.where(valid[..., None], gt_flo, 0.0), max_mag=6.0)
        gt_vis[~valid] = (120, 120, 120)
        fb_vis = draw_hsv_fixed(fb_flo, max_mag=6.0)
        gt_vis[6 : 6 + wheel.shape[0], 6 : 6 + wheel.shape[1]] = wheel
        cv2.putText(im0, f"{mseq} frame10", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(gt_vis, "GT Flow (gray=occl)", (84, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(fb_vis, "Farneback Flow (max=6px)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        rows.append(cv2.resize(np.hstack([im0, gt_vis, fb_vis]), (1350, 320)))
    mid_out = LOCAL_VIZ_DIR / "middlebury_flow_viz.png"
    cv2.imwrite(str(mid_out), np.vstack(rows))
    return davis_out, mid_out


def main() -> None:
    ensure_dirs()
    a, b = build_bicycle_storyboard()
    lvf = build_lk_vs_farneback_figure()
    mc = build_metric_charts()
    vid = build_demo_video()
    d_out, m_out = build_local_qualitative_viz()
    print("Generated figures:", a, b, lvf, mc, vid)
    print("Generated local qualitative visuals:", d_out, m_out)


if __name__ == "__main__":
    main()
