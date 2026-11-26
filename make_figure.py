"""Generate storyboards, IoU failure timelines, flow error heatmaps, camera compensation, and charts."""

from __future__ import annotations

import csv
import math
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


def _gt_box(g: Dict[str, str]) -> Tuple[float, float, float, float]:
    px = "vis_x" if "vis_x" in g else "x"
    py = "vis_y" if "vis_y" in g else "y"
    pw = "vis_w" if "vis_w" in g else "w"
    ph = "vis_h" if "vis_h" in g else "h"
    return (float(g[px]), float(g[py]), float(g[pw]), float(g[ph]))


def _read_csv(path: Path) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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


def build_intro_3frame_story() -> Path:
    """Build 3-frame beginner story (visible -> full occlusion -> confirmed reappearance) with captions."""
    seq = "ctrl_bicycle_occ"
    gt = _load_gt_csv(CSV_DIR / "ctrl_bicycle_occ_gt.csv")
    csrt = _load_track_csv(seq, "CSRT")
    csrt_rd = _load_track_csv(seq, "CSRT_REDETECT")
    tld = _load_track_csv(seq, "TLD")
    mil = _load_track_csv(seq, "MIL")

    frames_idx = [4, 19, 31]
    titles = [
        "1) Frame 4: Tsever kharagdah (Visible)",
        "2) Frame 19: Buren khalkhlalt (0% Visible)",
        "3) Frame 31: Dahin garch batlagdsan (Confirmed)",
    ]
    panels = []
    for fi, title in zip(frames_idx, titles):
        im = cv2.imread(str(DATA_DIR / "controlled" / "bicycle_occ_seq" / f"frame_{fi:04d}.png"))
        g = gt[fi]
        if int(g["present"]) == 1:
            _draw_box_label(im, (float(g["vis_x"]), float(g["vis_y"]), float(g["vis_w"]), float(g["vis_h"])), "GT", (0, 0, 255), 3, -6)
        else:
            ax, ay, aw, ah = (int(float(g[k])) for k in ("amodal_x", "amodal_y", "amodal_w", "amodal_h"))
            cv2.rectangle(im, (ax, ay), (ax + aw, ay + ah), (0, 140, 255), 1)
            cv2.putText(im, "FULL OCCLUSION", (ax - 20, ay - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        for r, lbl, col, yoff in [
            (csrt[fi], "CSRT", (0, 220, 0), 18),
            (csrt_rd[fi], "ReDetect", (255, 180, 0), 34),
            (tld[fi], "TLD", (255, 0, 255), 50),
            (mil[fi], "MIL", (0, 200, 255), -20),
        ]:
            if int(r["valid_pred"]):
                _draw_box_label(im, (float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"])), lbl, col, 2, yoff)

        cv2.putText(im, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # Clean white info footer showing exact GT, tracker status, and IoU
        footer = np.full((125, im.shape[1], 3), 250, dtype=np.uint8)
        cv2.line(footer, (0, 0), (im.shape[1], 0), (180, 180, 180), 1)
        gt_str = f"GT: present={g['present']} vis=({g['vis_x']},{g['vis_y']},{g['vis_w']},{g['vis_h']})"
        lines = [
            gt_str,
            f"CSRT: ok={csrt[fi]['success']} IoU={csrt[fi]['iou']}",
            f"CSRT_ReDetect: ok={csrt_rd[fi]['success']} IoU={csrt_rd[fi]['iou']}",
            f"TLD: ok={tld[fi]['success']} IoU={tld[fi]['iou']}  |  MIL: ok={mil[fi]['success']} IoU={mil[fi]['iou']}",
        ]
        for li, txt in enumerate(lines):
            cv2.putText(footer, txt, (10, 26 + li * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(np.vstack([im, footer]))

    canvas = cv2.resize(np.hstack(panels), (1650, 520), interpolation=cv2.INTER_AREA)
    out = FIGURES_DIR / "intro_3frame_story_cc0.png"
    cv2.imwrite(str(out), canvas)
    return out


def build_iou_timeline_figure() -> Path:
    """Plot frame-by-frame IoU over ctrl_bicycle_occ with shaded occlusion & recovery markers."""
    seq = "ctrl_bicycle_occ"
    gt = _load_gt_csv(CSV_DIR / "ctrl_bicycle_occ_gt.csv")
    summ = {r["tracker"]: r for r in _read_csv(CSV_DIR / "track_summary.csv") if r["sequence"] == seq}

    trackers = ["CSRT", "CSRT_REDETECT", "TLD", "KCF", "MIL"]
    colors = {"CSRT": "#1f77b4", "CSRT_REDETECT": "#2ca02c", "TLD": "#9467bd", "KCF": "#d62728", "MIL": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    occ_frames = [fi for fi, r in gt.items() if int(r["present"]) == 0]
    if occ_frames:
        ax.axvspan(min(occ_frames) - 0.5, max(occ_frames) + 0.5, color="#d0d0d0", alpha=0.55, label="Full occlusion (18–22)")

    frames_x = sorted(fi for fi in gt if fi >= 1)
    for tr in trackers:
        rows = _load_track_csv(seq, tr)
        ys = [float(rows[fi]["iou"]) for fi in frames_x]
        ax.plot(frames_x, ys, marker="o", markersize=3, linewidth=1.8, color=colors[tr], label=f"{tr} (mean={summ[tr]['mean_iou']})")

    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0, label="IoU = 0.50 threshold")

    # Annotate TLD first IoU>=0.5 and 3-frame confirmation from CSV (well-separated labels)
    ann_specs = [
        ("TLD", (-5.5, 0.87), (0.8, 0.92), "#9467bd"),
        ("CSRT_REDETECT", (-5.2, 0.43), (-1.5, 0.31), "#2ca02c"),
    ]
    for tr, off1, off2, c in ann_specs:
        f1 = summ[tr]["first_ok_frame"]
        fc = summ[tr]["confirmed_frame"]
        if f1 != "-" and fc != "-":
            f1i, fci = int(f1), int(fc)
            rows = _load_track_csv(seq, tr)
            y1, y2 = float(rows[f1i]["iou"]), float(rows[fci]["iou"])
            ax.scatter([f1i, fci], [y1, y2], color=c, s=55, zorder=5)
            box_st = dict(boxstyle="round,pad=0.18", fc="white", alpha=0.88, ec="none")
            ax.annotate(f"{tr} start f{f1i}", xy=(f1i, y1), xytext=(f1i + off1[0], off1[1]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.1), fontsize=8, color=c, fontweight="bold", bbox=box_st)
            ax.annotate(f"{tr} 3-fr ok f{fci}", xy=(fci, y2), xytext=(fci + off2[0], off2[1]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.1), fontsize=8, color=c, fontweight="bold", bbox=box_st)

    ax.set_xlim(1, max(frames_x))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Frame # (frame 0 excluded)")
    ax.set_ylabel("IoU (vs visible GT)")
    ax.set_title("Frame-by-Frame IoU Across Occlusion & Reappearance (CC0 Bicycle Sequence)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    fig.tight_layout()
    out = FIGURES_DIR / "iou_timeline_occlusion_cc0.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_ladder_and_stress_chart() -> Path:
    """Visualize occlusion ladder, visible vs amodal GT, stress tests, and ROI perturbation."""
    ladder = _read_csv(CSV_DIR / "track_occlusion_ladder.csv")
    stress = _read_csv(CSV_DIR / "track_stress_tests.csv")
    roi = _read_csv(CSV_DIR / "track_roi_sensitivity.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), dpi=160)

    # Panel A: Occlusion ladder Mean IoU
    ax = axes[0, 0]
    lseqs = ["ctrl_occ_ladder_3f", "ctrl_occ_ladder_10f", "ctrl_occ_ladder_20f", "ctrl_occ_ladder_10f_distractor"]
    llabels = ["3f Occ", "11f Occ", "22f Occ", "11f + Distractor"]
    ltrs = ["CSRT", "CSRT_REDETECT", "TLD", "KCF", "MIL"]
    x = np.arange(len(lseqs))
    bw = 0.15
    for ti, tr in enumerate(ltrs):
        vals = [float(next(r["mean_iou"] for r in ladder if r["sequence"] == s and r["tracker"] == tr)) for s in lseqs]
        ax.bar(x + (ti - 2) * bw, vals, width=bw, label=tr)
    ax.set_xticks(x)
    ax.set_xticklabels(llabels, fontsize=8.5)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Mean IoU (visible GT)")
    ax.set_title("A. Occlusion Ladder: Duration & Distractor Confusion")
    ax.legend(fontsize=7.5, ncol=3)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Visible GT vs Amodal GT on ctrl_occ_ladder_10f & ctrl_bicycle_occ
    ax = axes[0, 1]
    btrs = ["CSRT", "CSRT_REDETECT", "TLD", "MIL", "MEDIANFLOW"]
    xb = np.arange(len(btrs))
    vis_v = [float(next(r["mean_iou"] for r in ladder if r["sequence"] == "ctrl_occ_ladder_10f" and r["tracker"] == tr)) for tr in btrs]
    amo_v = [float(next(r["mean_iou_amodal"] for r in ladder if r["sequence"] == "ctrl_occ_ladder_10f" and r["tracker"] == tr)) for tr in btrs]
    ax.bar(xb - 0.18, vis_v, width=0.35, label="Visible-Part GT IoU")
    ax.bar(xb + 0.18, amo_v, width=0.35, label="Amodal (Full-Object) GT IoU")
    ax.set_xticks(xb)
    ax.set_xticklabels(btrs, fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Mean IoU")
    ax.set_title("B. Visible GT vs Amodal GT (11-Frame Occlusion)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Single-Factor Stress Tests on davis_bmx_trees_bike
    ax = axes[1, 0]
    conds = ["baseline_1x", "stride_2x", "stride_3x", "blur_gauss_k15", "blur_motion_15", "darken_05x"]
    clabels = ["1x Base", "2x Stride", "3x Stride", "Gauss Blur", "Motion Blur", "Darken 0.5x"]
    strs = ["CSRT", "CSRT_REDETECT", "TLD", "MIL"]
    xc = np.arange(len(conds))
    bw2 = 0.19
    for ti, tr in enumerate(strs):
        vals = [float(next(r["mean_iou"] for r in stress if r["sequence"] == "davis_bmx_trees_bike" and r["condition"] == c and r["tracker"] == tr)) for c in conds]
        ax.bar(xc + (ti - 1.5) * bw2, vals, width=bw2, label=tr)
    ax.set_xticks(xc)
    ax.set_xticklabels(clabels, fontsize=8, rotation=12)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Mean IoU")
    ax.set_title("C. Single-Factor Stress Tests (DAVIS bmx-trees bike)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel D: Initial ROI Sensitivity (delta IoU on ctrl_bicycle_occ)
    ax = axes[1, 1]
    perts = ["shift_10pct", "scale_plus_10pct", "scale_minus_10pct"]
    plabels = ["Shift +10%", "Scale +10%", "Scale -10%"]
    rtrs = ["CSRT", "CSRT_REDETECT", "TLD", "MEDIANFLOW", "MIL"]
    xd = np.arange(len(perts))
    bw3 = 0.15
    for ti, tr in enumerate(rtrs):
        vals = [float(next(r["delta_iou"] for r in roi if r["sequence"] == "ctrl_bicycle_occ" and r["perturbation"] == p and r["tracker"] == tr)) for p in perts]
        ax.bar(xd + (ti - 2) * bw3, vals, width=bw3, label=tr)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-0.12, 0.095)
    ax.set_xticks(xd)
    ax.set_xticklabels(plabels, fontsize=8.5)
    ax.set_ylabel("Δ Mean IoU vs Exact ROI")
    ax.set_title("D. Initial ROI Perturbation Sensitivity (CC0 Bicycle)")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "occlusion_ladder_stress_cc0.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_flow_error_and_dis_figure() -> Path:
    """Visualize Farneback vs DIS_MEDIUM spatial EPE heatmaps and Pareto speed/accuracy."""
    f0 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / "cam_shift_f0.png"))
    f1 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / "cam_shift_f1.png"))
    gt_flo, valid = read_flo(DATA_DIR / "controlled" / "flow_pairs" / "cam_shift_gt.flo")
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

    fb_fl = cv2.calcOpticalFlowFarneback(g0, g1, None, **FARNEBACK_PDF)
    dis_fl = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g0, g1, None)

    err_fb = np.where(valid, np.hypot(fb_fl[..., 0] - gt_flo[..., 0], fb_fl[..., 1] - gt_flo[..., 1]), 0.0)
    err_dis = np.where(valid, np.hypot(dis_fl[..., 0] - gt_flo[..., 0], dis_fl[..., 1] - gt_flo[..., 1]), 0.0)

    flow_rows = [r for r in _read_csv(CSV_DIR / "flow_summary.csv") if int(r["max_corners"]) == 10]
    reg_rows = _read_csv(CSV_DIR / "flow_dense_regions.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), dpi=160)

    # Top-left: Farneback spatial error on cam_shift (clamped 0..6 px)
    im1 = axes[0, 0].imshow(np.clip(err_fb, 0, 6.0), cmap="inferno")
    axes[0, 0].set_title(f"A. Farneback EPE Heatmap on cam_shift (All={err_fb[valid].mean():.3f} px)")
    axes[0, 0].axis("off")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, label="EPE (px)")

    # Top-right: DIS_MEDIUM spatial error on cam_shift (same 0..6 px scale!)
    im2 = axes[0, 1].imshow(np.clip(err_dis, 0, 6.0), cmap="inferno", vmin=0, vmax=6.0)
    axes[0, 1].set_title(f"B. DIS_MEDIUM EPE Heatmap on cam_shift (All={err_dis[valid].mean():.3f} px)")
    axes[0, 1].axis("off")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, label="EPE (px)")

    # Bottom-left: Region-wise EPE on cam_shift & obj_occ (Farneback vs DIS_MEDIUM)
    ax = axes[1, 0]
    rkeys = ["all_valid_epe", "border_band_epe", "low_texture_epe", "high_texture_epe"]
    rlabels = ["All Valid", "Border 24px", "Low Texture", "High Texture"]
    x = np.arange(len(rkeys))
    fb_row = next(r for r in reg_rows if r["dataset"] == "Controlled:cam_shift" and r["method"] == "Farneback_dense")
    df_row = next(r for r in reg_rows if r["dataset"] == "Controlled:cam_shift" and r["method"] == "DIS_FAST")
    dm_row = next(r for r in reg_rows if r["dataset"] == "Controlled:cam_shift" and r["method"] == "DIS_MEDIUM")
    fb_c = [float(fb_row[k]) for k in rkeys]
    df_c = [float(df_row[k]) for k in rkeys]
    dis_c = [float(dm_row[k]) for k in rkeys]
    ax.bar(x - 0.25, fb_c, width=0.24, label=f"Farneback ({float(fb_row['algo_median_ms']):.1f} ms)")
    ax.bar(x, df_c, width=0.24, label=f"DIS_FAST ({float(df_row['algo_median_ms']):.1f} ms)")
    ax.bar(x + 0.25, dis_c, width=0.24, label=f"DIS_MEDIUM ({float(dm_row['algo_median_ms']):.1f} ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(rlabels, fontsize=8.5)
    ax.set_ylabel("Mean EPE (px)")
    ax.set_title("C. Region-Wise EPE Breakdown (cam_shift)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Bottom-right: Dense Flow Accuracy vs Latency across Middlebury, CC0, Sintel
    ax = axes[1, 1]
    dsets = ["Middlebury:RubberWhale", "Middlebury:Grove2", "Controlled:obj_occ", "Controlled:cam_shift", "Controlled:cam_plus_obj", "Sintel:alley_1_clean", "Sintel:ambush_5_final"]
    dmethods = ["Farneback_dense", "DIS_ULTRAFAST", "DIS_FAST", "DIS_MEDIUM"]
    markers = {"Farneback_dense": "s", "DIS_ULTRAFAST": "^", "DIS_FAST": "o", "DIS_MEDIUM": "D"}
    for m in dmethods:
        ms_list = [float(next(r["algo_median_ms"] for r in flow_rows if r["dataset"] == d and r["method"] == m)) for d in dsets]
        epe_list = [float(next(r["all_valid_px_epe"] for r in flow_rows if r["dataset"] == d and r["method"] == m)) for d in dsets]
        ax.scatter(np.mean(ms_list), np.mean(epe_list), s=110, marker=markers[m], label=f"{m} (avg {np.mean(ms_list):.1f}ms, {np.mean(epe_list):.2f}px)")
    ax.set_xscale("log")
    ax.set_xlabel("Median Algorithm Runtime (ms/frame, log, 1 thread)")
    ax.set_ylabel("Mean All-Pixel EPE across 7 Pairs (px)")
    ax.set_title("D. Dense Flow Tradeoff: DIS vs Farneback (Lower-Left is Better)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "flow_error_and_dis_cc0.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_camera_compensation_figure() -> Tuple[Path, Path]:
    """Build 3-panel (Raw Flow | Camera Motion | Residual Flow) for CC0 cam+obj and DAVIS bmx-trees."""
    from benchmark_flow import compute_camera_motion_compensation

    wheel = make_color_wheel(84, max_mag=FARNEBACK_VIS_MAX_MAG)

    def render_3panel(im0: np.ndarray, im1: np.ndarray, tag: str) -> np.ndarray:
        raw, cam, res = compute_camera_motion_compensation(im0, im1, flow_method="dis_medium")
        panels = []
        for fl, lbl in [(raw, f"{tag}: Raw DIS_MEDIUM"), (cam, "Est. Camera Flow"), (res, "Residual (Raw - Camera)")]:
            vis = draw_hsv_fixed(fl, max_mag=FARNEBACK_VIS_MAX_MAG)
            vis[6 : 6 + wheel.shape[0], 6 : 6 + wheel.shape[1]] = wheel
            cv2.putText(vis, lbl, (96, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
            panels.append(vis)
        return np.hstack(panels)

    f0 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / "cam_plus_obj_f0.png"))
    f1 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / "cam_plus_obj_f1.png"))
    p_cc0 = cv2.resize(render_3panel(f0, f1, "CC0 Pan+Obj"), (1500, 380), interpolation=cv2.INTER_AREA)
    out_cc0 = FIGURES_DIR / "camera_motion_compensation_cc0.png"
    cv2.imwrite(str(out_cc0), p_cc0)

    b0 = cv2.imread(str(DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees" / "00010.jpg"))
    b1 = cv2.imread(str(DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees" / "00011.jpg"))
    p_davis = cv2.resize(render_3panel(b0, b1, "DAVIS bmx f10"), (1500, 380), interpolation=cv2.INTER_AREA)
    out_davis = LOCAL_VIZ_DIR / "camera_motion_compensation_davis.png"
    cv2.imwrite(str(out_davis), p_davis)
    return out_cc0, out_davis


def _compute_lk_and_fb_panels(frames: List[np.ndarray], target_indices: List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **SHI_TOMASI_PDF)
    if pts is None:
        pts = np.empty((0, 1, 2), dtype=np.float32)
    mask_lk = np.zeros_like(frames[0])
    wheel = make_color_wheel(84, max_mag=FARNEBACK_VIS_MAX_MAG)
    lk_panels, fb_panels = {}, {}
    h, w = prev_gray.shape
    for idx in range(1, len(frames)):
        cur_gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        if len(pts) > 0:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **LK_PDF)
            keep = (st.ravel() == 1) & np.isfinite(nxt).all(axis=(1, 2))
            good_new = nxt[keep].reshape(-1, 2)
            good_prev = pts[keep].reshape(-1, 2)
            for (xn, yn), (xp, yp) in zip(good_new, good_prev):
                cv2.line(mask_lk, (int(round(xn)), int(round(yn))), (int(round(xp)), int(round(yp))), (0, 255, 0), 2, cv2.LINE_AA)
            pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        if idx in target_indices:
            lk_vis = cv2.add(frames[idx].copy(), mask_lk)
            for p in pts.reshape(-1, 2):
                cv2.circle(lk_vis, (int(round(p[0])), int(round(p[1]))), 5, (0, 0, 255), -1)
            cv2.putText(lk_vis, f"LK pts={len(pts)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            lk_panels[idx] = lk_vis
            flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, **FARNEBACK_PDF)
            hsv_bgr = draw_hsv_fixed(flow, max_mag=FARNEBACK_VIS_MAX_MAG)
            step = 24
            yy, xx = np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(int)
            fx, fy = flow[yy, xx].T
            for x1, y1, u, v in zip(xx, yy, fx, fy):
                if math.hypot(float(u), float(v)) >= 0.8:
                    cv2.arrowedLine(hsv_bgr, (int(x1), int(y1)), (int(round(x1 + u * 1.5)), int(round(y1 + v * 1.5))), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
            hsv_bgr[6 : 6 + wheel.shape[0], 6 : 6 + wheel.shape[1]] = wheel
            cv2.putText(hsv_bgr, f"Farneback max={FARNEBACK_VIS_MAX_MAG:.0f}px", (96, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
            fb_panels[idx] = hsv_bgr
        prev_gray = cur_gray
    return lk_panels, fb_panels


def build_existing_storyboards_and_demo() -> None:
    """Refresh storyboards, metrics chart, demo MP4, and local qualitative DAVIS/Middlebury figures."""
    seq_a = "ctrl_bicycle_occ"
    gt_a = _load_gt_csv(CSV_DIR / "ctrl_bicycle_occ_gt.csv")
    csrt_a = _load_track_csv(seq_a, "CSRT")
    kcf_a = _load_track_csv(seq_a, "KCF")
    tld_a = _load_track_csv(seq_a, "TLD")
    rd_a = _load_track_csv(seq_a, "CSRT_REDETECT")
    frames_a = [cv2.imread(str(DATA_DIR / "controlled" / "bicycle_occ_seq" / f"frame_{i:04d}.png")) for i in range(40)]
    idxs_a = [4, 13, 19, 31]
    lk_a, fb_a = _compute_lk_and_fb_panels(frames_a, idxs_a)

    rows_a = [[], [], [], []]
    for fi in idxs_a:
        p1 = frames_a[fi].copy()
        g = gt_a[fi]
        if int(g["present"]) == 1:
            _draw_box_label(p1, (float(g["vis_x"]), float(g["vis_y"]), float(g["vis_w"]), float(g["vis_h"])), f"GT vis ({float(g['visible_fraction'])*100:.0f}%)", (0, 0, 255), 3)
        else:
            ax, ay, aw, ah = (int(float(g[k])) for k in ("amodal_x", "amodal_y", "amodal_w", "amodal_h"))
            cv2.rectangle(p1, (ax, ay), (ax + aw, ay + ah), (0, 140, 255), 1)
            cv2.putText(p1, "GT: FULL OCCLUSION (0%)", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(p1, f"A: Frame {fi}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        rows_a[0].append(p1)

        p2 = frames_a[fi].copy()
        for r, lbl, col, yoff in [
            (csrt_a[fi], "CSRT", (0, 255, 0), -6),
            (rd_a[fi], "ReDet", (255, 180, 0), 18),
            (tld_a[fi], "TLD", (255, 0, 255), 36),
            (kcf_a[fi], "KCF", (255, 220, 0), 54),
        ]:
            if int(r["valid_pred"]):
                _draw_box_label(p2, (float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"])), f"{lbl} {float(r['iou']):.2f}", col, 2, yoff)
        cv2.putText(p2, f"CSRT:{csrt_a[fi]['success']} ReDet:{rd_a[fi]['success']} TLD:{tld_a[fi]['success']}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        rows_a[1].append(p2)
        rows_a[2].append(lk_a[fi])
        rows_a[3].append(fb_a[fi])
    cv2.imwrite(str(FIGURES_DIR / "bicycle_storyboard_cc0.png"), cv2.resize(np.vstack([np.hstack(r) for r in rows_a]), (1600, 1200)))

    # Variant B storyboard: Camera shift
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
        _draw_box_label(p1, _gt_box(g), "GT", (0, 0, 255), 3)
        cv2.putText(p1, f"B (Camera Shift): Frame {fi}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        rows_b[0].append(p1)
        p2 = frames_b[fi].copy()
        _draw_box_label(p2, (float(csrt_b[fi]["box_x"]), float(csrt_b[fi]["box_y"]), float(csrt_b[fi]["box_w"]), float(csrt_b[fi]["box_h"])), f"CSRT {float(csrt_b[fi]['iou']):.2f}", (0, 255, 0), 2, -6)
        _draw_box_label(p2, (float(kcf_b[fi]["box_x"]), float(kcf_b[fi]["box_y"]), float(kcf_b[fi]["box_w"]), float(kcf_b[fi]["box_h"])), f"KCF {float(kcf_b[fi]['iou']):.2f}", (255, 220, 0), 2, 18)
        rows_b[1].append(p2)
        rows_b[2].append(lk_b[fi])
        rows_b[3].append(fb_b[fi])
    cv2.imwrite(str(FIGURES_DIR / "camera_shift_storyboard_cc0.png"), cv2.resize(np.vstack([np.hstack(r) for r in rows_b]), (1600, 1200)))

    # LK vs Farneback CC0 pair comparison
    wheel = make_color_wheel(84, max_mag=FARNEBACK_VIS_MAX_MAG)
    pair_rows = []
    for pair_key, lbl in [("obj_occ", "Variant A: Object + Occlusion"), ("cam_shift", "Variant B: Whole-Frame Camera Shift")]:
        im0 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / f"{pair_key}_f0.png"))
        im1 = cv2.imread(str(DATA_DIR / "controlled" / "flow_pairs" / f"{pair_key}_f1.png"))
        g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(g0, mask=None, **SHI_TOMASI_PDF)
        p_big, st_big, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK_PDF)
        p_sm, st_sm, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK_SMALL)
        vis_lk = im0.copy()
        for q0, qb, sb, qs, ss in zip(p0, p_big, st_big, p_sm, st_sm):
            x0, y0 = float(q0[0, 0]), float(q0[0, 1])
            if int(sb[0]) == 1:
                cv2.arrowedLine(vis_lk, (int(round(x0)), int(round(y0))), (int(round(x0 + (qb[0, 0] - x0) * 3)), int(round(y0 + (qb[0, 1] - y0) * 3))), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.35)
            if int(ss[0]) == 1:
                cv2.arrowedLine(vis_lk, (int(round(x0)), int(round(y0))), (int(round(x0 + (qs[0, 0] - x0) * 3)), int(round(y0 + (qs[0, 1] - y0) * 3))), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.35)
            cv2.circle(vis_lk, (int(round(x0)), int(round(y0))), 3, (255, 255, 0), -1)
        cv2.putText(vis_lk, f"{lbl} | LK (3x vectors)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(vis_lk, "Red=win200  Green=win15", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        fl = cv2.calcOpticalFlowFarneback(g0, g1, None, **FARNEBACK_PDF)
        vis_arr = draw_flow_arrows(g0, fl, step=16)
        cv2.putText(vis_arr, "Farneback Dense Arrows (step=16)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        vis_hsv = draw_hsv_fixed(fl, max_mag=FARNEBACK_VIS_MAX_MAG)
        vis_hsv[6 : 6 + wheel.shape[0], 6 : 6 + wheel.shape[1]] = wheel
        cv2.putText(vis_hsv, f"Farneback HSV (max={FARNEBACK_VIS_MAX_MAG:.0f}px)", (96, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1)
        pair_rows.append(np.hstack([vis_lk, vis_arr, vis_hsv]))
    cv2.imwrite(str(FIGURES_DIR / "lk_vs_farneback_cc0.png"), cv2.resize(np.vstack(pair_rows), (1600, 800)))

    # Local-only DAVIS bmx-trees qualitative comparison (from current CSVs!)
    gt_d = _load_gt_csv(CSV_DIR / "davis_bmx_trees_bike_gt.csv")
    csrt_d = _load_track_csv("davis_bmx_trees_bike", "CSRT")
    rd_d = _load_track_csv("davis_bmx_trees_bike", "CSRT_REDETECT")
    tld_d = _load_track_csv("davis_bmx_trees_bike", "TLD")
    mil_d = _load_track_csv("davis_bmx_trees_bike", "MIL")
    d_panels = []
    for fi in [10, 30, 61, 72]:
        im = cv2.imread(str(DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / "bmx-trees" / f"{fi:05d}.jpg"))
        g = gt_d[fi]
        if int(g["present"]) == 1:
            _draw_box_label(im, _gt_box(g), "GT", (0, 0, 255), 3)
        for r, lbl, col, yoff in [
            (csrt_d[fi], "CSRT", (0, 255, 0), 18),
            (rd_d[fi], "ReDet", (255, 180, 0), 34),
            (tld_d[fi], "TLD", (255, 0, 255), -6),
            (mil_d[fi], "MIL", (0, 210, 255), 50),
        ]:
            if int(r["valid_pred"]):
                _draw_box_label(im, (float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"])), f"{lbl}:{float(r['iou']):.2f}", col, 2, yoff)
        cv2.putText(im, f"DAVIS bmx-trees Frame {fi}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        d_panels.append(im)
    cv2.imwrite(str(LOCAL_VIZ_DIR / "davis_bmx_failures.png"), np.vstack([np.hstack(d_panels[:2]), np.hstack(d_panels[2:])]))

    # Demo MP4
    lk_all, fb_all = _compute_lk_and_fb_panels(frames_a, list(range(1, 40)))
    pw, ph = 480, 360
    wr = cv2.VideoWriter(str(FIGURES_DIR / "demo_bicycle_occlusion.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (pw * 2, ph * 2))
    try:
        for fi in range(1, 40):
            p_gt = frames_a[fi].copy()
            g = gt_a[fi]
            if int(g["present"]) == 1:
                _draw_box_label(p_gt, (float(g["vis_x"]), float(g["vis_y"]), float(g["vis_w"]), float(g["vis_h"])), "GT", (0, 0, 255), 3)
            else:
                cv2.putText(p_gt, "GT: FULL OCCLUSION", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(p_gt, f"CC0 Bicycle Demo | Frame {fi}/39", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            p_tr = frames_a[fi].copy()
            for r, lbl, col, yoff in [(csrt_a[fi], "CSRT", (0, 255, 0), -6), (rd_a[fi], "ReDet", (255, 180, 0), 18), (tld_a[fi], "TLD", (255, 0, 255), 36)]:
                if int(r["valid_pred"]):
                    _draw_box_label(p_tr, (float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"])), f"{lbl} {float(r['iou']):.2f}", col, 2, yoff)
            wr.write(np.vstack([np.hstack([cv2.resize(p_gt, (pw, ph)), cv2.resize(p_tr, (pw, ph))]), np.hstack([cv2.resize(lk_all[fi], (pw, ph)), cv2.resize(fb_all[fi], (pw, ph))])]))
    finally:
        wr.release()

    # Main 2x2 metrics chart (now includes CSRT_REDETECT and algo vs e2e FPS!)
    track_rows = _read_csv(CSV_DIR / "track_summary.csv")
    flow_rows = [r for r in _read_csv(CSV_DIR / "flow_summary.csv") if int(r["max_corners"]) == 10]
    ab_rows = _read_csv(CSV_DIR / "flow_ablation.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)
    seqs = ["ctrl_bicycle_occ", "ctrl_bicycle_cam_shift", "davis_bmx_trees_bike", "davis_bmx_trees_rider", "davis_car_shadow"]
    slabels = ["CC0 Bike+Occ", "CC0 CamShift", "DAVIS bmx-bike", "DAVIS bmx-rider", "DAVIS car-shadow"]
    x = np.arange(len(seqs))
    bw = 0.09
    for ti, tr in enumerate(ALL_TRACKERS):
        vals = [float(next(r["mean_iou"] for r in track_rows if r["sequence"] == s and r["tracker"] == tr)) for s in seqs]
        axes[0, 0].bar(x + (ti - 3.5) * bw, vals, width=bw, label=tr)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(slabels, fontsize=7.8)
    axes[0, 0].set_ylim(0, 1.15)
    axes[0, 0].set_ylabel("Mean IoU")
    axes[0, 0].set_title("A. Tracking Accuracy (5 Eval Cases across 4 Sequences)")
    axes[0, 0].legend(fontsize=6.8, ncol=4, loc="upper left")
    axes[0, 0].grid(axis="y", alpha=0.3)

    real_tr = [t for t in ALL_TRACKERS if t != "STATIC"]
    xb = np.arange(len(real_tr))
    algo_fps = [float(next(r["algo_fps"] for r in track_rows if r["sequence"] == "davis_car_shadow" and r["tracker"] == t)) for t in real_tr]
    e2e_fps = [float(next(r["e2e_fps"] for r in track_rows if r["sequence"] == "davis_car_shadow" and r["tracker"] == t)) for t in real_tr]
    axes[0, 1].bar(xb - 0.18, algo_fps, width=0.35, label="Algorithm-only FPS")
    axes[0, 1].bar(xb + 0.18, e2e_fps, width=0.35, label="End-to-End Pipeline FPS")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(xb)
    axes[0, 1].set_xticklabels(real_tr, fontsize=7.5, rotation=15)
    axes[0, 1].set_ylabel("FPS (log scale, 1 thread)")
    axes[0, 1].set_title("B. Algorithm-only vs End-to-End FPS (car-shadow 854x480)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.3)

    dsets = ["Middlebury:RubberWhale", "Middlebury:Grove2", "Controlled:obj_occ", "Controlled:cam_shift"]
    dlabels = ["RubberWhale", "Grove2", "CC0 obj_occ", "CC0 cam_shift"]
    xc = np.arange(len(dsets))
    for mi, m in enumerate(["LK_PDF_win200", "LK_small_win15", "Farneback_dense", "DIS_MEDIUM"]):
        vals = [float(next(r["common_pts_epe"] for r in flow_rows if r["dataset"] == d and r["method"] == m)) for d in dsets]
        axes[1, 0].bar(xc + (mi - 1.5) * 0.18, vals, width=0.17, label=f"{m} (common pts)")
    axes[1, 0].set_xticks(xc)
    axes[1, 0].set_xticklabels(dlabels, fontsize=8.5)
    axes[1, 0].set_ylabel("Mean EPE on Common Valid Points (px)")
    axes[1, 0].set_title("C. Sparse vs Dense EPE on Identical Common Valid Points")
    axes[1, 0].legend(fontsize=7.5)
    axes[1, 0].grid(axis="y", alpha=0.3)

    ab_seqs = ["Controlled:cam_shift_24f", "Controlled:bicycle_occ_40f", "DAVIS:bmx_trees_80f"]
    xd = np.arange(len(ab_seqs))
    vars_ab = ["LK_PDF_win200_noFB", "LK_PDF_win200_FB", "LK_PDF_win200_FB_redetect5"]
    for vi, v in enumerate(vars_ab):
        init_kept = [float(next(r["initial_retention_pct"] for r in ab_rows if r["sequence"] == s and r["variant"] == v)) for s in ab_seqs]
        axes[1, 1].bar(xd + (vi - 1) * 0.22, init_kept, width=0.20, label=v.replace("LK_PDF_win200_", "") + " (init ID)")
    axes[1, 1].set_xticks(xd)
    axes[1, 1].set_xticklabels(["CC0 cam_shift", "CC0 obj_occ", "DAVIS bmx-trees"], fontsize=8.5)
    axes[1, 1].set_ylim(0, 115)
    axes[1, 1].set_ylabel("Original Initial Points Retained (%)")
    axes[1, 1].set_title("D. True Initial-Point Retention (Tracking Point IDs)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tracking_flow_metrics.png")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    f1 = build_intro_3frame_story()
    f2 = build_iou_timeline_figure()
    f3 = build_ladder_and_stress_chart()
    f4 = build_flow_error_and_dis_figure()
    f5, f6 = build_camera_compensation_figure()
    build_existing_storyboards_and_demo()
    print("All figures rebuilt:", f1, f2, f3, f4, f5, f6)


if __name__ == "__main__":
    main()
