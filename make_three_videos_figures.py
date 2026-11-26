#!/usr/bin/env python3
"""Generate publishable source-image-free figures and local QA overlays for the 3 external video clips."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from benchmark_flow import _detect_stratified_corners, compute_camera_motion_compensation
from config import (
    ALL_TRACKERS,
    CSV_DIR,
    FARNEBACK_VIS_MAX_MAG,
    FIGURES_DIR,
    LK_PDF,
    LOCAL_VIZ_DIR,
    SHI_TOMASI_PDF,
    ensure_dirs,
)
from custom_video_lab import load_verified_gt, load_video_clip
from opt_flow import draw_hsv_fixed, make_color_wheel

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

CUSTOM_FIG_DIR = FIGURES_DIR / "custom_videos"
CUSTOM_LOCAL_DIR = LOCAL_VIZ_DIR / "custom_videos"

TRACKER_COLORS_HEX = {
    "CSRT": "#16a34a",
    "CSRT_REDETECT": "#0284c7",
    "KCF": "#ea580c",
    "BOOSTING": "#7c3aed",
    "MIL": "#db2777",
    "TLD": "#dc2626",
    "MEDIANFLOW": "#0d9488",
    "STATIC": "#64748b",
}

TRACKER_COLORS_BGR = {
    "CSRT": (74, 163, 22),
    "CSRT_REDETECT": (199, 132, 2),
    "KCF": (12, 88, 234),
    "BOOSTING": (237, 58, 124),
    "MIL": (119, 39, 219),
    "TLD": (38, 38, 220),
    "MEDIANFLOW": (136, 148, 13),
    "STATIC": (139, 116, 100),
}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_track_rows(seq: str, tracker: str) -> Dict[int, Dict[str, str]]:
    path = CSV_DIR / f"track_{seq}_{tracker}.csv"
    with open(path, newline="") as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def _draw_box_only(
    ax: plt.Axes,
    box: Tuple[float, float, float, float],
    color: str,
    lw: float = 2.0,
    ls: str = "-",
    fill_alpha: float = 0.0,
    zorder: int = 4,
) -> None:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return
    if fill_alpha > 0:
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=0, facecolor=color, alpha=fill_alpha, zorder=zorder - 1))
    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=lw, linestyle=ls, edgecolor=color, facecolor="none", zorder=zorder))


def _draw_score_pills(
    ax: plt.Axes,
    entries: List[Tuple[str, str]],
    anchor_xy: Tuple[float, float] = (0.03, 0.97),
    va: str = "top",
    fontsize: float = 7.6,
    step: float = 0.064,
) -> None:
    """Place clean non-overlapping vertical legend pills inside axes."""
    x0, y0 = anchor_xy
    for idx, (text, color) in enumerate(entries):
        yy = y0 - idx * step if va == "top" else y0 + idx * step
        ax.text(
            x0,
            yy,
            text,
            transform=ax.transAxes,
            color=color,
            fontsize=fontsize,
            fontweight="bold",
            va=va,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#0f172a", edgecolor=color, alpha=0.90, lw=0.9),
            zorder=9,
        )


def _setup_schematic_axis(ax: plt.Axes, width: int, height: int, title: str) -> None:
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.8, fontweight="bold", pad=6)
    ax.tick_params(labelsize=7, colors="#475569")
    step_x = 320 if width >= 1280 else 180
    step_y = 270 if height >= 1080 else 320
    ax.set_xticks(range(0, width + 1, step_x))
    ax.set_yticks(range(0, height + 1, step_y))
    ax.grid(color="#1e293b", linestyle=":", linewidth=0.8, zorder=0)


def _trajectory_points(gt_rows: List[Dict], up_to_idx: int) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for r in gt_rows[: up_to_idx + 1]:
        if r["present"] == 1 and r["box"][2] > 0:
            x, y, w, h = r["box"]
            xs.append(x + w * 0.5)
            ys.append(y + h * 0.5)
    return xs, ys


def build_three_videos_overview_schematic() -> Path:
    """Opening 1-row 3-panel source-free schematic summarizing representative frames of all 3 videos."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), dpi=170, gridspec_kw={"width_ratios": [1.35, 1.35, 0.72]})
    panels = [
        ("stable", 664, 721, 36, "1. Тогтвортой камер (stable)\norig #700 (clip #36, 1920x1080)", ["CSRT", "MEDIANFLOW", "KCF", "TLD"], None),
        ("moving_camera", 405, 515, 76, "2. Хөдөлгөөнтэй камер (moving_camera)\norig #481 (clip #76, 1920x1080)", ["BOOSTING", "CSRT", "MIL", "MEDIANFLOW"], "chassis"),
        ("occlusion", 42, 235, 131, "3. Халхлалт + хугарал (occlusion)\norig #173 (clip #131, 720x1280)", ["KCF", "CSRT", "BOOSTING", "TLD"], "shield"),
    ]
    summary_rows = _read_csv_rows(CSV_DIR / "phone_videos_summary.csv")

    for ax, (seq, s_f, e_f, ci, title, tr_list, cue) in zip(axes, panels):
        gt = load_verified_gt(CSV_DIR / f"phone_{seq}_gt.csv", list(range(s_f, e_f + 1)))
        w = 720 if seq == "occlusion" else 1920
        h = 1280 if seq == "occlusion" else 1080
        _setup_schematic_axis(ax, w, h, title)

        if cue == "shield":
            ax.axvspan(100, 360, color="#38bdf8", alpha=0.14, zorder=1)
            ax.axvline(100, color="#38bdf8", ls="--", lw=1.0, alpha=0.7)
            ax.axvline(360, color="#38bdf8", ls="--", lw=1.0, alpha=0.7)
            ax.text(230, 80, "Хугарлын\nхалхавч\n(x≈100–360)", color="#bae6fd", fontsize=7.6, ha="center", va="top", bbox=dict(facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.85))
        elif cue == "chassis":
            ax.axhspan(740, 1080, color="#94a3b8", alpha=0.16, zorder=1)
            ax.axhline(290, color="#facc15", ls=":", lw=1.0, alpha=0.6)
            ax.text(1220, 265, "Модны шугам / тэнгэрийн хаяа", color="#fde047", fontsize=8, ha="center")
            ax.text(1050, 910, "360° камертай тэрэгний бүс (≈0 px)", color="#cbd5e1", fontsize=8, ha="center")
        else:
            ax.add_patch(patches.Rectangle((430, 195), 95, 80, facecolor="#475569", edgecolor="#94a3b8", alpha=0.45))
            ax.text(550, 240, "Хөдлөхгүй шоо (0 px)", color="#cbd5e1", fontsize=7.5, ha="left", va="center")

        tx, ty = _trajectory_points(gt, ci)
        ax.plot(tx, ty, color="#f87171", lw=1.8, alpha=0.85, zorder=3)
        _draw_box_only(ax, gt[ci]["box"], "#ef4444", lw=2.5, ls="-", fill_alpha=0.12, zorder=4)

        pills = [("GT бодит хүрээ (улаан)", "#ef4444")]
        for tr in tr_list:
            r = _load_track_rows(seq, tr)[ci]
            if int(r["valid_pred"]):
                bx, by, bw, bh = float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"])
                _draw_box_only(ax, (bx, by, bw, bh), TRACKER_COLORS_HEX[tr], lw=1.8, ls="--", zorder=5)
                pills.append((f"{tr}: IoU={float(r['iou']):.2f}", TRACKER_COLORS_HEX[tr]))
            else:
                pills.append((f"{tr}: FAIL (0x0)", TRACKER_COLORS_HEX[tr]))

        if seq == "occlusion":
            _draw_score_pills(ax, pills, anchor_xy=(0.04, 0.39), va="top", fontsize=7.2, step=0.056)
        else:
            _draw_score_pills(ax, pills, anchor_xy=(0.03, 0.96), va="top", fontsize=7.4, step=0.064)

        best_row = max((r for r in summary_rows if r["sequence"] == seq and r["tracker"] != "STATIC"), key=lambda r: float(r["mean_iou"]))
        ax.text(
            0.03,
            0.03,
            f"Шилдэг Mean IoU: {best_row['tracker']} = {float(best_row['mean_iou']):.4f}",
            transform=ax.transAxes,
            color="#f8fafc",
            fontsize=8.0,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.92),
            zorder=8,
        )

    fig.suptitle("3 шинэ видео туршилтын нээлтийн харьцуулалт (эх дүрсгүй хүрээний схем, пиксел координат дээр)", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = CUSTOM_FIG_DIR / "three_videos_overview_schematic.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_stable_figure() -> Path:
    """Figure 2: stable sequence bbox scaling schematic + zoom inset + LK/DIS stationary floor vs moving car flow."""
    frames, orig_idx, _ = load_video_clip(Path("data/phone_videos/stable.mp4"), 664, 721, max_frames=300)
    gt = load_verified_gt(CSV_DIR / "phone_stable_gt.csv", orig_idx)
    csrt = _load_track_rows("stable", "CSRT")
    mf = _load_track_rows("stable", "MEDIANFLOW")
    kcf = _load_track_rows("stable", "KCF")
    tld = _load_track_rows("stable", "TLD")

    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, mask=None, **SHI_TOMASI_PDF)
    bx0, by0, bw0, bh0 = (int(round(v)) for v in gt[0]["box"])
    rmask = np.zeros_like(g0)
    rmask[max(0, by0) : min(g0.shape[0], by0 + bh0), max(0, bx0) : min(g0.shape[1], bx0 + bw0)] = 255
    p_roi0 = cv2.goodFeaturesToTrack(g0, mask=rmask, **SHI_TOMASI_PDF)

    def track_final(pts):
        cur = pts.copy()
        prev_g = g0.copy()
        for fr in frames[1:]:
            cg = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, cg, cur, None, **LK_PDF)
            cur = nxt[st.ravel() == 1].reshape(-1, 1, 2)
            prev_g = cg
        return cur

    p_roi_end = track_final(p_roi0)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    mid_ci = 26  # orig 690
    flow_mid = dis.calc(cv2.cvtColor(frames[mid_ci], cv2.COLOR_BGR2GRAY), cv2.cvtColor(frames[mid_ci + 1], cv2.COLOR_BGR2GRAY), None)
    hsv_bgr = draw_hsv_fixed(flow_mid, max_mag=FARNEBACK_VIS_MAX_MAG)
    wheel = make_color_wheel(140, max_mag=FARNEBACK_VIS_MAX_MAG)
    hsv_bgr[18 : 18 + wheel.shape[0], 18 : 18 + wheel.shape[1]] = wheel
    hsv_rgb = cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5), dpi=170)

    # Left: Full-frame trajectory + zoomed inset of orig #720 overlapping boxes
    ax0 = axes[0]
    _setup_schematic_axis(ax0, 1920, 1080, "A. Хөдөлгөөнгүй камер: томрох машины хүрээ ба томруулсан хэсэг (orig #664→#721)")
    tx, ty = _trajectory_points(gt, 57)
    ax0.plot(tx, ty, color="#f87171", lw=2.3, zorder=3)
    for ci_step in (1, 26, 56):
        _draw_box_only(ax0, gt[ci_step]["box"], "#ef4444", lw=2.0, fill_alpha=0.10)
    ci = 56
    for tr in ("CSRT", "MEDIANFLOW", "KCF", "TLD"):
        row = {"CSRT": csrt, "MEDIANFLOW": mf, "KCF": kcf, "TLD": tld}[tr][ci]
        box = (float(row["box_x"]), float(row["box_y"]), float(row["box_w"]), float(row["box_h"]))
        _draw_box_only(ax0, box, TRACKER_COLORS_HEX[tr], lw=1.7, ls="--")

    ax0.add_patch(patches.Rectangle((1430, 260), 470, 230, fill=False, edgecolor="#38bdf8", ls=":", lw=1.6))
    ax0.text(1665, 240, "Томруулсан хэсэг (orig #720)", color="#38bdf8", fontsize=8.0, ha="center", fontweight="bold")
    ax0.text(
        1020,
        740,
        "Хэмжээний өөрчлөлтийн дүгнэлт:\n"
        "• GT хүрээ 98×59 → 121×87 px (+31% талбай)\n"
        "• CSRT (0.8505) ба MedianFlow (0.8267) тэлэлт дагасан\n"
        "• KCF/BOOSTING/MIL тогтмол 98×59 тул 0.72–0.73\n"
        "• TLD (0.4070) жижигрэн дээвэр дээр хумигдсан",
        color="#e2e8f0",
        fontsize=8.0,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.95),
    )

    ax_zoom = inset_axes(ax0, width="46%", height="48%", loc="upper left", borderpad=1.6)
    ax_zoom.set_facecolor("#0b1120")
    ax_zoom.set_xlim(1435, 1635)
    ax_zoom.set_ylim(510, 345)
    ax_zoom.tick_params(labelsize=6.5, colors="#94a3b8")
    ax_zoom.grid(color="#1e293b", ls=":", lw=0.6)
    ax_zoom.set_title("orig #720 (clip #56) хүрээний нарийвчлал", fontsize=7.8, color="#38bdf8", fontweight="bold", pad=3)
    _draw_box_only(ax_zoom, gt[ci]["box"], "#ef4444", lw=2.3, fill_alpha=0.14)
    pills_z = [("GT: 121×87 px", "#ef4444")]
    for tr in ("CSRT", "MEDIANFLOW", "KCF", "TLD"):
        row = {"CSRT": csrt, "MEDIANFLOW": mf, "KCF": kcf, "TLD": tld}[tr][ci]
        box = (float(row["box_x"]), float(row["box_y"]), float(row["box_w"]), float(row["box_h"]))
        _draw_box_only(ax_zoom, box, TRACKER_COLORS_HEX[tr], lw=1.9, ls="--")
        pills_z.append((f"{tr}: {float(row['iou']):.2f}", TRACKER_COLORS_HEX[tr]))
    _draw_score_pills(ax_zoom, pills_z, anchor_xy=(0.03, 0.96), va="top", fontsize=6.7)
    for sp in ax_zoom.spines.values():
        sp.set_edgecolor("#38bdf8")
        sp.set_linewidth(1.4)

    # Right: DIS flow HSV + LK sparse points
    ax1 = axes[1]
    ax1.imshow(hsv_rgb)
    ax1.set_xlim(0, 1920)
    ax1.set_ylim(1080, 0)
    ax1.set_title("B. Sparse LK (10/10 дэвсгэр, 7/7 ROI) ба нягт DIS урсгал (orig #690→#691)", fontsize=9.8, fontweight="bold", pad=6)
    ax1.tick_params(labelsize=7)
    for i, pt in enumerate(p0):
        px, py = float(pt[0, 0]), float(pt[0, 1])
        ax1.scatter([px], [py], c="#22d3ee", s=38, edgecolors="white", linewidths=0.8, zorder=5, label="Дэвсгэр LK цэг (0.002–0.009 px/кадр)" if i == 0 else None)
    for i, (q0, q1) in enumerate(zip(p_roi0, p_roi_end)):
        x0, y0 = float(q0[0, 0]), float(q0[0, 1])
        x1, y1 = float(q1[0, 0]), float(q1[0, 1])
        ax1.plot([x0, x1], [y0, y1], color="#facc15", lw=1.7, zorder=6, label="Машин ROI LK зам (7/7 үлдсэн, 6.84 px/кадр)" if i == 0 else None)
        ax1.scatter([x1], [y1], c="#facc15", s=26, zorder=7)
    _draw_box_only(ax1, gt[mid_ci]["box"], "#facc15", lw=1.8, zorder=6)
    gx, gy, _, _ = gt[mid_ci]["box"]
    ax1.text(gx, gy - 18, "Машин: медиан 6.60 px", color="#facc15", fontsize=7.8, fontweight="bold", bbox=dict(facecolor="#0f172a", edgecolor="#facc15", alpha=0.88))
    ax1.text(24, 190, f"Color wheel max={FARNEBACK_VIS_MAX_MAG:.0f} px\nДэвсгэрийн DIS медиан: 0.059 px", color="#f8fafc", fontsize=8.0, bbox=dict(facecolor="#0f172a", edgecolor="#64748b", alpha=0.88))
    ax1.legend(loc="lower left", fontsize=7.8, facecolor="#0f172a", edgecolor="#64748b", labelcolor="#f8fafc")

    fig.tight_layout()
    out = CUSTOM_FIG_DIR / "stable_tracking.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_moving_camera_flow_figure() -> Path:
    """Figure 3: Moving camera global vs local residual DIS optical flow at orig #505 (best) vs #481 (worst)."""
    frames, orig_idx, _ = load_video_clip(Path("data/phone_videos/moving_camera.mp4"), 405, 515, max_frames=300)
    gt = load_verified_gt(CSV_DIR / "phone_moving_camera_gt.csv", orig_idx)
    pair_rows = {int(r["clip_frame"]): r for r in _read_csv_rows(CSV_DIR / "phone_moving_camera_flow_pairs.csv") if r["flow_method"] == "dis_medium"}
    wheel = make_color_wheel(120, max_mag=FARNEBACK_VIS_MAX_MAG)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), dpi=170)
    cases = [
        (100, "Эргэлт (pan) давамгай кадр: orig #505 (clip #100)\nДэвсгэр бууралт +35.85% | RANSAC inlier 60.0%"),
        (76, "Төвөгтэй эргэлт + 3D параллакс: orig #481 (clip #76)\nДэвсгэр алдаа -24.20% (өссөн!) | inlier 34.1%"),
    ]

    for row_idx, (ci, title_left) in enumerate(cases):
        raw, cam, res = compute_camera_motion_compensation(frames[ci], frames[ci + 1], flow_method="dis_medium")
        g0 = cv2.cvtColor(frames[ci], cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frames[ci + 1], cv2.COLOR_BGR2GRAY)
        pts0 = _detect_stratified_corners(g0)
        lk_bg = dict(winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        pts1, st_f, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, **lk_bg)
        pts0r, st_b, _ = cv2.calcOpticalFlowPyrLK(g1, g0, pts1, None, **lk_bg)
        fb = np.linalg.norm(pts0.reshape(-1, 2) - pts0r.reshape(-1, 2), axis=1)
        ok = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb <= 1.5)
        _, inl = cv2.estimateAffine2D(pts0[ok].reshape(-1, 2), pts1[ok].reshape(-1, 2), method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000, confidence=0.995)
        ok_pts0 = pts0[ok].reshape(-1, 2)
        ok_pts1 = pts1[ok].reshape(-1, 2)
        inl_mask = inl.ravel().astype(bool) if inl is not None else np.zeros(len(ok_pts0), dtype=bool)

        ax0 = axes[row_idx, 0]
        _setup_schematic_axis(ax0, 1920, 1080, title_left)
        ax0.axhspan(740, 1080, color="#64748b", alpha=0.18, zorder=1)
        ax0.axhline(290, color="#94a3b8", ls=":", lw=1.0, alpha=0.6)
        step_show = max(1, len(ok_pts0) // 60)
        for p_a, p_b, is_in in zip(ok_pts0[::step_show], ok_pts1[::step_show], inl_mask[::step_show]):
            dx, dy = (p_b[0] - p_a[0]) * 4.5, (p_b[1] - p_a[1]) * 4.5
            col = "#22c55e" if is_in else "#f43f5e"
            ax0.arrow(p_a[0], p_a[1], dx, dy, color=col, head_width=18, head_length=20, lw=1.1, alpha=0.85, zorder=4)
        _draw_box_only(ax0, gt[ci]["box"], "#38bdf8", lw=2.2, fill_alpha=0.12)
        gx, gy, _, _ = gt[ci]["box"]
        ax0.text(gx, gy - 20, "Нохойн GT", color="#38bdf8", fontsize=8.0, fontweight="bold", bbox=dict(facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.9))
        ax0.text(30, 1020, "Ногоон сум = RANSAC affine inlier   Улаан сум = outlier (өвс/хэв гажилт/нохой)", color="#f8fafc", fontsize=7.6, bbox=dict(facecolor="#0f172a", edgecolor="#475569", alpha=0.9))

        pr = pair_rows[ci]
        h_raw = draw_hsv_fixed(raw, max_mag=FARNEBACK_VIS_MAX_MAG)
        h_res = draw_hsv_fixed(res, max_mag=FARNEBACK_VIS_MAX_MAG)
        h_raw[18 : 18 + wheel.shape[0], 18 : 18 + wheel.shape[1]] = wheel
        for col_i, (img_bgr, ttl) in enumerate(
            [
                (h_raw, f"Түүхий DIS урсгал (max={FARNEBACK_VIS_MAX_MAG:.0f} px)\nДэвсгэр медиан = {float(pr['raw_bg_med_px']):.2f} px"),
                (h_res, f"Affine хассаны дараах үлдэгдэл урсгал\nҮлдэгдэл: дэвсгэр {float(pr['res_bg_med_px']):.2f} px | нохой {float(pr['res_fg_med_px']):.2f} px"),
            ],
            start=1,
        ):
            ax = axes[row_idx, col_i]
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax.set_xlim(0, 1920)
            ax.set_ylim(1080, 0)
            _draw_box_only(ax, gt[ci]["box"], "#ffffff", lw=1.8, ls="--")
            ax.text(gx, gy - 16, "Нохойн GT", color="#ffffff", fontsize=7.5, fontweight="bold", bbox=dict(facecolor="#0f172a", edgecolor="#ffffff", alpha=0.82))
            ax.set_title(ttl, fontsize=9.2, fontweight="bold", pad=5)
            ax.tick_params(labelsize=7)

    fig.suptitle("Хөдөлгөөнтэй камер (moving_camera): Камерын глобал хөдөлгөөнийг ялгах 2D RANSAC нөхөлт ба хязгаарлалт", fontsize=11.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = CUSTOM_FIG_DIR / "moving_camera_flow.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_occlusion_storyboard_and_timeline() -> Tuple[Path, Path]:
    """Figure 4 & 5: Source-free 4-stage occlusion storyboard + full IoU timeline (legend neatly below!)."""
    orig_indices = list(range(42, 236))
    gt = load_verified_gt(CSV_DIR / "phone_occlusion_gt.csv", orig_indices)
    track_data = {tr: _load_track_rows("occlusion", tr) for tr in ALL_TRACKERS}

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 6.0), dpi=170)
    stages = [
        (8, "1) Цэвэр харагдах\norig #50 (clip #8)"),
        (52, "2) Хугаралт оролт #1\norig #94 (clip #52)"),
        (108, "3) Хугаралт оролт #2\norig #150 (clip #108)"),
        (140, "4) Цэвэр гарч ирсний дараа\norig #182 (clip #140)"),
    ]
    for ax, (ci, title) in zip(axes, stages):
        _setup_schematic_axis(ax, 720, 1280, title)
        ax.axvspan(100, 360, color="#38bdf8", alpha=0.14, zorder=1)
        ax.axvline(100, color="#38bdf8", ls="--", lw=1.0, alpha=0.75)
        ax.axvline(360, color="#38bdf8", ls="--", lw=1.0, alpha=0.75)
        ax.text(230, 50, "Тунгалаг\nхугарлын\nхалхавч\n(pass 1 & 2)", color="#bae6fd", fontsize=7.5, ha="center", va="top")

        tx, ty = _trajectory_points(gt, ci)
        ax.plot(tx, ty, color="#f87171", lw=1.8, alpha=0.85, zorder=3)
        _draw_box_only(ax, gt[ci]["box"], "#ef4444", lw=2.4, fill_alpha=0.12, zorder=4)

        pills = [("GT алим (улаан)", "#ef4444")]
        for tr in ("KCF", "CSRT", "BOOSTING", "TLD", "MEDIANFLOW"):
            r = track_data[tr][ci]
            if int(r["valid_pred"]):
                box = (float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"]))
                _draw_box_only(ax, box, TRACKER_COLORS_HEX[tr], lw=1.7, ls="--", zorder=5)
                pills.append((f"{tr}: IoU={float(r['iou']):.2f}", TRACKER_COLORS_HEX[tr]))
            else:
                pills.append((f"{tr}: FAIL (алдсан)", TRACKER_COLORS_HEX[tr]))

        _draw_score_pills(ax, pills, anchor_xy=(0.04, 0.31), va="top", fontsize=7.0, step=0.050)

    fig.suptitle("Халхлалт + тунгалаг хугарлын бичлэг (occlusion): Алимны хөдөлгөөн ба tracker хүрээний схем", fontsize=11.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_story = CUSTOM_FIG_DIR / "occlusion_storyboard.png"
    fig.savefig(fig_story)
    plt.close(fig)

    # Figure 5: Frame-by-frame IoU timeline with legend neatly BELOW axis so zero overlap!
    fig2, ax = plt.subplots(figsize=(14.5, 6.2), dpi=170)
    frames_x = [int(gt[i]["frame"]) for i in range(1, len(gt))]
    ax.axvspan(25, 66, color="#f59e0b", alpha=0.16, label="Тунгалаг хугарал #1 (orig #67–#108, present=1)")
    ax.axvspan(84, 130, color="#ef4444", alpha=0.14, label="Тунгалаг хугарал #2 (orig #126–#172, гар+шил)")
    ax.axvline(131, color="#0f172a", ls="-.", lw=1.8, label="Бүрэн гарч ирсэн зааг: clip #131 (orig #173)")
    ax.axhline(0.5, color="#334155", ls=":", lw=1.3, label="Босго IoU = 0.50")

    for tr in ["KCF", "CSRT", "CSRT_REDETECT", "BOOSTING", "MIL", "MEDIANFLOW", "TLD", "STATIC"]:
        ious = [float(track_data[tr][fi]["iou"]) for fi in frames_x]
        lw = 1.2 if tr == "STATIC" else (2.3 if tr in ("KCF", "CSRT") else 1.8)
        ls = "--" if tr in ("STATIC", "CSRT_REDETECT") else "-"
        alpha = 0.7 if tr == "STATIC" else 0.92
        lbl = f"{tr}" if tr != "STATIC" else "STATIC (хөдлөхгүй суурь)"
        ax.plot(frames_x, ious, color=TRACKER_COLORS_HEX[tr], lw=lw, ls=ls, alpha=alpha, label=lbl)

    ax.annotate(
        "KCF / CSRT / BOOSTING:\nclip #131 (orig #173) батлагдсан\n(0 кадр хоцролтгүй сэргэсэн)",
        xy=(133, 0.72),
        xytext=(141, 0.86),
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5),
        fontsize=8.2,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#16a34a", alpha=0.95),
    )
    ax.annotate(
        "MIL: 31 кадр хоцорч\nclip #162 (orig #204)-т сэргэсэн",
        xy=(162, 0.51),
        xytext=(144, 0.34),
        arrowprops=dict(arrowstyle="->", color="#db2777", lw=1.5),
        fontsize=8.2,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#db2777", alpha=0.95),
    )
    ax.annotate(
        "TLD ба MedianFlow:\nхалхлалтын дараа\nэргэж олоогүй",
        xy=(131, 0.05),
        xytext=(69, 0.14),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
        fontsize=8.2,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#dc2626", alpha=0.95),
    )

    ax.set_xlim(1, 193)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Тайрдсын кадрын дугаар (clip frame, orig #42–#235; бүгд present=1, absent=0, no_gt=0)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Кадр бүрийн IoU (≥ 0.5 амжилттай)", fontsize=9.5, fontweight="bold")
    ax.set_title("Occlusion бичлэгийн кадр бүрийн IoU цагийн шугам ба сэргэлтийн шинжилгээ (2 удаа шилний цаагуур өнгөрөх үе)", fontsize=10.8, fontweight="bold", pad=26)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8.0, framealpha=0.95)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    top_ticks = [8, 25, 66, 84, 108, 131, 162, 193]
    ax_top.set_xticks(top_ticks)
    ax_top.set_xticklabels([f"orig #{t + 42}" for t in top_ticks], fontsize=7.8)

    fig2.tight_layout()
    fig_tl = CUSTOM_FIG_DIR / "occlusion_iou_timeline.png"
    fig2.savefig(fig_tl, bbox_inches="tight")
    plt.close(fig2)
    return fig_story, fig_tl


def build_benchmark_comparison_and_why_failed() -> Tuple[Path, Path]:
    """Figure 6 & 7: 3-video accuracy + real-tracker speed (STATIC excluded from FPS!), plus why_failed schematic."""
    rows = _read_csv_rows(CSV_DIR / "phone_videos_summary.csv")
    seqs = ["stable", "moving_camera", "occlusion"]
    slabels = ["Тогтвортой\n(stable, 1080p)", "Хөдөлгөөнтэй камер\n(moving_cam, 1080p)", "Халхлалт + хугарал\n(occlusion, 720x1280)"]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), dpi=170)
    x = np.arange(len(seqs))
    bw = 0.095
    for ti, tr in enumerate(ALL_TRACKERS):
        vals = [float(next(r["mean_iou"] for r in rows if r["sequence"] == s and r["tracker"] == tr)) for s in seqs]
        lbl = f"{tr} (суурь)" if tr == "STATIC" else tr
        hatch = "//" if tr == "STATIC" else None
        bars = axes[0].bar(x + (ti - 3.5) * bw, vals, width=bw, color=TRACKER_COLORS_HEX[tr], hatch=hatch, label=lbl, edgecolor="#1e293b", linewidth=0.5)
        for b in bars:
            vh = b.get_height()
            axes[0].text(b.get_x() + b.get_width() * 0.5, vh + 0.015, f"{vh:.2f}", ha="center", va="bottom", fontsize=6.4, rotation=90)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(slabels, fontsize=8.8, fontweight="bold")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Mean IoU (Өндөр байх тусам сайн)", fontsize=9.2, fontweight="bold")
    axes[0].set_title("A. 3 шинэ бичлэг дээрх нарийвчлалын харьцуулалт (Mean IoU)", fontsize=10.2, fontweight="bold")
    axes[0].legend(fontsize=7.3, ncol=2, loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)

    real_tr = [t for t in ALL_TRACKERS if t != "STATIC"]
    xb = np.arange(len(real_tr))
    algo_stable = [float(next(r["algo_fps"] for r in rows if r["sequence"] == "stable" and r["tracker"] == t)) for t in real_tr]
    e2e_stable = [float(next(r["e2e_fps"] for r in rows if r["sequence"] == "stable" and r["tracker"] == t)) for t in real_tr]
    algo_occ = [float(next(r["algo_fps"] for r in rows if r["sequence"] == "occlusion" and r["tracker"] == t)) for t in real_tr]
    e2e_occ = [float(next(r["e2e_fps"] for r in rows if r["sequence"] == "occlusion" and r["tracker"] == t)) for t in real_tr]

    w_b = 0.19
    axes[1].bar(xb - 1.5 * w_b, algo_stable, width=w_b, color="#0284c7", label="1080p Algo FPS (stable)")
    axes[1].bar(xb - 0.5 * w_b, e2e_stable, width=w_b, color="#38bdf8", hatch="..", label="1080p E2E FPS (угсралттай)")
    axes[1].bar(xb + 0.5 * w_b, algo_occ, width=w_b, color="#16a34a", label="720p Algo FPS (occlusion)")
    axes[1].bar(xb + 1.5 * w_b, e2e_occ, width=w_b, color="#86efac", hatch="..", label="720p E2E FPS (угсралттай)")
    axes[1].set_yscale("log")
    axes[1].axhline(24.0, color="#ef4444", ls="--", lw=1.2, label="Бодит 24 FPS босго")
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(real_tr, fontsize=8.0, rotation=15, fontweight="bold")
    axes[1].set_ylim(5, 1100)
    axes[1].set_ylabel("Кадр/сек (FPS, логарифм масштаб, 1 CPU thread)", fontsize=9.0, fontweight="bold")
    axes[1].set_title("B. Бодит 7 алгоритмын хурд (STATIC хурднаас хасагдсан)", fontsize=10.2, fontweight="bold")
    axes[1].legend(fontsize=7.3, ncol=2, loc="upper right")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig_comp = CUSTOM_FIG_DIR / "video_benchmark_comparison.png"
    fig.savefig(fig_comp)
    plt.close(fig)

    # Figure 7: Why did trackers fail? Clean 2-panel schematic with proper aspect ratios and score pills
    fig_w, ax_w = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=170, gridspec_kw={"width_ratios": [1.55, 0.72]})

    gt_m = load_verified_gt(CSV_DIR / "phone_moving_camera_gt.csv", list(range(405, 516)))
    _setup_schematic_axis(ax_w[0], 1920, 1080, "A. Алдааны шалтгаан 1: moving_camera (orig #481)\nАрын хурдан өвсний урсгалд хазайх")
    ax_w[0].axhspan(740, 1080, color="#64748b", alpha=0.18)
    _draw_box_only(ax_w[0], gt_m[76]["box"], "#ef4444", lw=2.5, fill_alpha=0.14)
    pills_a = [("GT нохой (улаан)", "#ef4444")]
    for tr in ("BOOSTING", "CSRT", "MIL", "MEDIANFLOW"):
        r = _load_track_rows("moving_camera", tr)[76]
        if int(r["valid_pred"]):
            box = (float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"]))
            _draw_box_only(ax_w[0], box, TRACKER_COLORS_HEX[tr], lw=1.9, ls="--")
            pills_a.append((f"{tr}: IoU={float(r['iou']):.2f}", TRACKER_COLORS_HEX[tr]))
    _draw_score_pills(ax_w[0], pills_a, anchor_xy=(0.03, 0.28), va="top", fontsize=7.6)
    ax_w[0].text(
        40,
        45,
        "Шалтгаан:\n"
        "• 360° урагш түрэлтээр хажуугийн өвс хурдан урсана\n"
        "• MedianFlow (0.118) ба MIL (0.209) зүүн өвс рүү хоцорч шилжсэн\n"
        "• BOOSTING (0.510) ба CSRT (0.470) нохойн дүрсийг хадгалсан",
        color="#f8fafc",
        fontsize=7.9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#1e293b", edgecolor="#ef4444", alpha=0.95),
    )

    gt_o = load_verified_gt(CSV_DIR / "phone_occlusion_gt.csv", list(range(42, 236)))
    _setup_schematic_axis(ax_w[1], 720, 1280, "B. Алдааны шалтгаан 2: occlusion (orig #173)\nХугарлын гажилт ба сэргэлтийн ялгаа")
    ax_w[1].axvspan(100, 360, color="#38bdf8", alpha=0.15)
    tx, ty = _trajectory_points(gt_o, 131)
    ax_w[1].plot(tx, ty, color="#f87171", lw=2.0)
    _draw_box_only(ax_w[1], gt_o[131]["box"], "#ef4444", lw=2.5, fill_alpha=0.14)
    pills_b = [("GT алим гарч ирлээ", "#ef4444")]
    for tr in ("KCF", "CSRT", "TLD", "MEDIANFLOW"):
        r = _load_track_rows("occlusion", tr)[131]
        if int(r["valid_pred"]):
            box = (float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"]))
            _draw_box_only(ax_w[1], box, TRACKER_COLORS_HEX[tr], lw=1.9, ls="--")
            pills_b.append((f"{tr}: IoU={float(r['iou']):.2f}", TRACKER_COLORS_HEX[tr]))
    _draw_score_pills(ax_w[1], pills_b, anchor_xy=(0.04, 0.30), va="top", fontsize=7.4)
    ax_w[1].text(
        25,
        45,
        "Шалтгаан:\n"
        "• Босоо линз шил алимыг хэвтээ сунгадаг\n"
        "  (0% бүрэн алга болоогүй)\n"
        "• KCF/CSRT улаан алимыг 100% барьсан\n"
        "• TLD хугарсан текстурыг таниагүй\n"
        "• MedianFlow босоо зураасанд\n"
        "  LK алдаа өгч гацсан",
        color="#f8fafc",
        fontsize=7.6,
        va="top",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.95),
    )

    fig_w.tight_layout()
    fig_why = CUSTOM_FIG_DIR / "why_failed_schematic.png"
    fig_w.savefig(fig_why)
    plt.close(fig_w)
    return fig_comp, fig_why


def build_local_raw_qa_overlays() -> List[Path]:
    """Render LOCAL-ONLY raw-frame tracker overlays for visual QA (never committed to Git)."""
    CUSTOM_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    configs = [
        ("moving_camera", Path("data/phone_videos/moving_camera.mp4"), 405, 515, [1, 40, 76, 108], ["BOOSTING", "CSRT", "MIL", "MEDIANFLOW"], (640, 360), "moving_camera_qa_trackers.png"),
        ("occlusion", Path("data/phone_videos/occlusion.mp4"), 42, 235, [8, 52, 108, 140], ["KCF", "CSRT", "BOOSTING", "TLD", "MEDIANFLOW"], (360, 640), "occlusion_qa_trackers.png"),
    ]
    for seq, vp, sf, ef, idxs, trackers, sz, out_name in configs:
        frames, orig_idx, _ = load_video_clip(vp, sf, ef, max_frames=300)
        gt = load_verified_gt(CSV_DIR / f"phone_{seq}_gt.csv", orig_idx)
        loaded = {t: _load_track_rows(seq, t) for t in trackers}
        panels = []
        for ci in idxs:
            im = frames[ci].copy()
            x, y, w, h = (int(round(v)) for v in gt[ci]["box"])
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 3)
            for ti, tr in enumerate(trackers):
                r = loaded[tr][ci]
                if int(r["valid_pred"]):
                    bx, by, bw, bh = int(round(float(r["box_x"]))), int(round(float(r["box_y"]))), int(round(float(r["box_w"]))), int(round(float(r["box_h"])))
                    col = TRACKER_COLORS_BGR[tr]
                    cv2.rectangle(im, (bx, by), (bx + bw, by + bh), col, 2)
                    cv2.putText(im, f"{tr}:{float(r['iou']):.2f}", (max(8, bx), max(28, by - 8 + ti * 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.putText(im, f"{seq} c#{ci} o#{orig_idx[ci]}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            panels.append(cv2.resize(im, sz))
        grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]) if len(panels) == 4 and sz[0] > sz[1] else np.hstack(panels)
        out_p = CUSTOM_LOCAL_DIR / out_name
        cv2.imwrite(str(out_p), grid)
        saved.append(out_p)
    return saved


def build_all_custom_figures() -> List[Path]:
    ensure_dirs()
    CUSTOM_FIG_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_three_videos_overview_schematic(),
        build_stable_figure(),
        build_moving_camera_flow_figure(),
        *build_occlusion_storyboard_and_timeline(),
        *build_benchmark_comparison_and_why_failed(),
    ]
    qa = build_local_raw_qa_overlays()
    print("Generated publishable figures:", [str(p) for p in outputs])
    print("Generated local QA overlays:", [str(p) for p in qa])
    return outputs


if __name__ == "__main__":
    build_all_custom_figures()
