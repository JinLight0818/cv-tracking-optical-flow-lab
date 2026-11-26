#!/usr/bin/env python3
"""Controlled CC0 photo sequence and 8-tracker benchmark for squirrel tree-trunk full occlusion."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from benchmark_track import SequenceSpec, evaluate_tracker, load_sequence
from config import ALL_TRACKERS, CSV_DIR, DATA_DIR, enforce_single_thread, ensure_dirs
from dataset_prep import fetch_url

SQUIRREL_PAGE = "https://commons.wikimedia.org/wiki/File:Squirrel_eating_nut_in_grass.png"
SQUIRREL_IMG_URL = "https://thumb.wikimedia.org/wikipedia/commons/thumb/8/87/Squirrel_eating_nut_in_grass.png/3840px-Squirrel_eating_nut_in_grass.png"
TRUNK_PAGE = "https://commons.wikimedia.org/wiki/File:Squirrel_on_tree_trunk.jpg"
TRUNK_IMG_URL = "https://upload.wikimedia.org/wikipedia/commons/6/65/Squirrel_on_tree_trunk.jpg"


def _extract_squirrel_cutout(sq_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract squirrel foreground BGR + soft de-spilled alpha matte from CC0 photo."""
    patch = sq_full[360:2000, 1020:3080]
    work = cv2.resize(patch, (1030, 820), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    b, g, r = work[:, :, 0].astype(int), work[:, :, 1].astype(int), work[:, :, 2].astype(int)
    sure_bg_green = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 65) & (g > r + 14) & (g > b + 14)
    warm_non_green = ((r >= g - 2) & (r > b + 12) & (hsv[:, :, 2] > 35)) | ((hsv[:, :, 2] < 45) & (r >= g - 5))

    gc_mask = np.full(work.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[100:760, 140:960] = cv2.GC_PR_FGD
    gc_mask[sure_bg_green] = cv2.GC_BGD
    gc_mask[:60, :] = cv2.GC_BGD
    gc_mask[:, :80] = cv2.GC_BGD
    gc_mask[:, 975:] = cv2.GC_BGD
    core = np.zeros(work.shape[:2], dtype=bool)
    core[240:690, 420:860] = True
    gc_mask[core & warm_non_green] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(work, gc_mask, None, bgd_model, fgd_model, 6, cv2.GC_INIT_WITH_MASK)
    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    _, lbls, stats, _ = cv2.connectedComponentsWithStats(fg)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = (lbls == best).astype(np.uint8)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_filled = np.zeros_like(fg)
    cv2.drawContours(fg_filled, contours, -1, 1, -1)

    alpha = cv2.GaussianBlur(fg_filled.astype(np.float32), (5, 5), 1.0)
    alpha[fg_filled == 0] = np.minimum(alpha[fg_filled == 0], 0.35)
    bgr_clean = work.astype(np.float32)
    edge_band = (alpha > 0.02) & (alpha < 0.98)
    bgr_clean[:, :, 1] = np.where(
        edge_band & (bgr_clean[:, :, 1] > bgr_clean[:, :, 2]),
        0.75 * bgr_clean[:, :, 2] + 0.25 * bgr_clean[:, :, 1],
        bgr_clean[:, :, 1],
    )

    ys, xs = np.where(alpha > 0.15)
    x0, x1 = max(0, int(xs.min()) - 4), min(alpha.shape[1], int(xs.max()) + 5)
    y0, y1 = max(0, int(ys.min()) - 4), min(alpha.shape[0], int(ys.max()) + 5)
    sq_bgr = bgr_clean[y0:y1, x0:x1].astype(np.uint8)
    sq_alpha = alpha[y0:y1, x0:x1].copy()

    sh = sq_alpha.shape[0]
    bin_m = (sq_alpha > 0.45).astype(np.uint8)
    closed_m = cv2.morphologyEx(bin_m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 11)))
    bot_zone = np.zeros_like(sq_alpha)
    bot_zone[int(sh * 0.72) :, :] = 1.0
    merged_m = np.where(bot_zone > 0.5, np.maximum(sq_alpha, closed_m.astype(np.float32)), sq_alpha)
    need_fill = merged_m > sq_alpha + 0.1
    if np.any(need_fill):
        inpaint_bgr = cv2.inpaint(
            sq_bgr,
            ((merged_m > 0.2) & (sq_alpha <= 0.6)).astype(np.uint8) * 255,
            9,
            cv2.INPAINT_TELEA,
        )
        sq_bgr[need_fill] = inpaint_bgr[need_fill]
    sq_alpha = np.clip(cv2.GaussianBlur(merged_m, (5, 5), 0.9), 0.0, 1.0)
    taper_rows = np.ones(sh, dtype=np.float32)
    taper_rows[-14:] = np.linspace(1.0, 0.25, 14, dtype=np.float32)
    sq_alpha *= taper_rows[:, None]

    ys2, xs2 = np.where(sq_alpha > 0.10)
    return (
        sq_bgr[int(ys2.min()) : int(ys2.max()) + 1, int(xs2.min()) : int(xs2.max()) + 1],
        sq_alpha[int(ys2.min()) : int(ys2.max()) + 1, int(xs2.min()) : int(xs2.max()) + 1],
    )


def generate_squirrel_occlusion_sequence() -> Tuple[Path, Path, int]:
    """Generate controlled CC0 photo sequence where a squirrel crosses behind a tree trunk."""
    ensure_dirs()
    wiki_dir = DATA_DIR / "wikimedia"
    sq_path = fetch_url(SQUIRREL_IMG_URL, wiki_dir / "Squirrel_eating_nut_in_grass.png")
    tr_path = fetch_url(TRUNK_IMG_URL, wiki_dir / "Squirrel_on_tree_trunk.jpg")

    sq_src = cv2.imread(str(sq_path))
    trunk_src = cv2.imread(str(tr_path))
    if sq_src is None or trunk_src is None:
        raise RuntimeError("Missing CC0 source photos for squirrel occlusion sequence")

    sq_bgr, sq_alpha = _extract_squirrel_cutout(sq_src)

    W, H = 720, 420
    park_patch = trunk_src[0:1540, 0:1080]
    m_rot = cv2.getRotationMatrix2D((540, 770), 17.0, 1.0)
    rot_park = cv2.warpAffine(park_patch, m_rot, (1080, 1540), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    bg_left = cv2.resize(rot_park[320:1240, 90:820], (430, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    bg_right = cv2.resize(rot_park[310:1230, 230:960], (430, H), interpolation=cv2.INTER_AREA).astype(np.float32)

    bg = np.zeros((H, W, 3), dtype=np.float32)
    bg[:, :290] = bg_left[:, :290]
    w_bl = 0.5 * (1.0 - np.cos(math.pi * np.linspace(0, 1, 140, dtype=np.float32)))[None, :, None]
    bg[:, 290:430] = (1.0 - w_bl) * bg_left[:, 290:430] + w_bl * bg_right[:, :140]
    bg[:, 430:] = bg_right[:, 140:]
    bg = cv2.GaussianBlur(bg, (5, 5), 0.8)

    yy, xx = np.mgrid[0:H, 0:W]
    trunk_ground_shad = np.exp(-(((xx - 360) / 135.0) ** 2)) * np.clip((yy - 260) / 140.0, 0.0, 1.0)
    bg = np.clip(bg * (1.0 - 0.20 * trunk_ground_shad[..., None]), 0, 255).astype(np.uint8)

    pure_bark = cv2.resize(trunk_src[1200:2900, 1520:2680], (260, H), interpolation=cv2.INTER_AREA)
    row_f = np.linspace(0, 1, H, dtype=np.float32)
    trunk_cx, trunk_half_w = 362.0, 95.0
    trunk_xL = trunk_cx - trunk_half_w + 1.6 * np.sin(row_f * 23.0) + 1.0 * np.cos(row_f * 51.0 + 0.7)
    trunk_xR = trunk_cx + trunk_half_w + 1.5 * np.sin(row_f * 19.0 + 1.8) + 1.1 * np.cos(row_f * 47.0 + 2.1)

    trunk_alpha = np.zeros((H, W), dtype=np.float32)
    trunk_bgr = np.zeros((H, W, 3), dtype=np.uint8)
    for r in range(H):
        xl, xr = trunk_xL[r], trunk_xR[r]
        trunk_alpha[r] = np.minimum(
            np.clip(xx[r] - xl + 0.6, 0.0, 1.6) / 1.6,
            np.clip(xr - xx[r] + 0.6, 0.0, 1.6) / 1.6,
        )
        u = np.clip((xx[r] - xl) / max(1.0, xr - xl), 0.0, 1.0)
        shade = 0.74 + 0.28 * (np.sin(math.pi * u) ** 0.5)
        bx = np.clip((xx[r] - 250).astype(int), 0, pure_bark.shape[1] - 1)
        trunk_bgr[r] = np.clip(pure_bark[r, bx].astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    seq_dir = DATA_DIR / "controlled" / "squirrel_occ_seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for old in seq_dir.glob("frame_*.png"):
        old.unlink()
    gt_csv = CSV_DIR / "squirrel_occlusion_gt.csv"

    n_frames = 66
    base_w = 122
    base_h = int(round(base_w * sq_bgr.shape[0] / sq_bgr.shape[1]))
    x_start, vx, y_ground = 42.0, 7.95, 342.0
    occ_frames: List[int] = []
    reappear_frame = -1
    prev_x = prev_y = None

    with open(gt_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "frame", "image_file", "present",
            "vis_x", "vis_y", "vis_w", "vis_h",
            "amodal_x", "amodal_y", "amodal_w", "amodal_h",
            "visible_fraction", "dx_gt", "dy_gt", "scale_factor",
        ])
        for i in range(n_frames):
            scale = 0.96 + 0.12 * (i / (n_frames - 1))
            tw, th = int(round(base_w * scale)), int(round(base_h * scale))
            p_bgr = cv2.resize(sq_bgr, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
            p_alp = cv2.resize(sq_alpha, (tw, th), interpolation=cv2.INTER_LINEAR)
            hop = 5.0 * abs(math.sin(i * 0.48))
            tx = int(round(x_start + i * vx))
            ty = int(round(y_ground - th - hop))

            frame = bg.copy().astype(np.float32)
            shad_cx, shad_cy = tx + tw * 0.52, int(round(y_ground - 3))
            s_gauss = np.exp(-(((xx - shad_cx) / (tw * 0.36)) ** 2 + ((yy - shad_cy) / 7.5) ** 2))
            frame = frame * (1.0 - 0.32 * s_gauss[..., None])

            sx0, sy0 = max(0, tx), max(0, ty)
            sx1, sy1 = min(W, tx + tw), min(H, ty + th)
            sq_mask_canvas = np.zeros((H, W), dtype=np.float32)
            if sx0 < sx1 and sy0 < sy1:
                sub_alp = p_alp[sy0 - ty : sy1 - ty, sx0 - tx : sx1 - tx]
                sub_bgr = p_bgr[sy0 - ty : sy1 - ty, sx0 - tx : sx1 - tx]
                sq_mask_canvas[sy0:sy1, sx0:sx1] = sub_alp
                frame[sy0:sy1, sx0:sx1] = (
                    sub_alp[..., None] * sub_bgr + (1.0 - sub_alp[..., None]) * frame[sy0:sy1, sx0:sx1]
                )

            t_a = trunk_alpha[..., None]
            out_u8 = np.clip(t_a * trunk_bgr.astype(np.float32) + (1.0 - t_a) * frame, 0, 255).astype(np.uint8)

            vis_bin = (sq_mask_canvas * (1.0 - trunk_alpha)) > 0.22
            amodal_bin = sq_mask_canvas > 0.22
            vis_frac = int(vis_bin.sum()) / max(1, int(amodal_bin.sum()))
            ys_v, xs_v = np.where(vis_bin)
            if xs_v.size < 25:
                present, vis_box = 0, (-1, -1, 0, 0)
                occ_frames.append(i)
            else:
                present = 1
                vis_box = (
                    int(xs_v.min()),
                    int(ys_v.min()),
                    int(xs_v.max() - xs_v.min() + 1),
                    int(ys_v.max() - ys_v.min() + 1),
                )
                if occ_frames and reappear_frame == -1:
                    reappear_frame = i

            dx_gt = 0.0 if prev_x is None else float(tx - prev_x)
            dy_gt = 0.0 if prev_y is None else float(ty - prev_y)
            prev_x, prev_y = tx, ty
            fname = f"frame_{i:04d}.png"
            cv2.imwrite(str(seq_dir / fname), out_u8)
            wcsv.writerow([
                i, fname, present, *vis_box, tx, ty, tw, th,
                round(vis_frac, 4), dx_gt, dy_gt, round(scale, 4),
            ])
    return seq_dir, gt_csv, reappear_frame


def run_squirrel_occlusion_benchmark(rebuild_seq: bool = False) -> Path:
    """Evaluate all 8 trackers cleanly on squirrel_occlusion and write summary + per-frame CSVs."""
    enforce_single_thread()
    ensure_dirs()
    seq_dir = DATA_DIR / "controlled" / "squirrel_occ_seq"
    gt_csv = CSV_DIR / "squirrel_occlusion_gt.csv"
    if rebuild_seq or not gt_csv.exists():
        seq_dir, gt_csv, reappear_frame = generate_squirrel_occlusion_sequence()
    else:
        with open(gt_csv, newline="") as f:
            rows_gt = list(csv.DictReader(f))
        occ = [int(r["frame"]) for r in rows_gt if int(r["present"]) == 0]
        reappear_frame = next(int(r["frame"]) for r in rows_gt if int(r["frame"]) > max(occ) and int(r["present"]) == 1)

    spec = SequenceSpec(
        name="squirrel_occlusion",
        gt_csv=gt_csv,
        img_dir=seq_dir,
        reappear_frame=reappear_frame,
        recovery_kind="full_occlusion",
        description="CC0 photo controlled: running squirrel + scale + tree trunk 9-frame 0% full occlusion",
    )
    img_paths, frames, gt_rows = load_sequence(spec)
    summaries: List[Dict] = []
    for tr in ALL_TRACKERS:
        summ, rows = evaluate_tracker(spec, tr, img_paths, frames, gt_rows, measure_e2e=True)
        summaries.append(summ)
        tr_csv = CSV_DIR / f"track_squirrel_occlusion_{tr}.csv"
        with open(tr_csv, "w", newline="") as tf:
            w = csv.DictWriter(tf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    sum_csv = CSV_DIR / "squirrel_occlusion_summary.csv"
    with open(sum_csv, "w", newline="") as sf:
        w = csv.DictWriter(sf, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    return sum_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-seq", action="store_true")
    args = ap.parse_args()
    out = run_squirrel_occlusion_benchmark(rebuild_seq=args.rebuild_seq)
    print("Wrote", out)
