#!/usr/bin/env python3
"""Generate lab figures and tracking GIFs for the README."""

from __future__ import annotations

import csv
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_flow import compute_camera_motion_compensation
from config import (
    ALL_TRACKERS,
    CSV_DIR,
    DATA_DIR,
    FARNEBACK_VIS_MAX_MAG,
    FIGURES_DIR,
    LOCAL_VIZ_DIR,
    ensure_dirs,
)
from custom_video_lab import load_verified_gt, load_video_clip
from opt_flow import draw_hsv_fixed, make_color_wheel

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

CUSTOM_FIG_DIR = FIGURES_DIR / "custom_videos"
CUSTOM_LOCAL_DIR = LOCAL_VIZ_DIR / "custom_videos"

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

TRACKER_COLORS_HEX = {
    "CSRT": "#059669",
    "CSRT_REDETECT": "#0284c7",
    "KCF": "#ea580c",
    "BOOSTING": "#7c3aed",
    "MIL": "#db2777",
    "TLD": "#dc2626",
    "MEDIANFLOW": "#0d9488",
    "STATIC": "#64748b",
}

PHONE_GIF_FPS = 11
PHONE_GIFS = {
    # Start, end, first tracker, crop, width, palette size, key source frames.
    "stable": (664, 721, "KCF", (1300, 258, 1920, 626), 512, 128, (693,)),
    "moving_camera": (405, 515, "BOOSTING", (470, 238, 1448, 786), 480, 80, (451, 460)),
    "occlusion": (42, 235, "KCF", (0, 425, 720, 1112), 384, 64, (85, 139, 172)),
}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_track_rows(seq: str, tracker: str) -> Dict[int, Dict[str, str]]:
    path = CSV_DIR / f"track_{seq}_{tracker}.csv"
    with open(path, newline="") as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: Tuple[int, int, int] = (255, 255, 255),
    width: int = 2,
    dash: int = 7,
    gap: int = 5,
) -> None:
    """Draw a clean thin dashed rectangle on Pillow canvas."""
    x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    segments = [
        ((x1i, y1i), (x2i, y1i)),
        ((x2i, y1i), (x2i, y2i)),
        ((x2i, y2i), (x1i, y2i)),
        ((x1i, y2i), (x1i, y1i)),
    ]
    for (ax, ay), (bx, by) in segments:
        dx, dy = bx - ax, by - ay
        length = float(np.hypot(dx, dy))
        if length < 1:
            continue
        ux, uy = dx / length, dy / length
        pos = 0.0
        while pos < length:
            end = min(length, pos + dash)
            sx, sy = int(round(ax + ux * pos)), int(round(ay + uy * pos))
            ex, ey = int(round(ax + ux * end)), int(round(ay + uy * end))
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos += dash + gap


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    bg: Tuple[int, int, int],
    fg: Tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = xy
    draw.rectangle([x, y, x + tw + 10, y + th + 6], fill=bg)
    draw.text((x + 5, y + 2), text, fill=fg, font=font)


def _draw_corner_brackets(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: Tuple[int, int, int],
    arm: int = 10,
    width: int = 3,
) -> None:
    xi1, yi1, xi2, yi2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    for cx, cy, sx, sy in [(xi1, yi1, 1, 1), (xi2, yi1, -1, 1), (xi1, yi2, 1, -1), (xi2, yi2, -1, -1)]:
        draw.line([(cx, cy), (cx + sx * arm, cy)], fill=color, width=width)
        draw.line([(cx, cy), (cx, cy + sy * arm)], fill=color, width=width)


def _phone_rows(path: Path) -> Dict[int, Dict[str, str]]:
    rows = _read_csv_rows(path)
    indexed = {int(row["orig_frame"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate orig_frame in {path}")
    return indexed


def _phone_box(row: Dict[str, str], flag: str, prefix: str = "") -> Tuple[float, float, float, float] | None:
    if row.get(flag) != "1":
        return None
    try:
        x, y, w, h = (float(row[f"{prefix}{axis}"]) for axis in "xywh")
    except (KeyError, ValueError):
        return None
    if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
        return None
    return x, y, w, h


def build_phone_gifs() -> None:
    """Sample the measured clips for playback with frame-matched CSV overlays."""
    header = 42
    font = ImageFont.truetype(FONT_BOLD, 21)
    white, orange, green = (255, 255, 255), (255, 164, 58), (98, 247, 172)
    CUSTOM_FIG_DIR.mkdir(parents=True, exist_ok=True)

    for seq, (start, end, tracker, crop, width, colors, key_frames) in PHONE_GIFS.items():
        gt = _phone_rows(CSV_DIR / f"phone_{seq}_gt.csv")
        first = _phone_rows(CSV_DIR / f"track_{seq}_{tracker}.csv")
        csrt = _phone_rows(CSV_DIR / f"track_{seq}_CSRT.csv")
        frames, _, source_fps = load_video_clip(DATA_DIR / "phone_videos" / f"{seq}.mp4", start, end)
        duration = len(frames) / source_fps
        required = {start, end, *key_frames}
        sample_count = max(len(required), round(duration * PHONE_GIF_FPS))
        sampled = set(np.linspace(start, end, sample_count).round().astype(int))
        # Keep key frames by replacing nearby samples without extending playback.
        for key in sorted(required - sampled):
            nearest = min(sampled - required, key=lambda orig: (abs(orig - key), orig))
            sampled.remove(nearest)
            sampled.add(key)
        playback_fps = len(sampled) / duration
        left, top, right, bottom = crop
        height = round((bottom - top) * width / (right - left))
        sx, sy = width / (right - left), height / (bottom - top)
        output = CUSTOM_FIG_DIR / f"tracking_{seq}.gif"
        csrt_label_x = width - 83
        tracker_label_x = csrt_label_x - round(font.getlength(tracker)) - 20
        gt_label_x = tracker_label_x - round(font.getlength("GT")) - 20

        with tempfile.TemporaryDirectory(dir=CUSTOM_FIG_DIR) as tmpdir:
            tmp = Path(tmpdir)
            for index, orig in enumerate(sorted(sampled)):
                frame = frames[orig - start]
                rgb = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                canvas = Image.new("RGB", (width, height + header), (18, 24, 33))
                canvas.paste(Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS), (0, header))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 9), f"#{orig}", font=font, fill=white)
                draw.text((gt_label_x, 9), "GT", font=font, fill=white)
                draw.text((tracker_label_x, 9), tracker, font=font, fill=orange)
                draw.text((csrt_label_x, 9), "CSRT", font=font, fill=green)

                boxes = (
                    (_phone_box(first.get(orig, {}), "valid_pred", "box_"), orange),
                    (_phone_box(gt.get(orig, {}), "present"), white),
                    (_phone_box(csrt.get(orig, {}), "valid_pred", "box_"), green),
                )
                for box, color in boxes:
                    if box is None:
                        continue
                    x, y, w, h = box
                    bounds = (
                        round((x - left) * sx),
                        round((y - top) * sy) + header,
                        min(width - 1, round((x + w - left) * sx)),
                        round((y + h - top) * sy) + header,
                    )
                    if color == white:
                        _draw_dashed_rect(draw, *bounds, color=white, width=2)
                    elif color == green:
                        _draw_corner_brackets(draw, *bounds, color=green, arm=12, width=3)
                    else:
                        draw.rectangle(bounds, outline=orange, width=2)
                canvas.save(tmp / f"frame_{index:03d}.png", compress_level=1)

            ffmpeg = ["ffmpeg", "-v", "error", "-y", "-filter_complex_threads", "1"]
            subprocess.run(
                ffmpeg + ["-framerate", f"{playback_fps:.6f}", "-i", str(tmp / "frame_%03d.png"),
                          "-filter_complex",
                          f"split[video][colors];[colors]palettegen=max_colors={colors}[pal];"
                          "[video][pal]paletteuse=dither=none:diff_mode=rectangle",
                          "-loop", "0", str(tmp / output.name)],
                check=True,
            )
            (tmp / output.name).replace(output)
        print(f"{output} ({len(sampled)} GIF frames from {len(frames)} source frames)")


def _render_leman_split_frame(
    frame_bgr: np.ndarray,
    ci: int,
    orig_f: int,
    fps: float,
    gt_row: Dict,
    kcf_row: Dict[str, str],
    csrt_row: Dict[str, str],
) -> Image.Image:
    """Render synchronized 1:1 cropped side-by-side frame (Left=GT, Right=KCF vs CSRT)."""
    crop_x, crop_y, crop_w, crop_h = 330, 125, 590, 395
    panel_w, panel_h = 330, 221
    sx, sy = panel_w / crop_w, panel_h / crop_h

    crop_rgb = cv2.cvtColor(frame_bgr[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w], cv2.COLOR_BGR2RGB)
    base_panel = Image.fromarray(crop_rgb).resize((panel_w, panel_h), Image.Resampling.LANCZOS)

    font_hdr = ImageFont.truetype(FONT_BOLD, 12)
    font_sub = ImageFont.truetype(FONT_REG, 10)
    font_badge = ImageFont.truetype(FONT_BOLD, 10)

    pad = 8
    top_h = 40
    bot_h = 24
    canvas_w = pad * 3 + panel_w * 2
    canvas_h = top_h + panel_h + bot_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    t_sec = ci / fps
    left_img = base_panel.copy()
    dl = ImageDraw.Draw(left_img)
    gx, gy, gw, gh = gt_row["box"]
    lx1, ly1 = (gx - crop_x) * sx, (gy - crop_y) * sy
    lx2, ly2 = lx1 + gw * sx, ly1 + gh * sy
    _draw_dashed_rect(dl, lx1 + 1, ly1 + 1, lx2 + 1, ly2 + 1, color=(15, 23, 42), width=3, dash=7, gap=5)
    _draw_dashed_rect(dl, lx1, ly1, lx2, ly2, color=(255, 255, 255), width=2, dash=7, gap=5)
    _draw_pill(dl, (max(4, int(lx1)), max(4, int(ly1) - 20)), "GT", (15, 23, 42), (255, 255, 255), font_badge)

    right_img = base_panel.copy()
    dr = ImageDraw.Draw(right_img)
    # Draw subtle GT reference too? Keep right clean for KCF vs CSRT, draw CSRT (emerald) & KCF (orange)
    csrt_ok = int(csrt_row["valid_pred"]) == 1 and int(csrt_row["success"]) == 1
    kcf_ok = int(kcf_row["valid_pred"]) == 1 and int(kcf_row["success"]) == 1

    if csrt_ok:
        cx, cy, cw, ch = float(csrt_row["box_x"]), float(csrt_row["box_y"]), float(csrt_row["box_w"]), float(csrt_row["box_h"])
        rx1, ry1 = (cx - crop_x) * sx, (cy - crop_y) * sy
        rx2, ry2 = rx1 + cw * sx, ry1 + ch * sy
        dr.rectangle([int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2))], outline=(16, 185, 129), width=3)
        _draw_corner_brackets(dr, rx1, ry1, rx2, ry2, (236, 253, 245), arm=7, width=2)
        _draw_pill(dr, (max(4, int(rx1)), max(4, int(ry1) - 20)), f"CSRT {float(csrt_row['iou']):.2f}", (5, 150, 105), (255, 255, 255), font_badge)
    else:
        _draw_pill(dr, (10, 10), "CSRT: Алдсан", (185, 28, 28), (255, 255, 255), font_badge)

    if kcf_ok:
        kx, ky, kw, kh = float(kcf_row["box_x"]), float(kcf_row["box_y"]), float(kcf_row["box_w"]), float(kcf_row["box_h"])
        kx1, ky1 = (kx - crop_x) * sx, (ky - crop_y) * sy
        kx2, ky2 = kx1 + kw * sx, ky1 + kh * sy
        dr.rectangle([int(round(kx1)), int(round(ky1)), int(round(kx2)), int(round(ky2))], outline=(249, 115, 22), width=2)
        # Distinct center '+' mark for colorblind shape separation
        kmx, kmy = int(round((kx1 + kx2) * 0.5)), int(round((ky1 + ky2) * 0.5))
        dr.line([(kmx - 5, kmy), (kmx + 5, kmy)], fill=(249, 115, 22), width=2)
        dr.line([(kmx, kmy - 5), (kmx, kmy + 5)], fill=(249, 115, 22), width=2)
        y_kcf_lbl = min(panel_h - 22, int(round(ky2)) + 3)
        _draw_pill(dr, (max(4, int(kx1)), y_kcf_lbl), f"KCF {float(kcf_row['iou']):.2f}", (234, 88, 12), (255, 255, 255), font_badge)
    else:
        _draw_pill(dr, (10, 34), "KCF: Алдсан", (185, 28, 28), (255, 255, 255), font_badge)

    canvas.paste(left_img, (pad, top_h))
    canvas.paste(right_img, (pad * 2 + panel_w, top_h))

    # Header titles and synchronized frame/time
    draw.text((pad, 8), "Бодит кадр + GT (гар шошго)", fill=(15, 23, 42), font=font_hdr)
    draw.text((pad, 25), f"CC0 Léman Express  |  #{orig_f}  ({t_sec:.2f}с)", fill=(71, 85, 105), font=font_sub)

    rx = pad * 2 + panel_w
    draw.text((rx, 8), "KCF (улбар шар +) vs CSRT (ногоон [ ])", fill=(15, 23, 42), font=font_hdr)
    draw.text(
        (rx, 25),
        f"#{orig_f} ({t_sec:.2f}с)  |  IoU: KCF={float(kcf_row['iou']):.2f}  CSRT={float(csrt_row['iou']):.2f}",
        fill=(71, 85, 105),
        font=font_sub,
    )

    # Bottom 1-line consistent legend strip
    draw.line([(pad, top_h + panel_h + 4), (canvas_w - pad, top_h + panel_h + 4)], fill=(226, 232, 240), width=1)
    draw.text(
        (pad, top_h + panel_h + 8),
        "Тэмдэглэгээ:  -- Цагаан = GT    [+] Улбар шар = KCF (тогтмол хэмжээ)    [ ] Ногоон = CSRT (хэмжээ тэлдэг)",
        fill=(51, 65, 85),
        font=font_sub,
    )
    return canvas


def build_cc0_hero_gif_and_keyframes() -> Tuple[Path, Path]:
    """Build real-camera CC0 tracking_demo.gif (1.0x real-time @ 10fps) and static 3-frame strip."""
    video_path = DATA_DIR / "commons" / "leman_express_cc0.webm"
    frames, orig_idx, fps = load_video_clip(video_path, start_frame=110, end_frame=260, max_frames=300)
    gt = load_verified_gt(CSV_DIR / "commons_leman_express_gt.csv", orig_idx)
    kcf = _load_track_rows("commons_leman_express", "KCF")
    csrt = _load_track_rows("commons_leman_express", "CSRT")

    # Export 10 FPS (step 3 on 30 fps) -> exact 1.0x real-time 5.00s loop!
    gif_path = CUSTOM_FIG_DIR / "tracking_demo.gif"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sampled = list(range(0, len(frames), 3))
        for seq_i, ci in enumerate(sampled):
            img = _render_leman_split_frame(frames[ci], ci, orig_idx[ci], fps, gt[ci], kcf[ci], csrt[ci])
            img.save(tmp / f"f_{seq_i:04d}.png")

        pal = tmp / "palette.png"
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "10", "-i", str(tmp / "f_%04d.png"), "-vf", "palettegen=stats_mode=diff:max_colors=160", str(pal)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "10", "-i", str(tmp / "f_%04d.png"), "-i", str(pal), "-lavfi", "paletteuse=dither=bayer:bayer_scale=2:diff_mode=rectangle", str(gif_path)],
            check=True,
            capture_output=True,
        )

    # Companion static keyframe comparison strip (0.00s, 2.50s, 5.00s)
    kf_path = CUSTOM_FIG_DIR / "tracking_demo_keyframes.png"
    strip_indices = [0, 75, 150]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=160, facecolor="white")
    crop_x, crop_y, crop_w, crop_h = 330, 125, 590, 395
    for ax, ci in zip(axes, strip_indices):
        rgb = cv2.cvtColor(frames[ci][crop_y : crop_y + crop_h, crop_x : crop_x + crop_w], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        gx, gy, gw, gh = gt[ci]["box"]
        ax.add_patch(patches.Rectangle((gx - crop_x, gy - crop_y), gw, gh, fill=False, edgecolor="#0f172a", lw=3.0, ls=(0, (5, 4))))
        ax.add_patch(patches.Rectangle((gx - crop_x, gy - crop_y), gw, gh, fill=False, edgecolor="#ffffff", lw=1.8, ls=(0, (5, 4)), label="GT"))
        for tr, col, ls_style in [("CSRT", "#059669", "-"), ("KCF", "#ea580c", "-")]:
            r = (csrt if tr == "CSRT" else kcf)[ci]
            bx, by, bw, bh = float(r["box_x"]), float(r["box_y"]), float(r["box_w"]), float(r["box_h"])
            ax.add_patch(patches.Rectangle((bx - crop_x, by - crop_y), bw, bh, fill=False, edgecolor=col, lw=2.3, ls=ls_style))
        ax.set_title(
            f"orig #{orig_idx[ci]} ({ci / fps:.2f}с)  |  KCF={float(kcf[ci]['iou']):.2f}, CSRT={float(csrt[ci]['iou']):.2f}",
            fontsize=9.2,
            fontweight="bold",
            color="#0f172a",
        )
        ax.axis("off")

    fig.suptitle(
        "Хөдөлгөөнгүй бодит камерт ойртож буй галт тэрэг (CC0 Léman Express): -- Цагаан = GT   Улбар шар = KCF   Ногоон = CSRT",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(kf_path, facecolor="white")
    plt.close(fig)
    print(f"Saved Hero GIF ({gif_path.stat().st_size / 1e6:.2f} MB): {gif_path}")
    return gif_path, kf_path


def _load_squirrel_gt_rows() -> Dict[int, Dict[str, str]]:
    return {int(r["frame"]): r for r in _read_csv_rows(CSV_DIR / "squirrel_occlusion_gt.csv")}


def _render_squirrel_split_frame(
    frame_bgr: np.ndarray,
    fi: int,
    fps: float,
    gt_row: Dict[str, str],
    kcf_row: Dict[str, str],
    tld_row: Dict[str, str],
) -> Image.Image:
    """Render synchronized 1:1 full-frame side-by-side panels (Left=KCF, Right=TLD)."""
    ih, iw = frame_bgr.shape[:2]
    panel_w, panel_h = 340, int(round(340 * ih / iw))  # 340x198
    sx, sy = panel_w / iw, panel_h / ih

    base_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base_panel = Image.fromarray(base_rgb).resize((panel_w, panel_h), Image.Resampling.LANCZOS)

    font_hdr = ImageFont.truetype(FONT_BOLD, 12)
    font_sub = ImageFont.truetype(FONT_REG, 10)
    font_badge = ImageFont.truetype(FONT_BOLD, 11)

    pad = 8
    top_h = 44
    bot_h = 26
    canvas_w = pad * 3 + panel_w * 2
    canvas_h = top_h + panel_h + bot_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    present = int(gt_row["present"]) == 1
    visible_fraction = float(gt_row["visible_fraction"])
    vis_pct = int(round(visible_fraction * 100))
    absent_label = "Бүрэн далд" if visible_fraction == 0 else "Үнэлгээнд далд"
    gx, gy, gw, gh = float(gt_row["vis_x"]), float(gt_row["vis_y"]), float(gt_row["vis_w"]), float(gt_row["vis_h"])

    def draw_gt(d_ctx: ImageDraw.ImageDraw) -> None:
        if not present:
            return
        x1, y1 = gx * sx, gy * sy
        x2, y2 = x1 + gw * sx, y1 + gh * sy
        _draw_dashed_rect(d_ctx, x1 + 1, y1 + 1, x2 + 1, y2 + 1, color=(15, 23, 42), width=3, dash=6, gap=4)
        _draw_dashed_rect(d_ctx, x1, y1, x2, y2, color=(255, 255, 255), width=2, dash=6, gap=4)

    # Left panel: KCF (reports Алдсан cleanly on occlusion -> 0/9 absent FP, never recovers)
    left_img = base_panel.copy()
    dl = ImageDraw.Draw(left_img)
    draw_gt(dl)
    kcf_ok = int(kcf_row["valid_pred"]) == 1 and int(kcf_row["success"]) == 1
    if kcf_ok:
        kx, ky, kw, kh = float(kcf_row["box_x"]), float(kcf_row["box_y"]), float(kcf_row["box_w"]), float(kcf_row["box_h"])
        x1, y1, x2, y2 = kx * sx, ky * sy, (kx + kw) * sx, (ky + kh) * sy
        dl.rectangle([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))], outline=(249, 115, 22), width=3)
        lbl = f"KCF {float(kcf_row['iou']):.2f}" if present else "KCF: Буруу хүрээ"
        _draw_pill(dl, (max(4, int(x1)), max(4, int(y1) - 21)), lbl, (234, 88, 12), (255, 255, 255), font_badge)
    else:
        _draw_pill(dl, (8, 8), "KCF: Алдсан", (185, 28, 28), (255, 255, 255), font_badge)
    if not present:
        _draw_pill(dl, (8, panel_h - 24), absent_label, (30, 41, 59), (255, 255, 255), font_badge)

    # Right panel: TLD (recovers after trunk at #49, confirmed 3-frame >=0.5 at #51)
    right_img = base_panel.copy()
    dr = ImageDraw.Draw(right_img)
    draw_gt(dr)
    tld_ok = int(tld_row["valid_pred"]) == 1 and int(tld_row["success"]) == 1
    tld_iou = float(tld_row["iou"])
    if tld_ok:
        tx, ty, tw, th = float(tld_row["box_x"]), float(tld_row["box_y"]), float(tld_row["box_w"]), float(tld_row["box_h"])
        x1, y1, x2, y2 = tx * sx, ty * sy, (tx + tw) * sx, (ty + th) * sy
        if present and tld_iou >= 0.5:
            box_col, pill_bg = (2, 132, 199), (2, 132, 199)
            tag = f"TLD {tld_iou:.2f} (Сэргэсэн)" if fi >= 49 else f"TLD {tld_iou:.2f}"
        elif present:
            box_col, pill_bg = (220, 38, 38), (185, 28, 28)
            tag = f"TLD {tld_iou:.2f}"
        else:
            box_col, pill_bg = (220, 38, 38), (185, 28, 28)
            tag = "TLD: Буруу хүрээ"
        dr.rectangle([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))], outline=box_col, width=3)
        _draw_corner_brackets(dr, x1, y1, x2, y2, (255, 255, 255) if tld_iou >= 0.5 else (254, 202, 202), arm=6, width=2)
        tw_px = dr.textbbox((0, 0), tag, font=font_badge)[2] + 12
        px = min(panel_w - tw_px - 4, max(4, int(x1)))
        py = max(4, int(y1) - 21)
        _draw_pill(dr, (px, py), tag, pill_bg, (255, 255, 255), font_badge)
    else:
        _draw_pill(dr, (8, 8), "TLD: Алдсан", (185, 28, 28), (255, 255, 255), font_badge)
    if not present:
        _draw_pill(dr, (8, panel_h - 24), absent_label, (30, 41, 59), (255, 255, 255), font_badge)

    canvas.paste(left_img, (pad, top_h))
    canvas.paste(right_img, (pad * 2 + panel_w, top_h))

    t_sec = fi / fps
    state_str = absent_label if not present else (f"Харагдалт {vis_pct}%" if vis_pct < 98 else "Цэвэр харагдаж байна")
    draw.text((pad, 6), "KCF (улбар шар): далд ороод алдсан", fill=(15, 23, 42), font=font_hdr)
    draw.text((pad, 24), f"#{fi:02d}/65 ({t_sec:.2f}с)  |  {state_str}  |  FP=0/9", fill=(71, 85, 105), font=font_sub)

    rx = pad * 2 + panel_w
    draw.text((rx, 6), "TLD (цэнхэр): нөгөө талд эргэж олсон", fill=(15, 23, 42), font=font_hdr)
    tld_stat = "Хуурамч хүрээ (холтост)" if not present else f"IoU={tld_iou:.2f}" + (" [Батлагдсан #51]" if fi >= 51 else "")
    draw.text((rx, 24), f"#{fi:02d}/65 ({t_sec:.2f}с)  |  {tld_stat}", fill=(71, 85, 105), font=font_sub)

    draw.line([(pad, top_h + panel_h + 4), (canvas_w - pad, top_h + panel_h + 4)], fill=(226, 232, 240), width=1)
    draw.text(
        (pad, top_h + panel_h + 7),
        "Бодит CC0 фото дараалал:  -- Цагаан = GT   Улбар шар = KCF   Цэнхэр = TLD (#49 олоод #51 батлав)",
        fill=(51, 65, 85),
        font=font_sub,
    )
    return canvas


def build_squirrel_occlusion_gif_and_keyframes() -> Tuple[Path, Path]:
    """Build squirrel_occlusion.gif (12 FPS, 5.50s loop) and static 3-frame comparison PNG."""
    seq_dir = DATA_DIR / "controlled" / "squirrel_occ_seq"
    gt = _load_squirrel_gt_rows()
    kcf = _load_track_rows("squirrel_occlusion", "KCF")
    csrt = _load_track_rows("squirrel_occlusion", "CSRT")
    tld = _load_track_rows("squirrel_occlusion", "TLD")

    fps = 12.0
    gif_path = CUSTOM_FIG_DIR / "squirrel_occlusion.gif"
    n_frames = len(gt)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for fi in range(n_frames):
            bgr = cv2.imread(str(seq_dir / gt[fi]["image_file"]))
            img = _render_squirrel_split_frame(bgr, fi, fps, gt[fi], kcf[fi], tld[fi])
            img.save(tmp / f"f_{fi:04d}.png")

        pal = tmp / "palette.png"
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(int(fps)), "-i", str(tmp / "f_%04d.png"), "-vf", "palettegen=stats_mode=diff:max_colors=192", str(pal)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(int(fps)), "-i", str(tmp / "f_%04d.png"), "-i", str(pal), "-lavfi", "paletteuse=dither=bayer:bayer_scale=2:diff_mode=rectangle", str(gif_path)],
            check=True,
            capture_output=True,
        )

    # Equal-width 3-frame static strip: visible (#15), completely hidden (#32), reappeared+confirmed (#51)
    kf_path = CUSTOM_FIG_DIR / "squirrel_occlusion_keyframes.png"
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.15), dpi=160, facecolor="white")
    panels = [
        (15, "1. Харагдаж байна (#15, 1.25с)"),
        (32, "2. Бүрэн далд 0% (#32, 2.67с)"),
        (51, "3. Буцаж гарсан (#51, +14 кадр)"),
    ]
    for ax, (fi, hdr) in zip(axes, panels):
        rgb = cv2.cvtColor(cv2.imread(str(seq_dir / gt[fi]["image_file"])), cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        pres = int(gt[fi]["present"]) == 1
        if pres:
            gx, gy, gw, gh = float(gt[fi]["vis_x"]), float(gt[fi]["vis_y"]), float(gt[fi]["vis_w"]), float(gt[fi]["vis_h"])
            ax.add_patch(patches.Rectangle((gx, gy), gw, gh, fill=False, edgecolor="#0f172a", lw=3.0, ls=(0, (5, 4))))
            ax.add_patch(patches.Rectangle((gx, gy), gw, gh, fill=False, edgecolor="#ffffff", lw=1.8, ls=(0, (5, 4))))

        # KCF, CSRT, TLD boxes (only when valid_pred == 1 and success == 1)
        for tr, col, lw_v in [("KCF", TRACKER_COLORS_HEX["KCF"], 2.2), ("CSRT", TRACKER_COLORS_HEX["CSRT"], 2.2), ("TLD", TRACKER_COLORS_HEX["CSRT_REDETECT"], 2.5)]:
            row = {"KCF": kcf, "CSRT": csrt, "TLD": tld}[tr][fi]
            if int(row["valid_pred"]) == 1 and int(row["success"]) == 1:
                bx, by, bw, bh = float(row["box_x"]), float(row["box_y"]), float(row["box_w"]), float(row["box_h"])
                ax.add_patch(patches.Rectangle((bx, by), bw, bh, fill=False, edgecolor=col, lw=lw_v))

        if fi == 15:
            ax.text(14, 34, f"KCF={float(kcf[fi]['iou']):.2f}  CSRT={float(csrt[fi]['iou']):.2f}  TLD={float(tld[fi]['iou']):.2f}", color="#ffffff", fontsize=8.4, fontweight="bold", bbox=dict(facecolor="#0f172a", alpha=0.86))
        elif fi == 32:
            ax.text(14, 34, "KCF: Алдсан (0/9 FP)", color="#ffffff", fontsize=8.4, fontweight="bold", bbox=dict(facecolor="#b91c1c", alpha=0.92))
            ax.text(14, 70, "CSRT (ногоон), TLD (цэнхэр): Холтост хуурамч", color="#ffffff", fontsize=8.0, fontweight="bold", bbox=dict(facecolor="#475569", alpha=0.90))
        else:
            ax.text(14, 34, f"TLD (цэнхэр): IoU={float(tld[fi]['iou']):.2f} (Сэргэсэн)", color="#ffffff", fontsize=8.4, fontweight="bold", bbox=dict(facecolor="#0284c7", alpha=0.94))
            ax.text(14, 70, f"CSRT (ногоон): {float(csrt[fi]['iou']):.2f} (гацсан) | KCF: Алдсан", color="#ffffff", fontsize=8.0, fontweight="bold", bbox=dict(facecolor="#059669", alpha=0.92))

        ax.set_title(hdr, fontsize=9.6, fontweight="bold", color="#0f172a")
        ax.axis("off")

    fig.suptitle(
        "Бодит CC0 фото ашигласан хяналттай дараалал (хэрэм + их бие): -- Цагаан = GT   Улбар шар = KCF   Ногоон = CSRT   Цэнхэр = TLD",
        fontsize=10.2,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(kf_path, facecolor="white")
    plt.close(fig)
    print(f"Saved Squirrel GIF ({gif_path.stat().st_size / 1e6:.2f} MB) & keyframes: {gif_path}, {kf_path}")
    return gif_path, kf_path


def build_stable_scale_figure() -> Path:
    """Clean white lab figure for stable video (scale expansion vs IoU across orig #664-#721)."""
    orig_idx = list(range(664, 722))
    gt = load_verified_gt(CSV_DIR / "phone_stable_gt.csv", orig_idx)
    rows = [r for r in _read_csv_rows(CSV_DIR / "phone_videos_summary.csv") if r["sequence"] == "stable"]
    frames_x = list(range(1, len(gt)))
    areas = [g["box"][2] * g["box"][3] / (gt[0]["box"][2] * gt[0]["box"][3]) for g in gt[1:]]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), dpi=160, facecolor="white")

    ax0 = axes[0]
    for tr, lw in [("CSRT", 2.4), ("MEDIANFLOW", 2.1), ("KCF", 2.0), ("TLD", 1.8)]:
        td = _load_track_rows("stable", tr)
        ax0.plot([o - 664 for o in orig_idx[1:]], [float(td[fi]["iou"]) for fi in frames_x], color=TRACKER_COLORS_HEX[tr], lw=lw, label=tr)
    ax0_area = ax0.twinx()
    ax0_area.plot(frames_x, areas, color="#475569", ls=":", lw=1.6, label="GT талбайн тэлэлт (×)")
    ax0_area.set_ylabel("GT хүрээний талбайн өсөлт (анхнаас ×)", fontsize=8.5, color="#475569")
    ax0_area.set_ylim(0.9, 2.1)
    ax0.axhline(0.5, color="#94a3b8", ls="--", lw=1.0)
    ax0.set_xlim(1, 57)
    ax0.set_ylim(0.15, 1.02)
    ax0.set_xlabel("Тайрдсын кадрын дугаар (orig #665–#721)", fontsize=9.0, fontweight="bold")
    ax0.set_ylabel("Кадр бүрийн IoU (≥ 0.5 амжилттай)", fontsize=9.0, fontweight="bold")
    ax0.set_title("А. Тэрэг томроход (98×59 → 121×87 px) IoU хэрхэн өөрчлөгдсөн бэ?", fontsize=9.8, fontweight="bold")
    ax0.grid(alpha=0.25)
    l1, lb1 = ax0.get_legend_handles_labels()
    l2, lb2 = ax0_area.get_legend_handles_labels()
    ax0.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=8.0, framealpha=0.95)

    ax1 = axes[1]
    order = ["CSRT", "CSRT_REDETECT", "MEDIANFLOW", "MIL", "BOOSTING", "KCF", "TLD", "STATIC"]
    vals = [float(next(r["mean_iou"] for r in rows if r["tracker"] == t)) for t in order]
    lbls = [f"{t}\n(суурь)" if t == "STATIC" else t for t in order]
    cols = [TRACKER_COLORS_HEX[t] for t in order]
    y = np.arange(len(order))
    bars = ax1.barh(y, vals, color=cols, edgecolor="#334155", linewidth=0.6)
    bars[-1].set_hatch("//")
    for b, v in zip(bars, vals):
        ax1.text(v + 0.015, b.get_y() + b.get_height() * 0.5, f"{v:.4f}", va="center", fontsize=8.4, fontweight="bold", color="#0f172a")
    ax1.set_yticks(y)
    ax1.set_yticklabels(lbls, fontsize=8.4, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.02)
    ax1.set_xlabel("Mean IoU (Өндөр байх тусам сайн)", fontsize=9.0, fontweight="bold")
    ax1.set_title("Б. Тогтвортой камер (stable): 8 аргын дундаж IoU", fontsize=9.8, fontweight="bold")
    ax1.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    out = CUSTOM_FIG_DIR / "stable_scale_iou.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def build_moving_camera_clean_figure() -> Path:
    """Clean white lab-paper 2x2 flow visualization (pure DIS flow HSV + dashed white GT)."""
    frames, orig_idx, _ = load_video_clip(Path("data/phone_videos/moving_camera.mp4"), 405, 515, max_frames=300)
    gt = load_verified_gt(CSV_DIR / "phone_moving_camera_gt.csv", orig_idx)
    pair_rows = {int(r["clip_frame"]): r for r in _read_csv_rows(CSV_DIR / "phone_moving_camera_flow_pairs.csv") if r["flow_method"] == "dis_medium"}
    wheel = make_color_wheel(115, max_mag=FARNEBACK_VIS_MAX_MAG)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6), dpi=160, facecolor="white")
    cases = [
        (100, "Эргэлт давамгай мөч: orig #505 (inlier 60.0%)"),
        (76, "360° хэв гажилт + өвсний параллакс: orig #481 (inlier 34.1%)"),
    ]
    for r_i, (ci, prefix) in enumerate(cases):
        raw, _, res = compute_camera_motion_compensation(frames[ci], frames[ci + 1], flow_method="dis_medium")
        pr = pair_rows[ci]
        h_raw = draw_hsv_fixed(raw, max_mag=FARNEBACK_VIS_MAX_MAG)
        h_res = draw_hsv_fixed(res, max_mag=FARNEBACK_VIS_MAX_MAG)
        h_raw[16 : 16 + wheel.shape[0], 16 : 16 + wheel.shape[1]] = wheel
        gx, gy, gw, gh = gt[ci]["box"]

        for c_i, (img_bgr, title) in enumerate([
            (h_raw, f"{prefix}\nТүүхий DIS урсгал: дэвсгэр медиан = {float(pr['raw_bg_med_px']):.2f} px"),
            (h_res, f"Affine нөхөлтийн дараах үлдэгдэл ({float(pr['bg_reduction_pct']):+.2f}%)\nДэвсгэр = {float(pr['res_bg_med_px']):.2f} px  |  Нохой = {float(pr['res_fg_med_px']):.2f} px"),
        ]):
            ax = axes[r_i, c_i]
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax.add_patch(patches.Rectangle((gx, gy), gw, gh, fill=False, edgecolor="#ffffff", lw=2.0, ls=(0, (5, 4))))
            ax.text(gx, gy - 16, "GT нохой", color="#ffffff", fontsize=8.0, fontweight="bold", bbox=dict(facecolor="#0f172a", edgecolor="#ffffff", alpha=0.82))
            ax.set_title(title, fontsize=9.2, fontweight="bold", color="#0f172a")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Өвс урсахад нохойгоо алдах уу? (moving_camera): Түүхий DIS оптик урсгал ба 2D Affine нөхөлтийн ялгаа",
        fontsize=10.8,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = CUSTOM_FIG_DIR / "moving_camera_flow.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def build_occlusion_timeline_clean() -> Path:
    """Refined occlusion IoU timeline with factual CSRT continuous-tracking wording."""
    orig_indices = list(range(42, 236))
    gt = load_verified_gt(CSV_DIR / "phone_occlusion_gt.csv", orig_indices)
    track_data = {tr: _load_track_rows("occlusion", tr) for tr in ALL_TRACKERS}

    fig, ax = plt.subplots(figsize=(13.8, 5.6), dpi=160, facecolor="white")
    frames_x = [int(gt[i]["frame"]) for i in range(1, len(gt))]
    ax.axvspan(25, 66, color="#fef3c7", alpha=0.75, label="Тунгалаг хугарал #1 (orig #67–#108)")
    ax.axvspan(84, 130, color="#fee2e2", alpha=0.65, label="Тунгалаг хугарал #2 (orig #126–#172)")
    ax.axvline(131, color="#0f172a", ls="-.", lw=1.6, label="Шилнээс гарсан зааг: orig #173 (clip #131)")
    ax.axhline(0.5, color="#475569", ls=":", lw=1.3, label="Амжилтын босго IoU = 0.50")

    for tr in ["KCF", "CSRT", "BOOSTING", "MIL", "MEDIANFLOW", "TLD", "STATIC"]:
        ious = [float(track_data[tr][fi]["iou"]) for fi in frames_x]
        lw = 1.2 if tr == "STATIC" else (2.3 if tr in ("KCF", "CSRT") else 1.7)
        ls = "--" if tr == "STATIC" else "-"
        lbl = f"{tr} (суурь)" if tr == "STATIC" else tr
        ax.plot(frames_x, ious, color=TRACKER_COLORS_HEX[tr], lw=lw, ls=ls, alpha=0.92, label=lbl)

    ax.annotate(
        "CSRT: 193/193 кадр бүрд ≥ 0.50\n(тасралтгүй дагасан)\nKCF: дундаж IoU = 0.7190",
        xy=(155, 0.72),
        xytext=(136, 0.85),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5),
        fontsize=8.1,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#059669", alpha=0.96),
    )
    ax.annotate(
        "MIL: 31 кадр хоцорч\norig #204 (clip #162)-т 0.5 давсан",
        xy=(162, 0.51),
        xytext=(142, 0.34),
        arrowprops=dict(arrowstyle="->", color="#db2777", lw=1.5),
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#db2777", alpha=0.96),
    )
    ax.annotate(
        "TLD (5.18%) ба MEDIANFLOW (18.13% алдсан):\nшилний дараа эргэж олоогүй",
        xy=(131, 0.05),
        xytext=(58, 0.13),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#dc2626", alpha=0.96),
    )

    ax.set_xlim(1, 193)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Тайрдсын кадрын дугаар (clip #1–#193; orig #42–#235; absent_frames = 0)", fontsize=9.2, fontweight="bold")
    ax.set_ylabel("Кадр бүрийн IoU (≥ 0.5 амжилттай)", fontsize=9.2, fontweight="bold")
    ax.set_title("Алим шилний дараа яаж харагдав? (occlusion кадр бүрийн IoU муруй)", fontsize=10.5, fontweight="bold", pad=24)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8.0, framealpha=0.96)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    top_ticks = [8, 25, 66, 84, 108, 131, 162, 193]
    ax_top.set_xticks(top_ticks)
    ax_top.set_xticklabels([f"orig #{t + 42}" for t in top_ticks], fontsize=7.8)

    fig.tight_layout()
    out = CUSTOM_FIG_DIR / "occlusion_iou_timeline.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def build_good_vs_fail_figure() -> Path:
    """Clean 1-good vs 1-fail visual comparison using strictly CC0 public-domain sources."""
    # Left: CC0 Léman Express orig #260 (CSRT adapts to growing train vs KCF fixed box)
    frames_l, orig_l, _ = load_video_clip(DATA_DIR / "commons" / "leman_express_cc0.webm", 110, 260, max_frames=300)
    gt_l = load_verified_gt(CSV_DIR / "commons_leman_express_gt.csv", orig_l)
    kcf_l = _load_track_rows("commons_leman_express", "KCF")
    csrt_l = _load_track_rows("commons_leman_express", "CSRT")

    # Right: CC0 controlled squirrel full-occlusion (#51, +14 frames after reappearance):
    # TLD recovers cleanly onto the squirrel on the right, whereas CSRT/CSRT_REDETECT froze on brown bark.
    gt_s = _load_squirrel_gt_rows()
    csrt_s = _load_track_rows("squirrel_occlusion", "CSRT")
    tld_s = _load_track_rows("squirrel_occlusion", "TLD")
    im_s = cv2.imread(str(DATA_DIR / "controlled" / "squirrel_occ_seq" / gt_s[51]["image_file"]))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.1),
        dpi=160,
        facecolor="white",
        gridspec_kw={"width_ratios": [590 / 395, 720 / 420]},
    )

    # Panel A: CC0 Léman Express end frame (orig #260)
    cx, cy, cw, ch = 330, 125, 590, 395
    rgb_l = cv2.cvtColor(frames_l[150][cy : cy + ch, cx : cx + cw], cv2.COLOR_BGR2RGB)
    axes[0].imshow(rgb_l)
    gx, gy, gw, gh = gt_l[150]["box"]
    axes[0].add_patch(patches.Rectangle((gx - cx, gy - cy), gw, gh, fill=False, edgecolor="#ffffff", lw=2.0, ls=(0, (5, 4))))
    bx, by, bw, bh = float(csrt_l[150]["box_x"]), float(csrt_l[150]["box_y"]), float(csrt_l[150]["box_w"]), float(csrt_l[150]["box_h"])
    axes[0].add_patch(patches.Rectangle((bx - cx, by - cy), bw, bh, fill=False, edgecolor="#059669", lw=2.5))
    kx, ky, kw, kh = float(kcf_l[150]["box_x"]), float(kcf_l[150]["box_y"]), float(kcf_l[150]["box_w"]), float(kcf_l[150]["box_h"])
    axes[0].add_patch(patches.Rectangle((kx - cx, ky - cy), kw, kh, fill=False, edgecolor="#ea580c", lw=2.4))
    axes[0].text(12, 26, f"CSRT: IoU={float(csrt_l[150]['iou']):.2f} (Тэлсэн)", color="#ffffff", fontsize=8.4, fontweight="bold", bbox=dict(facecolor="#059669", alpha=0.92))
    axes[0].text(12, 54, f"KCF: IoU={float(kcf_l[150]['iou']):.2f} (Тогтмол хэмжээ)", color="#ffffff", fontsize=8.4, fontweight="bold", bbox=dict(facecolor="#ea580c", alpha=0.92))
    axes[0].set_title("1. Амжилттай жишээ (CC0 бодит камер, orig #260)\nCSRT хэмжээ тэлж 0.83 хүрсэн бол KCF жижиг хэвээр хоцров", fontsize=9.4, fontweight="bold", color="#0f172a")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel B: CC0 controlled squirrel 100% trunk occlusion (#51 after reappearance)
    rgb_s = cv2.cvtColor(im_s, cv2.COLOR_BGR2RGB)
    axes[1].imshow(rgb_s)
    g51 = gt_s[51]
    axes[1].add_patch(patches.Rectangle((float(g51["vis_x"]), float(g51["vis_y"])), float(g51["vis_w"]), float(g51["vis_h"]), fill=False, edgecolor="#ffffff", lw=2.0, ls=(0, (5, 4))))
    r_tld = tld_s[51]
    if int(r_tld["valid_pred"]):
        axes[1].add_patch(patches.Rectangle((float(r_tld["box_x"]), float(r_tld["box_y"])), float(r_tld["box_w"]), float(r_tld["box_h"]), fill=False, edgecolor="#0284c7", lw=2.6))
    r_csrt = csrt_s[51]
    if int(r_csrt["valid_pred"]):
        axes[1].add_patch(patches.Rectangle((float(r_csrt["box_x"]), float(r_csrt["box_y"])), float(r_csrt["box_w"]), float(r_csrt["box_h"]), fill=False, edgecolor="#dc2626", lw=2.2))
    axes[1].text(14, 30, f"TLD (цэнхэр): IoU={float(r_tld['iou']):.2f} (#49-д олоод #51-д батлав)", color="#ffffff", fontsize=8.2, fontweight="bold", bbox=dict(facecolor="#0284c7", alpha=0.94))
    axes[1].text(14, 62, f"CSRT & CSRT_REDETECT (улаан): {float(r_csrt['iou']):.2f} (Холтост гацсан) | KCF: Алдсан", color="#ffffff", fontsize=7.8, fontweight="bold", bbox=dict(facecolor="#dc2626", alpha=0.92))
    axes[1].set_title("2. Бүрэн халхлалтаас эргэж олох (CC0 фото: хэрэм #51)\nМодны их биеийн цаанаас гарч ирэхэд зөвхөн TLD сэргэсэн", fontsize=9.4, fontweight="bold", color="#0f172a")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout(pad=1.0)
    out = CUSTOM_FIG_DIR / "summary_good_vs_fail.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return out


def build_all(only_squirrel: bool = False) -> None:
    ensure_dirs()
    CUSTOM_FIG_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    sg, sk = build_squirrel_occlusion_gif_and_keyframes()
    if only_squirrel:
        print("Generated squirrel figures:", [str(p) for p in (sg, sk)])
        return
    f_gf = build_good_vs_fail_figure()
    gif_p, kf_p = build_cc0_hero_gif_and_keyframes()
    f_st = build_stable_scale_figure()
    f_mv = build_moving_camera_clean_figure()
    f_oc = build_occlusion_timeline_clean()
    print("Generated all hero GIF & clean lab figures:", [str(p) for p in (gif_p, kf_p, sg, sk, f_st, f_mv, f_oc, f_gf)])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--only-squirrel", action="store_true")
    mode.add_argument("--phone-gifs", action="store_true", help="Render the three measured phone-video clips")
    args = ap.parse_args()
    if args.phone_gifs:
        build_phone_gifs()
    else:
        build_all(only_squirrel=args.only_squirrel)
