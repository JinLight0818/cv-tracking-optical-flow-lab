#!/usr/bin/env python3
"""Headless/GUI reproduction of the OpenCV Farneback dense optical flow demo.

Adapted from OpenCV sample `samples/python/opt_flow.py` (branch 4.x,
commit 0e40c8a03158e599193d26635fcc13f110c22896, Apache-2.0 License).
Adds fixed-scale HSV color-wheel visualization, headless MP4 output, and
clean resource release.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from lk_track import open_source


def draw_flow_arrows(img_gray: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
    """Overlay sampled optical flow vectors on a grayscale background."""
    h, w = img_gray.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(np.round(lines))
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, False, (0, 255, 0), 1, cv2.LINE_AA)
    for (x1, y1), _ in lines:
        cv2.circle(vis, (int(x1), int(y1)), 1, (0, 255, 0), -1)
    return vis


def draw_hsv_fixed(flow: np.ndarray, max_mag: float = 12.0) -> np.ndarray:
    """Encode dense flow direction (hue) and magnitude (value) with a fixed scale."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(ang / 2.0, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max(max_mag, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def make_color_wheel(size: int = 96, max_mag: float = 12.0) -> np.ndarray:
    """Render an HSV direction-and-magnitude color wheel legend matching draw_hsv_fixed."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = (size - 1) / 2.0
    dx = xx - cx
    dy = yy - cx
    r = np.hypot(dx, dy)
    radius = size * 0.42
    flow = np.zeros((size, size, 2), dtype=np.float32)
    inside = r <= radius
    flow[inside, 0] = dx[inside] / radius * max_mag
    flow[inside, 1] = dy[inside] / radius * max_mag
    wheel = draw_hsv_fixed(flow, max_mag=max_mag)
    wheel[~inside] = (24, 24, 24)
    cv2.circle(wheel, (int(cx), int(cx)), int(radius), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(wheel, "0", (size - 14, int(cx) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
    cv2.putText(wheel, "90", (int(cx) - 8, size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
    cv2.putText(wheel, "180", (2, int(cx) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
    cv2.putText(wheel, "270", (int(cx) - 12, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
    return wheel


def run_opt_flow(
    source: str,
    out_video: str = "",
    no_show: bool = True,
    max_frames: int = 0,
    max_mag: float = 12.0,
) -> dict:
    """Compute Farneback flow across frames and save side-by-side arrows + HSV video."""
    is_cap, handle = open_source(source)
    prev_gray = None
    writer = None
    frame_idx = 0
    wheel = make_color_wheel(96, max_mag=max_mag)
    mags = []

    try:
        while True:
            if max_frames > 0 and frame_idx >= max_frames:
                break
            if is_cap:
                ret, frame = handle.read()
                if not ret or frame is None:
                    break
            else:
                if frame_idx >= len(handle):
                    break
                frame = cv2.imread(str(handle[frame_idx]))
                if frame is None:
                    break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.hypot(flow[..., 0], flow[..., 1])
                mags.append(float(mag.mean()))
                arrows = draw_flow_arrows(gray, flow)
                hsv_bgr = draw_hsv_fixed(flow, max_mag=max_mag)
                hsv_bgr[8 : 8 + wheel.shape[0], 8 : 8 + wheel.shape[1]] = wheel
                cv2.putText(hsv_bgr, f"max={max_mag:.1f}px", (8, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
                panel = np.hstack([arrows, hsv_bgr])
                if out_video:
                    if writer is None:
                        Path(out_video).parent.mkdir(parents=True, exist_ok=True)
                        ph, pw = panel.shape[:2]
                        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (pw, ph))
                    writer.write(panel)
                if not no_show:
                    cv2.imshow("opt_flow", panel)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break

            prev_gray = gray
            frame_idx += 1
    finally:
        if is_cap:
            handle.release()
        if writer is not None:
            writer.release()
        if not no_show:
            cv2.destroyAllWindows()

    return {"frames_processed": frame_idx, "pairs": len(mags), "mean_mag": sum(mags) / len(mags) if mags else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenCV opt_flow.py standalone adaptation.")
    ap.add_argument("--source", default="0")
    ap.add_argument("--out-video", default="")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-mag", type=float, default=12.0)
    args = ap.parse_args()
    print(run_opt_flow(args.source, args.out_video, args.no_show, args.max_frames, args.max_mag))


if __name__ == "__main__":
    main()
