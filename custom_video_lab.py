#!/usr/bin/env python3
"""Ready-to-run evaluation helper for user-recorded smartphone videos.

Expected clips (e.g., book or toy car moving on a desk):
  data/phone_videos/stable.mp4
  data/phone_videos/handheld.mp4
  data/phone_videos/box_occlusion.mp4

Usage:
  1) Annotate keyframes or frame-0 ROI (GUI):
       python custom_video_lab.py --annotate data/phone_videos/stable.mp4 --out-gt results/csv/phone_stable_gt.csv
  2) Evaluate all trackers + camera-compensated optical flow headlessly:
       python custom_video_lab.py --evaluate data/phone_videos/stable.mp4 --gt results/csv/phone_stable_gt.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from benchmark_flow import compute_camera_motion_compensation
from benchmark_track import SequenceSpec, evaluate_tracker
from config import ALL_TRACKERS, CSV_DIR, VIDEOS_DIR, enforce_single_thread, ensure_dirs
from opt_flow import draw_hsv_fixed


def load_video_frames(video_path: Path, max_frames: int = 300) -> List[np.ndarray]:
    """Decode frames from an MP4/MOV video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames: List[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            frames.append(fr)
    finally:
        cap.release()
    return frames


def annotate_video_gt(video_path: Path, out_gt: Path, step: int = 5) -> None:
    """Interactive keyframe ROI annotation with linear interpolation across intermediate frames."""
    frames = load_video_frames(video_path)
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")

    keyframes: Dict[int, Tuple[int, float, float, float, float]] = {}
    indices = sorted(set(list(range(0, len(frames), step)) + [len(frames) - 1]))
    print("Draw ROI and press ENTER/SPACE. Press 'c' or select empty box to mark frame as occluded (present=0).")
    for idx in indices:
        win = f"Annotate frame {idx}/{len(frames) - 1}"
        roi = cv2.selectROI(win, frames[idx], False)
        cv2.destroyWindow(win)
        if roi[2] > 0 and roi[3] > 0:
            keyframes[idx] = (1, float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
        else:
            keyframes[idx] = (0, -1.0, -1.0, 0.0, 0.0)

    out_gt.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(keyframes)
    with open(out_gt, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "present", "x", "y", "w", "h"])
        for fi in range(len(frames)):
            if fi in keyframes:
                p, bx, by, bw, bh = keyframes[fi]
            else:
                k0 = max(k for k in keys if k < fi)
                k1 = min(k for k in keys if k > fi)
                p0, x0, y0, w0, h0 = keyframes[k0]
                p1, x1, y1, w1, h1 = keyframes[k1]
                if p0 == 0 or p1 == 0:
                    p, bx, by, bw, bh = (0, -1.0, -1.0, 0.0, 0.0)
                else:
                    alpha = (fi - k0) / (k1 - k0)
                    p = 1
                    bx = x0 + alpha * (x1 - x0)
                    by = y0 + alpha * (y1 - y0)
                    bw = w0 + alpha * (w1 - w0)
                    bh = h0 + alpha * (h1 - h0)
            w.writerow([fi, p, round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)])
    print("Saved interpolated GT:", out_gt)


def evaluate_phone_video(video_path: Path, gt_csv: Path, reappear_frame: int | None = None) -> None:
    """Evaluate all trackers + camera-motion-compensated flow on a labeled custom video."""
    enforce_single_thread()
    ensure_dirs()
    frames = load_video_frames(video_path)
    gt_rows: List[Dict] = []
    with open(gt_csv, newline="") as f:
        for r in csv.DictReader(f):
            gt_rows.append({
                "frame": int(r["frame"]),
                "present": int(r["present"]),
                "box": (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])),
                "amodal_box": None,
                "dist_box": None,
            })

    n = min(len(frames), len(gt_rows))
    frames, gt_rows = frames[:n], gt_rows[:n]
    spec = SequenceSpec(f"phone_{video_path.stem}", gt_csv, video_path.parent, reappear_frame, "full_occlusion" if reappear_frame else "none", str(video_path.name))

    summaries = []
    for tr in ALL_TRACKERS:
        summ, rows = evaluate_tracker(spec, tr, [], frames, gt_rows, measure_e2e=False)
        summaries.append(summ)
        print(f"  {tr:13s} | IoU={summ['mean_iou']:.4f} | >=0.5={summ['iou_ge_05_pct']:6.2f}% | rec={summ['recovery_start_frames']}/{summ['recovery_confirmed_frames']}")

    out_csv = CSV_DIR / f"phone_{video_path.stem}_summary.csv"
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        wr.writeheader()
        wr.writerows(summaries)

    # Save camera-compensated residual flow video preview
    if len(frames) >= 2:
        h, w = frames[0].shape[:2]
        out_vid = VIDEOS_DIR / f"phone_{video_path.stem}_cam_comp.mp4"
        wr = cv2.VideoWriter(str(out_vid), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w * 2, h))
        for i in range(1, len(frames)):
            raw, _cam, res = compute_camera_motion_compensation(frames[i - 1], frames[i])
            wr.write(np.hstack([draw_hsv_fixed(raw, 12.0), draw_hsv_fixed(res, 12.0)]))
        wr.release()
        print("Saved camera-compensated flow comparison:", out_vid)


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate and evaluate user-recorded smartphone desk clips.")
    ap.add_argument("--annotate", type=Path, help="Video path to annotate GT boxes via GUI")
    ap.add_argument("--evaluate", type=Path, help="Video path to benchmark")
    ap.add_argument("--gt", type=Path, required=False, help="GT CSV path")
    ap.add_argument("--out-gt", type=Path, default=Path("results/csv/phone_custom_gt.csv"))
    ap.add_argument("--step", type=int, default=5, help="Keyframe step during annotation")
    ap.add_argument("--reappear-frame", type=int, default=None)
    args = ap.parse_args()

    if args.annotate:
        annotate_video_gt(args.annotate, args.out_gt, step=args.step)
    elif args.evaluate:
        if not args.gt:
            raise SystemExit("--gt is required with --evaluate")
        evaluate_phone_video(args.evaluate, args.gt, reappear_frame=args.reappear_frame)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
