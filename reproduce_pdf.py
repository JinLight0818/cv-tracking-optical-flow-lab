"""Reproduce the three PDF laboratory demos headlessly (and smoke-test /dev/video0).

Covers:
1) 06-Tracking-APIs.ipynb - Colaboratory.pdf (pp. 1-2):
   Boosting, MIL, KCF, TLD, MedianFlow (+ CSRT) with file/CLI ROI support.
2) 06-Optical-Flow.ipynb - Colaboratory.pdf (pp. 1-2):
   Shi-Tomasi (maxCorners=10) + pyramidal Lucas-Kanade (winSize=(200,200)).
3) 06-Optical-Flow.ipynb - Colaboratory.pdf (pp. 2-3):
   Farneback dense optical flow with completed `prvs_img = next_img`, fixed HSV
   magnitude scale, angle color wheel legend, and proper resource release.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    CSV_DIR,
    FARNEBACK_PDF,
    FARNEBACK_VIS_MAX_MAG,
    FIGURES_DIR,
    LK_PDF,
    PDF_TRACKERS,
    SHI_TOMASI_PDF,
    VIDEOS_DIR,
    ensure_dirs,
)
from lk_track import open_source
from opt_flow import draw_hsv_fixed, make_color_wheel
from tracker_binding_check import resolve_tracker_factory, safe_tracker_init


def parse_roi(roi_str: Optional[str], roi_file: Optional[str], frame: np.ndarray) -> Tuple[int, int, int, int]:
    """Resolve initial ROI from CLI string, JSON/CSV file, or interactive GUI."""
    if roi_str:
        parts = [int(round(float(x.strip()))) for x in roi_str.split(",")]
        if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
            raise ValueError(f"Invalid --roi value: {roi_str}")
        return tuple(parts)  # type: ignore[return-value]
    if roi_file:
        p = Path(roi_file)
        if p.suffix.lower() == ".json":
            d = json.loads(p.read_text())
            return (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))
        with open(p, newline="") as f:
            row = next(csv.DictReader(f))
            x_key = "vis_x" if "vis_x" in row else "x"
            y_key = "vis_y" if "vis_y" in row else "y"
            w_key = "vis_w" if "vis_w" in row else "w"
            h_key = "vis_h" if "vis_h" in row else "h"
            return (int(float(row[x_key])), int(float(row[y_key])), int(float(row[w_key])), int(float(row[h_key])))
    roi = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")
    if roi[2] <= 0 or roi[3] <= 0:
        raise ValueError("Empty ROI selected.")
    return tuple(int(v) for v in roi)  # type: ignore[return-value]


def read_frame_stream(source: str, max_frames: int = 0) -> List[np.ndarray]:
    """Load frames cleanly from video, directory, or camera index and always release."""
    is_cap, handle = open_source(source)
    frames: List[np.ndarray] = []
    try:
        if is_cap:
            if not handle.isOpened():
                raise RuntimeError(f"Could not open video/camera source: {source}")
            while max_frames <= 0 or len(frames) < max_frames:
                ret, frame = handle.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
        else:
            paths = handle[:max_frames] if max_frames > 0 else handle
            for p in paths:
                img = cv2.imread(str(p))
                if img is None:
                    break
                frames.append(img)
    finally:
        if is_cap:
            handle.release()
    return frames


def replay_tracking_pdf(
    frames: List[np.ndarray],
    roi: Tuple[int, int, int, int],
    tracker_name: str,
    out_video: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    out_png: Optional[Path] = None,
    no_show: bool = True,
) -> Dict:
    """Reproduce PDF Tracking APIs loop (pp. 1-2) with full edge-case handling."""
    if not frames:
        raise ValueError("No frames provided for tracking replay.")
    factory, binding = resolve_tracker_factory(tracker_name)
    if factory is None:
        raise RuntimeError(f"Tracker {tracker_name} unavailable ({binding}).")

    tracker = factory()
    init_ok, init_note = safe_tracker_init(tracker, frames[0], roi)
    if not init_ok:
        raise RuntimeError(f"Tracker init failed for {tracker_name}: {init_note}")

    h, w = frames[0].shape[:2]
    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))

    rows = []
    sample_vis = None
    try:
        for idx, raw in enumerate(frames):
            vis = raw.copy()
            if idx == 0:
                success, box = True, tuple(float(v) for v in roi)
            else:
                success, box = tracker.update(raw)
                success = bool(success)
            bx, by, bw, bh = (float(v) for v in box)
            if success and bw > 0 and bh > 0:
                p1 = (int(round(bx)), int(round(by)))
                p2 = (int(round(bx + bw)), int(round(by + bh)))
                cv2.rectangle(vis, p1, p2, (0, 255, 0), 3)
            else:
                cv2.putText(vis, "Failure to Detect Tracking!!", (40, max(60, h // 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis, f"{tracker_name} ({binding})", (16, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            if writer is not None:
                writer.write(vis)
            if idx == len(frames) // 2 or idx == len(frames) - 1:
                sample_vis = vis
            rows.append({
                "frame": idx, "tracker": tracker_name, "binding": binding,
                "success": int(success), "x": round(bx, 2), "y": round(by, 2),
                "w": round(bw, 2), "h": round(bh, 2),
            })
            if not no_show:
                cv2.imshow(tracker_name, vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
    finally:
        if writer is not None:
            writer.release()
        if not no_show:
            cv2.destroyAllWindows()

    if out_png is not None and sample_vis is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), sample_vis)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return {
        "tracker": tracker_name,
        "binding": binding,
        "init_note": init_note,
        "frames": len(frames),
        "success_frames": sum(r["success"] for r in rows),
    }


def replay_lk_pdf(
    frames: List[np.ndarray],
    out_video: Optional[Path] = None,
    out_png: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    no_show: bool = True,
) -> Dict:
    """Reproduce PDF Shi-Tomasi + Pyramidal LK loop (pp. 1-2) with float coords & point guards."""
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for Lucas-Kanade optical flow.")

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **SHI_TOMASI_PDF)
    if prev_pts is None:
        prev_pts = np.empty((0, 1, 2), dtype=np.float32)

    init_count = int(len(prev_pts))
    mask = np.zeros_like(frames[0])
    h, w = frames[0].shape[:2]
    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))

    rows = []
    last_img = frames[0].copy()
    try:
        for idx in range(1, len(frames)):
            frame = frames[idx].copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            active_count = 0
            mean_disp = 0.0
            if len(prev_pts) > 0:
                next_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **LK_PDF)
                if next_pts is not None and status is not None:
                    st = status.ravel() == 1
                    valid_coords = np.isfinite(next_pts).all(axis=(1, 2))
                    in_bounds = (
                        (next_pts[:, 0, 0] >= 0) & (next_pts[:, 0, 0] < w)
                        & (next_pts[:, 0, 1] >= 0) & (next_pts[:, 0, 1] < h)
                    )
                    keep = st & valid_coords & in_bounds
                    good_new = next_pts[keep].reshape(-1, 2)
                    good_prev = prev_pts[keep].reshape(-1, 2)
                    disps = []
                    for (x_new, y_new), (x_prev, y_prev) in zip(good_new, good_prev):
                        x_new, y_new = float(x_new), float(y_new)
                        x_prev, y_prev = float(x_prev), float(y_prev)
                        disps.append(float(np.hypot(x_new - x_prev, y_new - y_prev)))
                        cv2.line(mask, (int(round(x_new)), int(round(y_new))),
                                 (int(round(x_prev)), int(round(y_prev))), (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.circle(frame, (int(round(x_new)), int(round(y_new))), 6, (0, 0, 255), -1)
                    active_count = int(len(good_new))
                    mean_disp = float(np.mean(disps)) if disps else 0.0
                    # Keep float32 coordinates for next step!
                    prev_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
                else:
                    prev_pts = np.empty((0, 1, 2), dtype=np.float32)

            img = cv2.add(frame, mask)
            cv2.putText(img, f"LK PDF (pts={active_count}/{init_count})", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            last_img = img
            if writer is not None:
                writer.write(img)
            if not no_show:
                cv2.imshow("LK PDF", img)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            rows.append({"frame": idx, "active_pts": active_count, "init_pts": init_count, "mean_step_px": round(mean_disp, 4)})
            prev_gray = frame_gray.copy()
    finally:
        if writer is not None:
            writer.release()
        if not no_show:
            cv2.destroyAllWindows()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), last_img)
    if out_csv is not None and rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return {"init_pts": init_count, "final_pts": int(len(prev_pts)), "pairs": len(rows)}


def replay_farneback_pdf(
    frames: List[np.ndarray],
    out_video: Optional[Path] = None,
    out_png: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    max_mag: float = FARNEBACK_VIS_MAX_MAG,
    no_show: bool = True,
) -> Dict:
    """Reproduce PDF Farneback dense flow (pp. 2-3) with completed loop & fixed scale."""
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for Farneback optical flow.")

    prvs_img = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    h, w = frames[0].shape[:2]
    wheel = make_color_wheel(96, max_mag=max_mag)
    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))

    rows = []
    last_bgr = None
    try:
        for idx in range(1, len(frames)):
            next_img = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs_img, next_img, None, **FARNEBACK_PDF)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            bgr = draw_hsv_fixed(flow, max_mag=max_mag)
            bgr[8 : 8 + wheel.shape[0], 8 : 8 + wheel.shape[1]] = wheel
            cv2.putText(bgr, f"Farneback fixed scale max={max_mag:.1f}px", (112, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
            last_bgr = bgr
            if writer is not None:
                writer.write(bgr)
            if not no_show:
                cv2.imshow("Farneback PDF", bgr)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            rows.append({
                "frame": idx,
                "mean_mag_px": round(float(mag.mean()), 4),
                "p95_mag_px": round(float(np.percentile(mag, 95)), 4),
                "max_mag_px": round(float(mag.max()), 4),
            })
            # Missing line completed from official OpenCV opt_flow.py:
            prvs_img = next_img
    finally:
        if writer is not None:
            writer.release()
        if not no_show:
            cv2.destroyAllWindows()

    if out_png is not None and last_bgr is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), last_bgr)
    if out_csv is not None and rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return {"pairs": len(rows), "mean_mag": round(float(np.mean([r["mean_mag_px"] for r in rows])), 4)}


def smoke_webcam(device_idx: int = 0, n_frames: int = 12) -> Path:
    """Short in-memory webcam smoke run (no raw webcam media committed to git)."""
    frames = read_frame_stream(str(device_idx), max_frames=n_frames)
    if len(frames) < 2:
        raise RuntimeError(f"Webcam /dev/video{device_idx} returned fewer than 2 frames.")
    h, w = frames[0].shape[:2]
    roi = (w // 3, h // 3, w // 4, h // 4)

    records = []
    for name in PDF_TRACKERS + ["CSRT"]:
        res = replay_tracking_pdf(frames, roi, name, no_show=True)
        records.append({"task": f"tracker:{name}", "detail": res["binding"], "frames": res["frames"],
                        "metric": f"success={res['success_frames']}/{res['frames']}", "status": "OK"})

    lk_res = replay_lk_pdf(frames, no_show=True)
    records.append({"task": "optical_flow:LK_PDF", "detail": "winSize=(200,200)", "frames": len(frames),
                    "metric": f"pts={lk_res['final_pts']}/{lk_res['init_pts']}", "status": "OK"})

    fb_res = replay_farneback_pdf(frames, no_show=True)
    records.append({"task": "optical_flow:Farneback_PDF", "detail": "pyr_scale=0.5,levels=3", "frames": len(frames),
                    "metric": f"mean_mag={fb_res['mean_mag']}", "status": "OK"})

    out_csv = CSV_DIR / "webcam_smoke.csv"
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["task", "detail", "frames", "metric", "status"])
        wr.writeheader()
        wr.writerows(records)
    print("Webcam smoke run complete (no raw frames saved):", out_csv)
    return out_csv


def main() -> None:
    ensure_dirs()
    ap = argparse.ArgumentParser(description="Headless/GUI reproduction of PDF tracking & optical-flow labs.")
    ap.add_argument("--mode", choices=["tracking", "lk", "farneback", "all-pdf", "smoke-webcam"], required=True)
    ap.add_argument("--source", default="data/controlled/bicycle_occ_seq")
    ap.add_argument("--tracker", default="CSRT")
    ap.add_argument("--roi", default=None, help="Initial ROI x,y,w,h")
    ap.add_argument("--roi-file", default="results/csv/ctrl_bicycle_occ_gt.csv")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out-video", default="")
    ap.add_argument("--out-png", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if args.mode == "smoke-webcam":
        smoke_webcam()
        return

    frames = read_frame_stream(args.source, max_frames=args.max_frames)
    if not frames:
        raise SystemExit(f"No frames loaded from {args.source}")

    if args.mode == "tracking":
        roi = parse_roi(args.roi, args.roi_file, frames[0])
        res = replay_tracking_pdf(
            frames, roi, args.tracker,
            out_video=Path(args.out_video) if args.out_video else VIDEOS_DIR / f"pdf_replay_{args.tracker}.mp4",
            out_csv=Path(args.out_csv) if args.out_csv else CSV_DIR / f"pdf_replay_{args.tracker}.csv",
            out_png=Path(args.out_png) if args.out_png else None,
            no_show=args.no_show,
        )
        print(res)
    elif args.mode == "lk":
        res = replay_lk_pdf(
            frames,
            out_video=Path(args.out_video) if args.out_video else VIDEOS_DIR / "pdf_replay_lk.mp4",
            out_png=Path(args.out_png) if args.out_png else FIGURES_DIR / "pdf_replay_lk_cc0.png",
            out_csv=Path(args.out_csv) if args.out_csv else CSV_DIR / "pdf_replay_lk.csv",
            no_show=args.no_show,
        )
        print(res)
    elif args.mode == "farneback":
        res = replay_farneback_pdf(
            frames,
            out_video=Path(args.out_video) if args.out_video else VIDEOS_DIR / "pdf_replay_farneback.mp4",
            out_png=Path(args.out_png) if args.out_png else FIGURES_DIR / "pdf_replay_farneback_cc0.png",
            out_csv=Path(args.out_csv) if args.out_csv else CSV_DIR / "pdf_replay_farneback.csv",
            no_show=args.no_show,
        )
        print(res)
    elif args.mode == "all-pdf":
        roi = parse_roi(args.roi, args.roi_file, frames[0])
        for name in PDF_TRACKERS:
            res = replay_tracking_pdf(
                frames, roi, name,
                out_video=VIDEOS_DIR / f"pdf_replay_{name}.mp4",
                out_csv=CSV_DIR / f"pdf_replay_{name}.csv",
                no_show=True,
            )
            print("Tracking PDF:", res)
        print("LK PDF:", replay_lk_pdf(
            frames,
            out_video=VIDEOS_DIR / "pdf_replay_lk.mp4",
            out_png=FIGURES_DIR / "pdf_replay_lk_cc0.png",
            out_csv=CSV_DIR / "pdf_replay_lk.csv",
            no_show=True,
        ))
        print("Farneback PDF:", replay_farneback_pdf(
            frames,
            out_video=VIDEOS_DIR / "pdf_replay_farneback.mp4",
            out_png=FIGURES_DIR / "pdf_replay_farneback_cc0.png",
            out_csv=CSV_DIR / "pdf_replay_farneback.csv",
            no_show=True,
        ))


if __name__ == "__main__":
    main()
