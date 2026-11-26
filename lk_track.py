#!/usr/bin/env python3
"""Headless/GUI reproduction of the OpenCV Lukas-Kanade sparse optical flow demo.

Adapted from OpenCV sample `samples/python/lk_track.py` (branch 4.x,
commit 5e80bd3cc922bad5eb1fa02b56bc3db44abb96e9, Apache-2.0 License).
Self-contained: removes external `video`/`common` sample helpers and adds
--no-show, --out-video, --max-frames, and empty-frame/empty-point guards.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

LK_PARAMS = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
}

FEATURE_PARAMS = {
    "maxCorners": 500,
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7,
}


def open_source(source: Union[str, int]) -> Tuple[bool, Union[cv2.VideoCapture, List[Path]]]:
    """Open either an image folder/glob or a video/camera capture."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
        return True, cap
    p = Path(source)
    if p.is_dir():
        imgs = sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg")))
        return False, imgs
    cap = cv2.VideoCapture(str(p))
    return True, cap


def run_lk_track(
    source: str,
    out_video: str = "",
    no_show: bool = True,
    max_frames: int = 0,
    fb_thresh: float = 1.0,
    detect_interval: int = 5,
    track_len: int = 10,
) -> dict:
    """Run Lucas-Kanade tracking with forward-backward verification and periodic re-detection."""
    is_cap, handle = open_source(source)
    tracks: List[List[Tuple[float, float]]] = []
    prev_gray = None
    writer = None
    frame_idx = 0

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

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(tracks) > 0 and prev_gray is not None:
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                p1, st_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **LK_PARAMS)
                p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **LK_PARAMS)
                if p1 is not None and p0r is not None:
                    d = np.abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = (d < fb_thresh) & (st_f.ravel() == 1) & (st_b.ravel() == 1)
                    new_tracks = []
                    for tr, (x, y), flag in zip(tracks, p1.reshape(-1, 2), good):
                        if not flag:
                            continue
                        tr.append((float(x), float(y)))
                        if len(tr) > track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
                    tracks = new_tracks
                    if tracks:
                        cv2.polylines(vis, [np.int32(np.round(tr)) for tr in tracks], False, (0, 255, 0))
                else:
                    tracks = []

            if frame_idx % detect_interval == 0:
                mask = np.full_like(frame_gray, 255)
                for tr in tracks:
                    x, y = tr[-1]
                    cv2.circle(mask, (int(round(x)), int(round(y))), 5, 0, -1)
                pts = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **FEATURE_PARAMS)
                if pts is not None:
                    for x, y in np.float32(pts).reshape(-1, 2):
                        tracks.append([(float(x), float(y))])

            cv2.putText(vis, f"track count: {len(tracks)}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if out_video:
                if writer is None:
                    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                writer.write(vis)
            if not no_show:
                cv2.imshow("lk_track", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            prev_gray = frame_gray
            frame_idx += 1
    finally:
        if is_cap:
            handle.release()
        if writer is not None:
            writer.release()
        if not no_show:
            cv2.destroyAllWindows()

    return {"frames_processed": frame_idx, "final_tracks": len(tracks)}


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenCV lk_track.py standalone adaptation.")
    ap.add_argument("--source", default="0", help="Video file, frame directory, or camera index")
    ap.add_argument("--out-video", default="")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    stats = run_lk_track(args.source, out_video=args.out_video, no_show=args.no_show, max_frames=args.max_frames)
    print(stats)


if __name__ == "__main__":
    main()
