#!/usr/bin/env python3
"""Ready-to-run clip slicing, annotation, tracking, and flow evaluation for user/external videos.

Expected clips (e.g., placed or symlinked into data/phone_videos/):
  data/phone_videos/stable.mp4
  data/phone_videos/moving_camera.mp4
  data/phone_videos/occlusion.mp4

Usage:
  1) Annotate keyframes cleanly (GUI):
       python custom_video_lab.py --annotate data/phone_videos/stable.mp4 --start-frame 664 --end-frame 721 --step 2 --out-gt results/csv/phone_stable_gt.csv
  2) Evaluate all 8 trackers + flow/camera compensation headlessly:
       python custom_video_lab.py --evaluate data/phone_videos/stable.mp4 --gt results/csv/phone_stable_gt.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from benchmark_flow import _detect_stratified_corners, compute_camera_motion_compensation
from benchmark_track import SequenceSpec, evaluate_tracker
from config import (
    ALL_TRACKERS,
    CSV_DIR,
    FB_THRESH_PX,
    LK_PDF,
    LK_SMALL,
    LOCAL_VIZ_DIR,
    SHI_TOMASI_PDF,
    VIDEOS_DIR,
    enforce_single_thread,
    ensure_dirs,
)
from opt_flow import draw_hsv_fixed


def load_video_clip(
    video_path: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_frames: int = 300,
) -> Tuple[List[np.ndarray], List[int], float]:
    """Decode continuous clip frames [start_frame..end_frame] and verify length <= max_frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total - 1
    if start_frame < 0 or end_frame < start_frame:
        cap.release()
        raise ValueError(f"Invalid clip bounds: start={start_frame}, end={end_frame}, total={total}")
    expected_len = end_frame - start_frame + 1
    if expected_len > max_frames:
        cap.release()
        raise ValueError(f"Requested clip {expected_len} frames exceeds max_frames={max_frames}")

    frames: List[np.ndarray] = []
    orig_indices: List[int] = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for orig_idx in range(start_frame, end_frame + 1):
            ok, fr = cap.read()
            if not ok or fr is None:
                raise RuntimeError(f"Failed decoding frame {orig_idx} from {video_path}")
            frames.append(fr)
            orig_indices.append(orig_idx)
    finally:
        cap.release()
    return frames, orig_indices, fps


def load_video_frames(video_path: Path, max_frames: int = 300) -> List[np.ndarray]:
    """Backward-compatible wrapper returning frames starting from 0 up to max_frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    end_idx = min(total - 1, max_frames - 1) if total > 0 else max_frames - 1
    frames, _, _ = load_video_clip(video_path, 0, end_idx, max_frames=max_frames)
    return frames


def annotate_video_gt(
    video_path: Path,
    out_gt: Path,
    step: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> None:
    """Interactive keyframe ROI annotation distinguishing visible, occluded_absent, and no_gt states."""
    frames, orig_indices, fps = load_video_clip(video_path, start_frame=start_frame, end_frame=end_frame, max_frames=300)
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    if step > 1:
        print("Warning: use --step 1 near occlusion/direction changes so boundaries stay exact.")

    keyframes: Dict[int, Tuple[int, str, float, float, float, float]] = {}
    indices = sorted(set(list(range(0, len(frames), step)) + [len(frames) - 1]))
    print("Draw ROI and press ENTER/SPACE. Empty box marks frame occluded_absent (present=0).")
    for idx in indices:
        win = f"Annotate clip f{idx} (orig f{orig_indices[idx]})/{len(frames) - 1}"
        roi = cv2.selectROI(win, frames[idx], False)
        cv2.destroyWindow(win)
        if roi[2] > 0 and roi[3] > 0:
            keyframes[idx] = (1, "visible", float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
        else:
            keyframes[idx] = (0, "occluded_absent", -1.0, -1.0, 0.0, 0.0)

    out_gt.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(keyframes)
    with open(out_gt, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "orig_frame", "timestamp_s", "present", "gt_state", "x", "y", "w", "h"])
        for fi in range(len(frames)):
            orig_f = orig_indices[fi]
            ts = round(orig_f / fps, 4)
            if fi in keyframes:
                p, st, bx, by, bw, bh = keyframes[fi]
            else:
                k0 = max(k for k in keys if k < fi)
                k1 = min(k for k in keys if k > fi)
                p0, st0, x0, y0, w0, h0 = keyframes[k0]
                p1, st1, x1, y1, w1, h1 = keyframes[k1]
                if p0 == 0 and p1 == 0:
                    p, st, bx, by, bw, bh = (0, "occluded_absent", -1.0, -1.0, 0.0, 0.0)
                elif p0 == 0 or p1 == 0:
                    # Do not silently mark mixed boundary intervals as absent: mark no_gt unless --step 1 used!
                    p, st, bx, by, bw, bh = (-1, "no_gt", -1.0, -1.0, 0.0, 0.0)
                else:
                    alpha = (fi - k0) / (k1 - k0)
                    p, st = 1, "visible"
                    bx = x0 + alpha * (x1 - x0)
                    by = y0 + alpha * (y1 - y0)
                    bw = w0 + alpha * (w1 - w0)
                    bh = h0 + alpha * (h1 - h0)
            w.writerow([fi, orig_f, ts, p, st, round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)])
    print("Saved verified GT:", out_gt)


def load_verified_gt(gt_csv: Path, expected_orig_indices: List[int]) -> List[Dict]:
    """Read GT CSV and strictly verify frame IDs, original frame alignment, and row count."""
    gt_rows: List[Dict] = []
    with open(gt_csv, newline="") as f:
        for r in csv.DictReader(f):
            gt_rows.append({
                "frame": int(r["frame"]),
                "orig_frame": int(r["orig_frame"]) if "orig_frame" in r else None,
                "present": int(r["present"]),
                "gt_state": r.get("gt_state", "visible" if int(r["present"]) == 1 else "occluded_absent"),
                "box": (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])),
                "amodal_box": None,
                "dist_box": None,
            })

    if len(gt_rows) != len(expected_orig_indices):
        raise ValueError(f"Frame count mismatch: video clip has {len(expected_orig_indices)} frames vs {len(gt_rows)} rows in {gt_csv}")

    for i, (row, exp_orig) in enumerate(zip(gt_rows, expected_orig_indices)):
        if row["frame"] != i:
            raise ValueError(f"Clip frame ID mismatch at index {i}: CSV has {row['frame']} in {gt_csv}")
        if row["orig_frame"] is not None and row["orig_frame"] != exp_orig:
            raise ValueError(f"Original frame mismatch at index {i}: video {exp_orig} vs CSV {row['orig_frame']} in {gt_csv}")
    return gt_rows


def _eval_phone_lk_retention(seq_id: str, frames: List[np.ndarray], init_box: Tuple[float, float, float, float]) -> List[Dict]:
    """Measure sparse LK point-ID retention, step magnitude, and initial vs redetected counts without flow GT."""
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    x, y, w, h = (int(round(v)) for v in init_box)
    roi_mask = np.zeros_like(g0)
    roi_mask[max(0, y) : min(g0.shape[0], y + h), max(0, x) : min(g0.shape[1], x + w)] = 255

    out_rows: List[Dict] = []
    for scope, mask in [("full_frame", None), ("init_roi", roi_mask)]:
        pts0 = cv2.goodFeaturesToTrack(g0, mask=mask, **SHI_TOMASI_PDF)
        init_n = 0 if pts0 is None else len(pts0)
        for vname, cfg, use_fb, redet in [
            ("LK_PDF_win200_noFB", LK_PDF, False, 0),
            ("LK_PDF_win200_FB", LK_PDF, True, 0),
            ("LK_PDF_win200_FB_redet5", LK_PDF, True, 5),
            ("LK_small_win15_FB", LK_SMALL, True, 0),
        ]:
            # tracks entries: (x, y, is_initial)
            tracks = [] if pts0 is None else [(float(p[0, 0]), float(p[0, 1]), True) for p in pts0]
            prev_g = g0.copy()
            step_mags: List[float] = []
            for fi, fr in enumerate(frames[1:], start=1):
                cur_g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                if tracks:
                    p0 = np.array([[[tx, ty]] for tx, ty, _ in tracks], dtype=np.float32)
                    p1, st_f, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, p0, None, **cfg)
                    keep = (st_f.ravel() == 1) & np.isfinite(p1).all(axis=(1, 2))
                    fh, fw = cur_g.shape
                    keep &= (p1[:, 0, 0] >= 0) & (p1[:, 0, 0] < fw) & (p1[:, 0, 1] >= 0) & (p1[:, 0, 1] < fh)
                    if use_fb:
                        p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(cur_g, prev_g, p1, None, **cfg)
                        fb_err = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
                        keep &= (st_b.ravel() == 1) & (fb_err <= FB_THRESH_PX)
                    new_tracks = []
                    for idx_p, ok_p in enumerate(keep):
                        if ok_p:
                            nx, ny = float(p1[idx_p, 0, 0]), float(p1[idx_p, 0, 1])
                            ox, oy, is_init = tracks[idx_p]
                            step_mags.append(float(np.hypot(nx - ox, ny - oy)))
                            new_tracks.append((nx, ny, is_init))
                    tracks = new_tracks
                if redet > 0 and fi % redet == 0 and mask is None:
                    dmask = np.full_like(cur_g, 255)
                    for tx, ty, _ in tracks:
                        cv2.circle(dmask, (int(round(tx)), int(round(ty))), 7, 0, -1)
                    fresh = cv2.goodFeaturesToTrack(cur_g, mask=dmask, **SHI_TOMASI_PDF)
                    if fresh is not None:
                        for fp in fresh:
                            if len(tracks) >= SHI_TOMASI_PDF["maxCorners"]:
                                break
                            tracks.append((float(fp[0, 0]), float(fp[0, 1]), False))
                prev_g = cur_g
            kept_init = sum(1 for _, _, is_i in tracks if is_i)
            redet_n = sum(1 for _, _, is_i in tracks if not is_i)
            out_rows.append({
                "sequence": seq_id,
                "scope": scope,
                "variant": vname,
                "init_pts": init_n,
                "retained_initial_pts": kept_init,
                "redetected_pts": redet_n,
                "final_active_pts": len(tracks),
                "retention_pct": round(100.0 * kept_init / init_n, 2) if init_n else 0.0,
                "median_step_px": round(statistics.median(step_mags), 4) if step_mags else 0.0,
                "flow_gt": "none",
            })
    return out_rows


def _eval_dense_and_cam_comp(
    seq_id: str,
    frames: List[np.ndarray],
    gt_rows: List[Dict],
    orig_indices: List[int],
    sample_step: int = 5,
) -> Tuple[List[Dict], Dict]:
    """Evaluate foreground vs background dense flow and global affine RANSAC compensation."""
    pair_rows: List[Dict] = []
    h, w = frames[0].shape[:2]

    for i in range(0, len(frames) - 1, sample_step):
        im0, im1 = frames[i], frames[i + 1]
        bx, by, bw, bh = (int(round(v)) for v in gt_rows[i]["box"])
        fg_mask = np.zeros((h, w), dtype=bool)
        if gt_rows[i]["present"] == 1 and bw > 0 and bh > 0:
            fg_mask[max(0, by) : min(h, by + bh), max(0, bx) : min(w, bx + bw)] = True
        bg_mask = ~fg_mask

        # Measure RANSAC inliers on stratified corners
        g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        pts0 = _detect_stratified_corners(g0)
        inlier_ratio = 0.0
        n_bg_pts = len(pts0)
        if n_bg_pts >= 8:
            lk_bg = dict(winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
            pts1, st_f, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, **lk_bg)
            pts0r, st_b, _ = cv2.calcOpticalFlowPyrLK(g1, g0, pts1, None, **lk_bg)
            fb = np.linalg.norm(pts0.reshape(-1, 2) - pts0r.reshape(-1, 2), axis=1)
            ok = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb <= 1.5)
            if ok.sum() >= 6:
                _, inl = cv2.estimateAffine2D(pts0[ok].reshape(-1, 2), pts1[ok].reshape(-1, 2), method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000, confidence=0.995)
                if inl is not None:
                    inlier_ratio = float(inl.sum()) / float(ok.sum())

        for fm in ("dis_medium", "farneback"):
            raw, cam, res = compute_camera_motion_compensation(im0, im1, flow_method=fm)
            raw_m = np.hypot(raw[..., 0], raw[..., 1])
            cam_m = np.hypot(cam[..., 0], cam[..., 1])
            res_m = np.hypot(res[..., 0], res[..., 1])

            raw_bg_med = float(np.median(raw_m[bg_mask]))
            res_bg_med = float(np.median(res_m[bg_mask]))
            est_cam_med = float(np.median(cam_m[bg_mask]))
            raw_fg_med = float(np.median(raw_m[fg_mask])) if fg_mask.any() else float("nan")
            res_fg_med = float(np.median(res_m[fg_mask])) if fg_mask.any() else float("nan")
            bg_red = 100.0 * (raw_bg_med - res_bg_med) / raw_bg_med if raw_bg_med > 1e-4 else 0.0
            pair_rows.append({
                "sequence": seq_id,
                "clip_frame": i,
                "orig_frame": orig_indices[i],
                "flow_method": fm,
                "lk_inlier_ratio": round(inlier_ratio, 3),
                "raw_bg_med_px": round(raw_bg_med, 4),
                "est_cam_med_px": round(est_cam_med, 4),
                "res_bg_med_px": round(res_bg_med, 4),
                "bg_reduction_pct": round(bg_red, 2),
                "raw_fg_med_px": round(raw_fg_med, 4) if np.isfinite(raw_fg_med) else "-",
                "res_fg_med_px": round(res_fg_med, 4) if np.isfinite(res_fg_med) else "-",
                "flow_gt": "none",
            })

    dis_rows = [r for r in pair_rows if r["flow_method"] == "dis_medium"]
    fb_rows = [r for r in pair_rows if r["flow_method"] == "farneback"]
    best_dis = max(dis_rows, key=lambda r: float(r["bg_reduction_pct"]))
    worst_dis = min(dis_rows, key=lambda r: float(r["bg_reduction_pct"]))
    summary = {
        "sequence": seq_id,
        "sampled_pairs": len(dis_rows),
        "dis_raw_bg_med": round(statistics.median(r["raw_bg_med_px"] for r in dis_rows), 4),
        "dis_res_bg_med": round(statistics.median(r["res_bg_med_px"] for r in dis_rows), 4),
        "dis_bg_red_med_pct": round(statistics.median(r["bg_reduction_pct"] for r in dis_rows), 2),
        "dis_res_fg_med": round(statistics.median(r["res_fg_med_px"] for r in dis_rows if r["res_fg_med_px"] != "-"), 4),
        "fb_raw_bg_med": round(statistics.median(r["raw_bg_med_px"] for r in fb_rows), 4),
        "fb_res_bg_med": round(statistics.median(r["res_bg_med_px"] for r in fb_rows), 4),
        "fb_bg_red_med_pct": round(statistics.median(r["bg_reduction_pct"] for r in fb_rows), 2),
        "best_dis_clip_frame": best_dis["clip_frame"],
        "best_dis_orig_frame": best_dis["orig_frame"],
        "best_dis_red_pct": best_dis["bg_reduction_pct"],
        "worst_dis_clip_frame": worst_dis["clip_frame"],
        "worst_dis_orig_frame": worst_dis["orig_frame"],
        "worst_dis_red_pct": worst_dis["bg_reduction_pct"],
    }
    return pair_rows, summary


def evaluate_phone_video(
    video_path: Path,
    gt_csv: Path,
    seq_id: Optional[str] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    reappear_frame: Optional[int] = None,
    recovery_kind: str = "none",
    description: str = "",
    run_flow: bool = True,
) -> Tuple[List[Dict], List[Dict], Optional[Dict]]:
    """Evaluate all 8 trackers (single-thread, algo & e2e timings) + LK/DIS flow on a verified clip."""
    enforce_single_thread()
    ensure_dirs()
    LOCAL_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    seq = seq_id or f"phone_{video_path.stem}"

    frames, orig_indices, fps = load_video_clip(video_path, start_frame=start_frame, end_frame=end_frame, max_frames=300)
    gt_rows = load_verified_gt(gt_csv, orig_indices)
    spec = SequenceSpec(seq, gt_csv, video_path.parent, reappear_frame, recovery_kind, description or video_path.name)

    summaries: List[Dict] = []
    print(f"\n=== Tracking Benchmark: {seq} ({len(frames)} frames, orig {orig_indices[0]}..{orig_indices[-1]} @ {fps:.3f}fps, {frames[0].shape[1]}x{frames[0].shape[0]}) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths: List[Path] = []
        for idx, fr in enumerate(frames):
            tp = Path(tmpdir) / f"f_{idx:04d}.jpg"
            cv2.imwrite(str(tp), fr)
            tmp_paths.append(tp)

        for tr in ALL_TRACKERS:
            summ, per_frame_rows = evaluate_tracker(spec, tr, tmp_paths, frames, gt_rows, measure_e2e=True)
            summ["orig_start"] = orig_indices[0]
            summ["orig_end"] = orig_indices[-1]
            summ["fps_source"] = round(fps, 3)
            summaries.append(summ)

            out_pf = CSV_DIR / f"track_{seq}_{tr}.csv"
            with open(out_pf, "w", newline="") as f:
                fields = list(per_frame_rows[0].keys()) + ["orig_frame"]
                wr = csv.DictWriter(f, fieldnames=fields)
                wr.writeheader()
                for r in per_frame_rows:
                    r_copy = dict(r)
                    r_copy["orig_frame"] = orig_indices[int(r["frame"])]
                    wr.writerow(r_copy)

            print(
                f"  {tr:13s} | IoU={summ['mean_iou']:.4f} | >=0.5={summ['iou_ge_05_pct']:6.2f}% "
                f"| fail={summ['failure_pct']:5.1f}% | absent_fp={summ['absent_fp_pct']} "
                f"| algo={summ['algo_median_ms']:6.2f}ms (p95={summ['algo_p95_ms']})"
            )

    out_csv = CSV_DIR / f"phone_{seq}_summary.csv"
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        wr.writeheader()
        wr.writerows(summaries)

    lk_rows = _eval_phone_lk_retention(seq, frames, gt_rows[0]["box"])
    with open(CSV_DIR / f"phone_{seq}_lk.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(lk_rows[0].keys()))
        wr.writeheader()
        wr.writerows(lk_rows)

    flow_summ: Optional[Dict] = None
    if run_flow:
        step = 2 if len(frames) <= 70 else 4
        flow_pair_rows, flow_summ = _eval_dense_and_cam_comp(seq, frames, gt_rows, orig_indices, sample_step=step)
        with open(CSV_DIR / f"phone_{seq}_flow_pairs.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(flow_pair_rows[0].keys()))
            wr.writeheader()
            wr.writerows(flow_pair_rows)
        print("Flow summary:", flow_summ)
    return summaries, lk_rows, flow_summ


def run_all_three_videos(run_flow: bool = True) -> None:
    """Run all three verified external video clips and write unified CSV summaries."""
    specs = [
        ("stable", Path("data/phone_videos/stable.mp4"), CSV_DIR / "phone_stable_gt.csv", 664, 721, None, "none", "Locked overhead studio camera: black miniature car moving across floor"),
        ("moving_camera", Path("data/phone_videos/moving_camera.mp4"), CSV_DIR / "phone_moving_camera_gt.csv", 405, 515, None, "none", "Moving RC-buggy Insta360 camera: running dark dog across sunlit field"),
        ("occlusion", Path("data/phone_videos/occlusion.mp4"), CSV_DIR / "phone_occlusion_gt.csv", 42, 235, 131, "heavy_translucent_occlusion", "Static camera: red apple behind translucent refraction shield (pass1 o67-108, pass2 o126-172)"),
    ]
    all_tr: List[Dict] = []
    all_lk: List[Dict] = []
    all_fl: List[Dict] = []
    for sid, vp, gp, sf, ef, rf, rk, desc in specs:
        tr_s, lk_r, fl_s = evaluate_phone_video(vp, gp, seq_id=sid, start_frame=sf, end_frame=ef, reappear_frame=rf, recovery_kind=rk, description=desc, run_flow=run_flow)
        all_tr.extend(tr_s)
        all_lk.extend(lk_r)
        if fl_s is not None:
            all_fl.append(fl_s)

    pairs: List[Tuple[str, List[Dict]]] = [
        ("phone_videos_summary.csv", all_tr),
        ("phone_videos_lk.csv", all_lk),
    ]
    if all_fl:
        pairs.append(("phone_videos_flow_summary.csv", all_fl))
    for fname, rows in pairs:
        with open(CSV_DIR / fname, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print("\nSaved unified summaries:", ", ".join(name for name, _ in pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Clip-slice, annotate, and evaluate videos cleanly.")
    ap.add_argument("--annotate", type=Path, help="Video path to annotate GT boxes via GUI")
    ap.add_argument("--evaluate", type=Path, help="Video path to benchmark")
    ap.add_argument("--run-three", action="store_true", help="Run all 3 verified clips (stable, moving_camera, occlusion)")
    ap.add_argument("--skip-flow", action="store_true", help="Skip dense flow pairs (keep existing flow CSVs)")
    ap.add_argument("--gt", type=Path, required=False, help="GT CSV path")
    ap.add_argument("--seq-id", type=str, default=None, help="Short sequence ID (e.g. stable, moving_camera, occlusion)")
    ap.add_argument("--out-gt", type=Path, default=Path("results/csv/phone_custom_gt.csv"))
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--step", type=int, default=1, help="Keyframe step during annotation (default 1)")
    ap.add_argument("--reappear-frame", type=int, default=None, help="Clip-relative reappear frame index")
    ap.add_argument("--recovery-kind", type=str, default="none")
    args = ap.parse_args()

    if args.run_three:
        run_all_three_videos(run_flow=not args.skip_flow)
    elif args.annotate:
        annotate_video_gt(args.annotate, args.out_gt, step=args.step, start_frame=args.start_frame, end_frame=args.end_frame)
    elif args.evaluate:
        if not args.gt:
            raise SystemExit("--gt is required with --evaluate")
        evaluate_phone_video(
            args.evaluate,
            args.gt,
            seq_id=args.seq_id,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            reappear_frame=args.reappear_frame,
            recovery_kind=args.recovery_kind,
            run_flow=not args.skip_flow,
        )
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
