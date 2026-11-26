"""Download public datasets cleanly and generate controlled CC0 test sequences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import math
import struct
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config import CSV_DIR, DATA_DIR, ensure_dirs
from metrics import mask_to_bbox, write_flo

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
MIDDLEBURY_IMG_URL = "https://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip"
MIDDLEBURY_GT_URL = "https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip"
MIDDLEBURY_CODE_URL = "https://vision.middlebury.edu/flow/code/flow-code.zip"
WIKIMEDIA_BIKE_URL = "https://upload.wikimedia.org/wikipedia/commons/d/d4/Front_Side_of_Bicyle.jpg"
WIKIMEDIA_BIKE_PAGE = "https://commons.wikimedia.org/wiki/File:Front_Side_of_Bicyle.jpg"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_url(url: str, dest: Path, headers: Optional[Dict[str, str]] = None) -> Path:
    """Download URL to dest if missing or empty."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "CVLabFetcher/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def fetch_middlebury() -> None:
    """Download and extract Middlebury public-GT two-frame color data and .flo files."""
    mdir = DATA_DIR / "middlebury"
    img_zip = fetch_url(MIDDLEBURY_IMG_URL, mdir / "other-color-twoframes.zip")
    gt_zip = fetch_url(MIDDLEBURY_GT_URL, mdir / "other-gt-flow.zip")
    code_zip = fetch_url(MIDDLEBURY_CODE_URL, mdir / "flow-code.zip")
    for zpath in (img_zip, gt_zip, code_zip):
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(mdir)
    print("Middlebury extracted to", mdir)


def fetch_wikimedia() -> Path:
    """Download CC0 bicycle image from Wikimedia Commons."""
    dest = DATA_DIR / "wikimedia" / "Front_Side_of_Bicyle.jpg"
    return fetch_url(
        WIKIMEDIA_BIKE_URL,
        dest,
        headers={"User-Agent": "CVLabFetcher/1.0 (https://commons.wikimedia.org/wiki/File:Front_Side_of_Bicyle.jpg)"},
    )


def fetch_davis_subset(sequences: List[str]) -> None:
    """Fetch only selected DAVIS 2017 480p sequences via coalesced HTTP Range requests."""
    out_root = DATA_DIR / "davis"
    targets = set(sequences)
    u = urllib.parse.urlparse(DAVIS_URL)
    conn = http.client.HTTPSConnection(u.netloc, timeout=30)

    def get_range(start: int, end: int) -> bytes:
        conn.request("GET", u.path, headers={"Range": f"bytes={start}-{end}", "Connection": "keep-alive"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status} for range {start}-{end}")
        return data

    conn.request("HEAD", u.path)
    head_resp = conn.getresponse()
    head_resp.read()
    size = int(head_resp.getheader("Content-Length"))
    tail = get_range(max(0, size - 65536), size - 1)
    idx = tail.rfind(b"PK\x05\x06")
    cd_size, cd_offset = struct.unpack("<II", tail[idx + 12 : idx + 20])
    cd = get_range(cd_offset, cd_offset + cd_size - 1)

    pos = 0
    entries = []
    while pos < len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        comp_method, = struct.unpack("<H", cd[pos + 10 : pos + 12])
        comp_size, _uncomp_size = struct.unpack("<II", cd[pos + 20 : pos + 28])
        fname_len, extra_len, cmt_len = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
        lh_offset, = struct.unpack("<I", cd[pos + 42 : pos + 46])
        fname = cd[pos + 46 : pos + 46 + fname_len].decode("utf-8", "replace")
        pos += 46 + fname_len + extra_len + cmt_len
        parts = fname.split("/")
        if not fname.endswith("/") and (
            fname in ("DAVIS/README.md", "DAVIS/SOURCES.md")
            or (len(parts) >= 4 and parts[3] in targets)
        ):
            dest = out_root / fname
            if not dest.exists():
                entries.append((lh_offset, fname, comp_method, comp_size))

    if not entries:
        print("DAVIS subset already present:", sorted(targets))
        return

    entries.sort()
    chunks = []
    cur = []
    for entry in entries:
        if not cur:
            cur = [entry]
            continue
        prev_end = cur[-1][0] + 30 + len(cur[-1][1].encode()) + 256 + cur[-1][3]
        if entry[0] - cur[0][0] < 8 * 1024 * 1024 and entry[0] - prev_end < 65536:
            cur.append(entry)
        else:
            chunks.append(cur)
            cur = [entry]
    if cur:
        chunks.append(cur)

    for chunk in chunks:
        start = chunk[0][0]
        last = chunk[-1]
        end = min(size - 1, last[0] + 30 + len(last[1].encode()) + 256 + last[3])
        buf = get_range(start, end)
        for lh_offset, fname, comp_method, comp_size in chunk:
            rel = lh_offset - start
            flen, xlen = struct.unpack("<HH", buf[rel + 26 : rel + 30])
            payload = buf[rel + 30 + flen + xlen : rel + 30 + flen + xlen + comp_size]
            raw = zlib.decompress(payload, -15) if comp_method == 8 else payload
            dest = out_root / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
    print("DAVIS subset extracted:", sorted(targets))


def export_davis_gt_csv() -> None:
    """Export tight visible-part bounding boxes from DAVIS segmentation masks."""
    specs = [
        ("bmx-trees", 1, "davis_bmx_trees_bike_gt.csv"),
        ("bmx-trees", 2, "davis_bmx_trees_rider_gt.csv"),
        ("car-shadow", 1, "davis_car_shadow_gt.csv"),
    ]
    for seq, obj_id, out_name in specs:
        ann_dir = DATA_DIR / "davis" / "DAVIS" / "Annotations" / "480p" / seq
        img_dir = DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / seq
        if not ann_dir.exists():
            continue
        masks = sorted(ann_dir.glob("*.png"))
        imgs = sorted(img_dir.glob("*.jpg"))
        out_csv = CSV_DIR / out_name
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "image_file", "obj_id", "present", "x", "y", "w", "h", "mask_area"])
            for idx, (mp, ip) in enumerate(zip(masks, imgs)):
                arr = np.array(Image.open(mp))
                area = int((arr == obj_id).sum())
                box = mask_to_bbox(arr, obj_id)
                if box is None:
                    w.writerow([idx, ip.name, obj_id, 0, -1, -1, 0, 0, 0])
                else:
                    w.writerow([idx, ip.name, obj_id, 1, *box, area])
        print("Exported DAVIS GT:", out_csv)


def _clipped_paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> Tuple[int, int, int, int]:
    """Paste patch onto canvas at (x, y), returning visible clipped (vx, vy, vw, vh)."""
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(cw, x + pw)
    y1 = min(ch, y + ph)
    if x0 >= x1 or y0 >= y1:
        return (0, 0, 0, 0)
    canvas[y0:y1, x0:x1] = patch[y0 - y : y1 - y, x0 - x : x1 - x]
    return (x0, y0, x1 - x0, y1 - y0)


def generate_controlled_bicycle_experiments() -> None:
    """Create photo-generated controlled sequences and flow pairs from CC0 Front_Side_of_Bicyle.jpg."""
    bike_path = fetch_wikimedia()
    src = cv2.imread(str(bike_path))
    if src is None:
        raise RuntimeError(f"Could not read {bike_path}")

    ctrl_root = DATA_DIR / "controlled"
    w, h = 640, 480

    # Background canvas: clean textured asphalt from left & right road strips of CC0 photo (640x480)
    bg_full = cv2.resize(src, (960, 1280), interpolation=cv2.INTER_AREA)
    road_left = bg_full[420 : 420 + h, 20:340]   # 320 px wide clean road
    road_right = bg_full[420 : 420 + h, 610:930]  # 320 px wide clean road
    bg_base = np.hstack([road_left, road_right])
    # Slightly darken/blend vertical seam at x=320 (covered mostly by column anyway!)
    seam_blend = cv2.GaussianBlur(bg_base[:, 308:332], (15, 1), 0)
    bg_base[:, 308:332] = seam_blend

    # Target object patch: distinct bicycle frame + "18 SPEED" + fork/wheel center from CC0 photo
    target_src = bg_full[360:720, 355:535].copy()
    target_base = cv2.resize(target_src, (92, 164), interpolation=cv2.INTER_AREA)

    # Secondary distractor patch (handlebar/shifter texture from top of CC0 photo)
    distractor_patch = cv2.resize(bg_full[10:150, 700:830], (72, 110), interpolation=cv2.INTER_AREA)

    # Vertical occluding column patch (dark handlebar grip + textured road strip)
    col_x0, col_x1 = 258, 398  # 140 px wide column -> ensures multi-frame full occlusion of target
    col_patch = cv2.resize(bg_full[200:200 + h, 750:750 + (col_x1 - col_x0)], (col_x1 - col_x0, h))
    cv2.rectangle(col_patch, (0, 0), (col_x1 - col_x0 - 1, h - 1), (45, 45, 45), 2)

    # --- 1. Variant A: Moving bicycle target behind stationary column (translation + scale + full occlusion) ---
    seq_a_dir = ctrl_root / "bicycle_occ_seq"
    seq_a_dir.mkdir(parents=True, exist_ok=True)
    n_frames = 40
    gt_a_csv = CSV_DIR / "ctrl_bicycle_occ_gt.csv"
    with open(gt_a_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "frame", "image_file", "present",
            "vis_x", "vis_y", "vis_w", "vis_h",
            "amodal_x", "amodal_y", "amodal_w", "amodal_h",
            "visible_fraction", "dx_gt", "dy_gt", "scale_factor",
        ])

        prev_x, prev_y = None, None
        for i in range(n_frames):
            canvas = bg_base.copy()
            # Target moves smoothly left-to-right across the column, while scaling from 1.0 -> 1.18 -> 0.86
            scale = 1.0 + 0.18 * math.sin(2.0 * math.pi * i / (n_frames - 1))
            tw = int(round(target_base.shape[1] * scale))
            th = int(round(target_base.shape[0] * scale))
            patch = cv2.resize(target_base, (tw, th), interpolation=cv2.INTER_LINEAR)

            tx = int(round(52 + i * 11.8))
            ty = int(round(160 + 24 * math.sin(i * 0.2)))
            _clipped_paste(canvas, patch, tx, ty)

            # Distractor moves opposite direction near bottom-right / crosses near frame 24-34
            dx_dist = int(round(520 - i * 11.0))
            dy_dist = 290
            _clipped_paste(canvas, distractor_patch, dx_dist, dy_dist)

            # Stationary foreground column occludes target as it passes x=[278, 376]
            canvas[:, col_x0:col_x1] = col_patch

            # Compute exact analytical visible-part mask and GT bbox
            vis_mask = np.zeros((h, w), dtype=np.uint8)
            vx0, vy0 = max(0, tx), max(0, ty)
            vx1, vy1 = min(w, tx + tw), min(h, ty + th)
            if vx0 < vx1 and vy0 < vy1:
                vis_mask[vy0:vy1, vx0:vx1] = 1
            vis_mask[:, col_x0:col_x1] = 0
            vis_area = int(vis_mask.sum())
            amodal_area = tw * th
            vis_frac = vis_area / amodal_area if amodal_area > 0 else 0.0

            ys, xs = np.where(vis_mask > 0)
            if xs.size == 0:
                present = 0
                vis_box = (-1, -1, 0, 0)
            else:
                present = 1
                vis_box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))

            dx_gt = 0.0 if prev_x is None else float(tx - prev_x)
            dy_gt = 0.0 if prev_y is None else float(ty - prev_y)
            prev_x, prev_y = tx, ty

            fname = f"frame_{i:04d}.png"
            cv2.imwrite(str(seq_a_dir / fname), canvas)
            wcsv.writerow([
                i, fname, present,
                *vis_box, tx, ty, tw, th,
                round(vis_frac, 4), dx_gt, dy_gt, round(scale, 4),
            ])

    # --- 2. Variant B: Whole-frame camera shift on CC0 photo (for flow + storyboard comparison) ---
    seq_b_dir = ctrl_root / "bicycle_cam_shift_seq"
    seq_b_dir.mkdir(parents=True, exist_ok=True)
    # Use a wider crop of bg_full so camera panning reveals real pixels with ZERO border wrapping artifacts
    pan_src = cv2.resize(src, (1000, 1333), interpolation=cv2.INTER_AREA)
    # Initial ROI on frame 0 around "18 SPEED" tube area
    roi0 = (268, 220, 108, 170)
    dx_cam, dy_cam = 6, 1  # camera pans right-down by (+6, +1) -> scene shifts (-6, -1) in image coordinates
    gt_b_csv = CSV_DIR / "ctrl_bicycle_cam_shift_gt.csv"
    with open(gt_b_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "image_file", "present", "x", "y", "w", "h", "u_gt", "v_gt"])
        for i in range(24):
            ox = 120 + i * dx_cam
            oy = 420 + i * dy_cam
            frame = pan_src[oy : oy + h, ox : ox + w].copy()
            # Overlay the same static column painted in world coordinates (shifts with scene!)
            col_world_x = 430 - i * dx_cam
            if 0 < col_world_x < w:
                c0 = max(0, col_world_x)
                c1 = min(w, col_world_x + 70)
                if c0 < c1:
                    frame[:, c0:c1] = col_patch[:, : c1 - c0]
            bx = roi0[0] - i * dx_cam
            by = roi0[1] - i * dy_cam
            fname = f"frame_{i:04d}.png"
            cv2.imwrite(str(seq_b_dir / fname), frame)
            wcsv.writerow([i, fname, 1, bx, by, roi0[2], roi0[3], -float(dx_cam), -float(dy_cam)])

    # --- 3. Controlled Optical Flow Two-Frame Pairs with exact occlusion/disocclusion masks ---
    pair_dir = ctrl_root / "flow_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Pair 1: Camera shift pair (frame 0 -> frame 1 of pan_src with known u=-6, v=-1)
    cam_f0 = pan_src[420 : 420 + h, 120 : 120 + w].copy()
    cam_f1 = pan_src[420 + dy_cam : 420 + dy_cam + h, 120 + dx_cam : 120 + dx_cam + w].copy()
    cv2.imwrite(str(pair_dir / "cam_shift_f0.png"), cam_f0)
    cv2.imwrite(str(pair_dir / "cam_shift_f1.png"), cam_f1)
    flow_cam = np.zeros((h, w, 2), dtype=np.float32)
    flow_cam[..., 0] = -float(dx_cam)
    flow_cam[..., 1] = -float(dy_cam)
    # Pixels leaving frame 0 across left/top border have no target in frame 1 -> mask invalid
    valid_cam = np.ones((h, w), dtype=bool)
    valid_cam[:, :dx_cam] = False
    valid_cam[:dy_cam, :] = False
    write_flo(pair_dir / "cam_shift_gt.flo", flow_cam, valid_cam)

    # Pair 2: Foreground object motion + column occlusion (frame 15 -> frame 16 of bicycle_occ_seq style)
    # Build clean pair where ONLY the target moves (+12 px right, +2 px down) behind stationary column
    obj_f0 = bg_base.copy()
    obj_f1 = bg_base.copy()
    tx0, ty0 = 190, 155
    dx_o, dy_o = 12, 2
    tx1, ty1 = tx0 + dx_o, ty0 + dy_o
    tw_o, th_o = target_base.shape[1], target_base.shape[0]
    _clipped_paste(obj_f0, target_base, tx0, ty0)
    _clipped_paste(obj_f1, target_base, tx1, ty1)
    obj_f0[:, col_x0:col_x1] = col_patch
    obj_f1[:, col_x0:col_x1] = col_patch
    cv2.imwrite(str(pair_dir / "obj_occ_f0.png"), obj_f0)
    cv2.imwrite(str(pair_dir / "obj_occ_f1.png"), obj_f1)

    flow_obj = np.zeros((h, w, 2), dtype=np.float32)
    valid_obj = np.ones((h, w), dtype=bool)
    mask_t0 = np.zeros((h, w), dtype=bool)
    mask_t1 = np.zeros((h, w), dtype=bool)
    mask_col = np.zeros((h, w), dtype=bool)
    mask_col[:, col_x0:col_x1] = True
    mask_t0[ty0 : ty0 + th_o, tx0 : tx0 + tw_o] = True
    mask_t1[ty1 : ty1 + th_o, tx1 : tx1 + tw_o] = True

    vis_t0 = mask_t0 & (~mask_col)
    vis_t1 = mask_t1 & (~mask_col)
    flow_obj[vis_t0, 0] = float(dx_o)
    flow_obj[vis_t0, 1] = float(dy_o)
    # Pixels on target at t0 that move behind column at t1 are occluded; pixels of background at t0
    # that become covered by target at t1 are also occluded -> exclude both from valid flow GT!
    occluded_t0_target = vis_t0 & (
        np.roll(mask_col, -dx_o, axis=1)
    )
    covered_bg_at_t1 = (~mask_t0) & (~mask_col) & vis_t1
    valid_obj[occluded_t0_target | covered_bg_at_t1] = False
    write_flo(pair_dir / "obj_occ_gt.flo", flow_obj, valid_obj)

    print("Generated controlled CC0 experiments in", ctrl_root)


def write_dataset_manifest() -> Path:
    """Record source URL, ID, frame range, license, and SHA-256 for every dataset asset."""
    rows = []

    def add(source: str, item_id: str, frames_used: str, license_terms: str, path: Optional[Path]) -> None:
        digest = sha256_file(path) if path is not None and path.exists() else "-"
        rel = str(path.relative_to(DATA_DIR.parent)) if path is not None and path.exists() else "-"
        rows.append([source, item_id, frames_used, license_terms, digest, rel])

    bike = DATA_DIR / "wikimedia" / "Front_Side_of_Bicyle.jpg"
    add(WIKIMEDIA_BIKE_PAGE, "Wikimedia:Front_Side_of_Bicyle.jpg", "1 image (1200x1600) -> 40/24 controlled frames", "CC0 1.0 Public Domain Dedication", bike)

    for seq in ("RubberWhale", "Grove2"):
        f10 = DATA_DIR / "middlebury" / "other-data" / seq / "frame10.png"
        f11 = DATA_DIR / "middlebury" / "other-data" / seq / "frame11.png"
        flo = DATA_DIR / "middlebury" / "other-gt-flow" / seq / "flow10.flo"
        add("https://vision.middlebury.edu/flow/data/", f"Middlebury:{seq}:frame10", "frame10-frame11", "Middlebury Optical Flow Evaluation public GT (research/eval; raw excluded from GitHub)", f10)
        add("https://vision.middlebury.edu/flow/data/", f"Middlebury:{seq}:frame11", "frame10-frame11", "Middlebury Optical Flow Evaluation public GT (research/eval; raw excluded from GitHub)", f11)
        add("https://vision.middlebury.edu/flow/data/", f"Middlebury:{seq}:flow10.flo", "frame10->frame11 GT", "Middlebury Optical Flow Evaluation public GT (research/eval; raw excluded from GitHub)", flo)

    for seq, fcount in (("bmx-trees", "00000-00079 (80 frames)"), ("car-shadow", "00000-00039 (40 frames)")):
        f0 = DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / seq / "00000.jpg"
        m0 = DATA_DIR / "davis" / "DAVIS" / "Annotations" / "480p" / seq / "00000.png"
        add("https://davischallenge.org/davis2017/code.html", f"DAVIS-2017:{seq}:00000.jpg", fcount, "CC BY-NC 4.0 (raw excluded from GitHub)", f0)
        add("https://davischallenge.org/davis2017/code.html", f"DAVIS-2017:{seq}:00000.png", fcount, "CC BY-NC 4.0 (raw excluded from GitHub)", m0)

    for name, rel in (
        ("Controlled:obj_occ_gt.flo", DATA_DIR / "controlled" / "flow_pairs" / "obj_occ_gt.flo"),
        ("Controlled:cam_shift_gt.flo", DATA_DIR / "controlled" / "flow_pairs" / "cam_shift_gt.flo"),
    ):
        add("Photo-generated from CC0 Front_Side_of_Bicyle.jpg", name, "frame0->frame1", "CC0 1.0 derived controlled experiment", rel)

    out_csv = CSV_DIR / "dataset_manifest.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_url", "dataset_id", "frames_used", "license_or_terms", "sha256", "local_path"])
        w.writerows(rows)
    print("Dataset manifest written:", out_csv)
    return out_csv


def main() -> None:
    ensure_dirs()
    ap = argparse.ArgumentParser(description="Prepare datasets and controlled CC0 experiments.")
    ap.add_argument("--all", action="store_true", help="Fetch all public assets, generate controlled tests, write manifest")
    ap.add_argument("--fetch-middlebury", action="store_true")
    ap.add_argument("--fetch-wikimedia", action="store_true")
    ap.add_argument("--fetch-davis", nargs="*", default=None, help="DAVIS sequence names (default: bmx-trees car-shadow)")
    ap.add_argument("--generate-controlled", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    args = ap.parse_args()

    if args.all or args.fetch_middlebury:
        fetch_middlebury()
    if args.all or args.fetch_wikimedia:
        fetch_wikimedia()
    if args.all or args.fetch_davis is not None:
        seqs = args.fetch_davis if args.fetch_davis else ["bmx-trees", "car-shadow"]
        fetch_davis_subset(seqs)
        export_davis_gt_csv()
    if args.all or args.generate_controlled:
        generate_controlled_bicycle_experiments()
    if args.all or args.manifest:
        write_dataset_manifest()


if __name__ == "__main__":
    main()
