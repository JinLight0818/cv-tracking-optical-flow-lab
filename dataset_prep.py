"""Download public datasets cleanly (HTTP Range ZIP64-aware) and build controlled CC0 suites."""

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
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config import CSV_DIR, DATA_DIR, ensure_dirs
from metrics import mask_to_bbox, write_flo

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
MIDDLEBURY_IMG_URL = "https://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip"
MIDDLEBURY_GT_URL = "https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip"
MIDDLEBURY_CODE_URL = "https://vision.middlebury.edu/flow/code/flow-code.zip"
SINTEL_IMG_URL = "https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip"
SINTEL_EXTRAS_URL = "https://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip"
LASOT_HF_BICYCLE_URL = "https://huggingface.co/datasets/l-lt/LaSOT/resolve/main/bicycle.zip"
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


class HttpRangeZipClient:
    """Read and extract specific entries from remote ZIP/ZIP64 archives via HTTP Range."""

    def __init__(self, url: str) -> None:
        self.url = self._resolve_redirects(url)
        self.u = urllib.parse.urlparse(self.url)
        self.conn = self._connect()
        self.size = self._head_size()

    @staticmethod
    def _resolve_redirects(url: str) -> str:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "CVLabFetcher/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.geturl()

    def _connect(self) -> http.client.HTTPConnection:
        if self.u.scheme == "https":
            return http.client.HTTPSConnection(self.u.netloc, timeout=60)
        return http.client.HTTPConnection(self.u.netloc, timeout=60)

    def _head_size(self) -> int:
        full_path = self.u.path + (f"?{self.u.query}" if self.u.query else "")
        for attempt in range(3):
            try:
                self.conn.request("HEAD", full_path)
                resp = self.conn.getresponse()
                resp.read()
                return int(resp.getheader("Content-Length"))
            except Exception:
                if attempt == 2:
                    raise
                self.conn.close()
                self.conn = self._connect()
        raise RuntimeError("HEAD failed")

    def get_range(self, start: int, end: int) -> bytes:
        full_path = self.u.path + (f"?{self.u.query}" if self.u.query else "")
        for attempt in range(2):
            try:
                self.conn.request(
                    "GET",
                    full_path,
                    headers={"Range": f"bytes={start}-{end}", "Connection": "keep-alive"},
                )
                resp = self.conn.getresponse()
                data = resp.read()
                if resp.status not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status} for range {start}-{end}")
                return data
            except Exception:
                if attempt == 1:
                    raise
                self.conn.close()
                self.conn = self._connect()
        raise RuntimeError("Unreachable")

    def read_central_directory(self, max_bytes: Optional[int] = None) -> List[Tuple[int, str, int, int]]:
        tail = self.get_range(max(0, self.size - 65536), self.size - 1)
        idx64 = tail.rfind(b"PK\x06\x06")
        if idx64 != -1:
            cd_size, cd_offset = struct.unpack("<QQ", tail[idx64 + 40 : idx64 + 56])
        else:
            idx = tail.rfind(b"PK\x05\x06")
            cd_size, cd_offset = struct.unpack("<II", tail[idx + 12 : idx + 20])
        if max_bytes is not None:
            cd_size = min(cd_size, max_bytes)
        cd = self.get_range(cd_offset, cd_offset + cd_size - 1)

        pos = 0
        entries: List[Tuple[int, str, int, int]] = []
        while pos + 46 <= len(cd):
            if cd[pos : pos + 4] != b"PK\x01\x02":
                break
            comp_method, = struct.unpack("<H", cd[pos + 10 : pos + 12])
            comp_size, _ = struct.unpack("<II", cd[pos + 20 : pos + 28])
            fn_len, ex_len, cm_len = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
            lh_off, = struct.unpack("<I", cd[pos + 42 : pos + 46])
            extra = cd[pos + 46 + fn_len : pos + 46 + fn_len + ex_len]
            if lh_off == 0xFFFFFFFF:
                ep = 0
                while ep + 4 <= len(extra):
                    eid, esz = struct.unpack("<HH", extra[ep : ep + 4])
                    if eid == 0x0001:
                        vals = struct.unpack("<" + "Q" * (esz // 8), extra[ep + 4 : ep + 4 + esz])
                        lh_off = vals[-1]
                        break
                    ep += 4 + esz
            fn = cd[pos + 46 : pos + 46 + fn_len].decode("utf-8", "replace")
            if not fn.endswith("/"):
                entries.append((lh_off, fn, comp_method, comp_size))
            pos += 46 + fn_len + ex_len + cm_len
        return entries

    def extract_filtered(
        self,
        out_root: Path,
        predicate: Callable[[str], bool],
        max_cd_bytes: Optional[int] = None,
    ) -> int:
        entries = [e for e in self.read_central_directory(max_cd_bytes) if predicate(e[1])]
        todo = [e for e in entries if not (out_root / e[1]).exists()]
        if not todo:
            return 0
        todo.sort()
        chunks: List[List[Tuple[int, str, int, int]]] = []
        cur: List[Tuple[int, str, int, int]] = []
        for entry in todo:
            if not cur:
                cur = [entry]
                continue
            prev_end = cur[-1][0] + 30 + len(cur[-1][1].encode()) + 512 + cur[-1][3]
            if entry[0] - cur[0][0] < 8 * 1024 * 1024 and entry[0] - prev_end < 65536:
                cur.append(entry)
            else:
                chunks.append(cur)
                cur = [entry]
        if cur:
            chunks.append(cur)

        extracted = 0
        for ch in chunks:
            start = ch[0][0]
            last = ch[-1]
            end = min(self.size - 1, last[0] + 30 + len(last[1].encode()) + 512 + last[3])
            buf = self.get_range(start, end)
            for lh_off, fn, comp_method, comp_size in ch:
                rel = lh_off - start
                flen, xlen = struct.unpack("<HH", buf[rel + 26 : rel + 30])
                payload = buf[rel + 30 + flen + xlen : rel + 30 + flen + xlen + comp_size]
                raw = zlib.decompress(payload, -15) if comp_method == 8 else payload
                dest = out_root / fn
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
                extracted += 1
        return extracted


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


def fetch_sintel_pairs() -> None:
    """Fetch held-out MPI Sintel image pairs, .flo GT, and occlusion/invalid masks."""
    sroot = DATA_DIR / "sintel"
    want_imgs = {
        "training/clean/alley_1/frame_0001.png",
        "training/clean/alley_1/frame_0002.png",
        "training/final/ambush_5/frame_0010.png",
        "training/final/ambush_5/frame_0011.png",
    }
    want_extras = {
        "README.txt",
        "training/flow/alley_1/frame_0001.flo",
        "training/occlusions/alley_1/frame_0001.png",
        "training/invalid/alley_1/frame_0001.png",
        "training/flow/ambush_5/frame_0010.flo",
        "training/occlusions/ambush_5/frame_0010.png",
        "training/invalid/ambush_5/frame_0010.png",
    }
    c_img = HttpRangeZipClient(SINTEL_IMG_URL)
    n1 = c_img.extract_filtered(sroot, lambda fn: fn in want_imgs)
    c_ext = HttpRangeZipClient(SINTEL_EXTRAS_URL)
    n2 = c_ext.extract_filtered(sroot, lambda fn: fn in want_extras)
    print(f"MPI Sintel subsets ready (downloaded {n1 + n2} files in {sroot})")


def fetch_lasot_subset(seq_name: str = "bicycle-11", max_frames: int = 120) -> None:
    """Fetch a held-out LaSOT sequence slice (default: bicycle-11, frames 1..120) via HTTP Range."""
    lroot = DATA_DIR / "lasot"
    prefix = f"{seq_name}/"
    wanted_frames = {f"{prefix}img/{i:08d}.jpg" for i in range(1, max_frames + 1)}
    wanted_meta = {
        f"{prefix}groundtruth.txt",
        f"{prefix}full_occlusion.txt",
        f"{prefix}out_of_view.txt",
        f"{prefix}nlp.txt",
    }
    want_all = wanted_frames | wanted_meta
    client = HttpRangeZipClient(LASOT_HF_BICYCLE_URL)
    n = client.extract_filtered(lroot, lambda fn: fn in want_all)
    print(f"LaSOT {seq_name} ({max_frames} frames) ready (downloaded {n} files)")
    export_lasot_gt_csv(seq_name, max_frames)


def export_lasot_gt_csv(seq_name: str = "bicycle-11", max_frames: int = 120) -> Path:
    """Export LaSOT groundtruth.txt matched explicitly to numbered img/%08d.jpg files."""
    seq_dir = DATA_DIR / "lasot" / seq_name
    gt_lines = (seq_dir / "groundtruth.txt").read_text().strip().splitlines()
    occ_raw = (seq_dir / "full_occlusion.txt").read_text().strip().replace("\n", ",")
    oov_raw = (seq_dir / "out_of_view.txt").read_text().strip().replace("\n", ",")
    occ_vals = [int(x.strip()) for x in occ_raw.split(",") if x.strip()]
    oov_vals = [int(x.strip()) for x in oov_raw.split(",") if x.strip()]

    out_csv = CSV_DIR / f"lasot_{seq_name.replace('-', '')}_gt.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "image_file", "present", "x", "y", "w", "h", "full_occlusion", "out_of_view"])
        for idx in range(min(max_frames, len(gt_lines))):
            img_name = f"{idx + 1:08d}.jpg"
            img_path = seq_dir / "img" / img_name
            if not img_path.exists():
                raise FileNotFoundError(f"Missing LaSOT frame {img_path}")
            bx, by, bw, bh = (float(v) for v in gt_lines[idx].split(","))
            fo = occ_vals[idx] if idx < len(occ_vals) else 0
            ov = oov_vals[idx] if idx < len(oov_vals) else 0
            present = 0 if (fo == 1 or ov == 1 or bw <= 0 or bh <= 0) else 1
            w.writerow([idx, img_name, present, int(round(bx)), int(round(by)), int(round(bw)), int(round(bh)), fo, ov])
    print("Exported LaSOT GT:", out_csv)
    return out_csv


def fetch_wikimedia() -> Path:
    """Download CC0 bicycle image from Wikimedia Commons."""
    dest = DATA_DIR / "wikimedia" / "Front_Side_of_Bicyle.jpg"
    return fetch_url(
        WIKIMEDIA_BIKE_URL,
        dest,
        headers={"User-Agent": "CVLabFetcher/1.0 (https://commons.wikimedia.org/wiki/File:Front_Side_of_Bicyle.jpg)"},
    )


def fetch_davis_subset(sequences: List[str]) -> None:
    """Fetch selected DAVIS 2017 480p sequences via coalesced HTTP Range requests."""
    out_root = DATA_DIR / "davis"
    targets = set(sequences)
    client = HttpRangeZipClient(DAVIS_URL)
    n = client.extract_filtered(
        out_root,
        lambda fn: fn in ("DAVIS/README.md", "DAVIS/SOURCES.md")
        or (len(fn.split("/")) >= 4 and fn.split("/")[3] in targets),
    )
    print(f"DAVIS subset ready {sorted(targets)} (downloaded {n} files)")


def match_frames_and_masks(img_dir: Path, mask_dir: Path) -> Tuple[List[Tuple[str, Path, Path]], Dict]:
    """Match JPEG frames and PNG masks explicitly by stem (00042.jpg <-> 00042.png) and check integrity."""
    imgs = {p.stem: p for p in img_dir.glob("*.jpg")}
    masks = {p.stem: p for p in mask_dir.glob("*.png")}
    missing_masks = sorted(set(imgs) - set(masks))
    missing_imgs = sorted(set(masks) - set(imgs))
    common = sorted(set(imgs) & set(masks))

    matched: List[Tuple[str, Path, Path]] = []
    size_mismatches: List[str] = []
    for stem in common:
        ip, mp = imgs[stem], masks[stem]
        im = cv2.imread(str(ip))
        mk = np.array(Image.open(mp))
        if im is None or im.shape[:2] != mk.shape[:2]:
            size_mismatches.append(f"{stem}: img={None if im is None else im.shape[:2]} mask={mk.shape[:2]}")
            continue
        matched.append((stem, ip, mp))

    report = {
        "matched": len(matched),
        "missing_masks": missing_masks,
        "missing_imgs": missing_imgs,
        "size_mismatches": size_mismatches,
    }
    return matched, report


def export_davis_gt_csv() -> None:
    """Export tight visible-part bounding boxes from DAVIS masks matched by stem."""
    specs = [
        ("bmx-trees", 1, "davis_bmx_trees_bike_gt.csv"),
        ("bmx-trees", 2, "davis_bmx_trees_rider_gt.csv"),
        ("car-shadow", 1, "davis_car_shadow_gt.csv"),
        ("scooter-black", 1, "davis_scooter_black_gt.csv"),
    ]
    for seq, obj_id, out_name in specs:
        ann_dir = DATA_DIR / "davis" / "DAVIS" / "Annotations" / "480p" / seq
        img_dir = DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / seq
        if not ann_dir.exists() or not img_dir.exists():
            continue
        matched, report = match_frames_and_masks(img_dir, ann_dir)
        if report["missing_masks"] or report["missing_imgs"] or report["size_mismatches"]:
            raise RuntimeError(f"DAVIS integrity issue in {seq}: {report}")
        out_csv = CSV_DIR / out_name
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "stem", "image_file", "mask_file", "obj_id", "present", "x", "y", "w", "h", "mask_area"])
            for idx, (stem, ip, mp) in enumerate(matched):
                arr = np.array(Image.open(mp))
                area = int((arr == obj_id).sum())
                box = mask_to_bbox(arr, obj_id)
                if box is None:
                    w.writerow([idx, stem, ip.name, mp.name, obj_id, 0, -1, -1, 0, 0, 0])
                else:
                    w.writerow([idx, stem, ip.name, mp.name, obj_id, 1, *box, area])
        print(f"Exported DAVIS GT ({report['matched']} matched pairs, 0 mismatches):", out_csv)


def _clipped_paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> Tuple[int, int, int, int]:
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + pw), min(ch, y + ph)
    if x0 >= x1 or y0 >= y1:
        return (0, 0, 0, 0)
    canvas[y0:y1, x0:x1] = patch[y0 - y : y1 - y, x0 - x : x1 - x]
    return (x0, y0, x1 - x0, y1 - y0)


def _build_bicycle_assets(src: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract reusable background canvas, bicycle target, handlebar distractor, bike distractor, column texture."""
    w, h = 640, 480
    bg_full = cv2.resize(src, (960, 1280), interpolation=cv2.INTER_AREA)
    road_left = bg_full[420 : 420 + h, 20:340]
    road_right = bg_full[420 : 420 + h, 610:930]
    bg_base = np.hstack([road_left, road_right])
    bg_base[:, 308:332] = cv2.GaussianBlur(bg_base[:, 308:332], (15, 1), 0)

    target_base = cv2.resize(bg_full[360:720, 355:535], (92, 164), interpolation=cv2.INTER_AREA)
    shifter_distractor = cv2.resize(bg_full[10:150, 700:830], (72, 110), interpolation=cv2.INTER_AREA)
    # Nearly identical bicycle-tube distractor (cropped slightly & flipped vertically for realistic confusion)
    bike_distractor = cv2.resize(bg_full[380:700, 365:525], (88, 156), interpolation=cv2.INTER_AREA)
    col_strip = bg_full[200 : 200 + h, 750:890]
    pan_src = cv2.resize(src, (1000, 1333), interpolation=cv2.INTER_AREA)
    return bg_base, target_base, shifter_distractor, bike_distractor, col_strip, pan_src


def _generate_occ_sequence(
    seq_dir: Path,
    gt_csv: Path,
    bg_base: np.ndarray,
    target_base: np.ndarray,
    distractor_patch: np.ndarray,
    col_strip: np.ndarray,
    col_x0: int,
    col_x1: int,
    n_frames: int,
    x0: float,
    vx: float,
    scale_amp: float,
    y_amp: float,
    distractor_x0: float,
    distractor_vx: float,
    distractor_y: int,
) -> Tuple[int, int]:
    """Generate a controlled occlusion sequence with exact visible & amodal GT boxes.

    Returns (full_occlusion_count, reappear_frame).
    """
    seq_dir.mkdir(parents=True, exist_ok=True)
    h, w = bg_base.shape[:2]
    col_w = col_x1 - col_x0
    col_patch = cv2.resize(col_strip, (col_w, h), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(col_patch, (0, 0), (col_w - 1, h - 1), (45, 45, 45), 2)

    occ_frames: List[int] = []
    reappear_frame = -1
    prev_x, prev_y = None, None

    with open(gt_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "frame", "image_file", "present",
            "vis_x", "vis_y", "vis_w", "vis_h",
            "amodal_x", "amodal_y", "amodal_w", "amodal_h",
            "visible_fraction", "dx_gt", "dy_gt", "scale_factor",
            "dist_x", "dist_y", "dist_w", "dist_h",
        ])

        for i in range(n_frames):
            canvas = bg_base.copy()
            scale = 1.0 + scale_amp * math.sin(2.0 * math.pi * i / max(1, n_frames - 1))
            tw = int(round(target_base.shape[1] * scale))
            th = int(round(target_base.shape[0] * scale))
            patch = cv2.resize(target_base, (tw, th), interpolation=cv2.INTER_LINEAR)

            tx = int(round(x0 + i * vx))
            ty = int(round(160 + y_amp * math.sin(i * 0.2)))
            _clipped_paste(canvas, patch, tx, ty)

            dx_dist = int(round(distractor_x0 + i * distractor_vx))
            dy_dist = distractor_y
            _clipped_paste(canvas, distractor_patch, dx_dist, dy_dist)

            canvas[:, col_x0:col_x1] = col_patch

            vis_mask = np.zeros((h, w), dtype=np.uint8)
            vx0, vy0 = max(0, tx), max(0, ty)
            vx1, vy1 = min(w, tx + tw), min(h, ty + th)
            if vx0 < vx1 and vy0 < vy1:
                vis_mask[vy0:vy1, vx0:vx1] = 1
            vis_mask[:, col_x0:col_x1] = 0
            vis_area = int(vis_mask.sum())
            amodal_area = max(1, tw * th)
            vis_frac = vis_area / amodal_area

            ys, xs = np.where(vis_mask > 0)
            if xs.size == 0:
                present = 0
                vis_box = (-1, -1, 0, 0)
                occ_frames.append(i)
            else:
                present = 1
                vis_box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
                if occ_frames and reappear_frame == -1:
                    reappear_frame = i

            dx_gt = 0.0 if prev_x is None else float(tx - prev_x)
            dy_gt = 0.0 if prev_y is None else float(ty - prev_y)
            prev_x, prev_y = tx, ty

            fname = f"frame_{i:04d}.png"
            cv2.imwrite(str(seq_dir / fname), canvas)
            wcsv.writerow([
                i, fname, present,
                *vis_box, tx, ty, tw, th,
                round(vis_frac, 4), dx_gt, dy_gt, round(scale, 4),
                dx_dist, dy_dist, distractor_patch.shape[1], distractor_patch.shape[0],
            ])
    return len(occ_frames), reappear_frame


def generate_controlled_bicycle_experiments() -> None:
    """Create photo-generated controlled sequences, occlusion ladder, and flow pairs from CC0 photo."""
    bike_path = fetch_wikimedia()
    src = cv2.imread(str(bike_path))
    if src is None:
        raise RuntimeError(f"Could not read {bike_path}")

    ctrl_root = DATA_DIR / "controlled"
    w, h = 640, 480
    bg_base, target_base, shifter_dist, bike_dist, col_strip, pan_src = _build_bicycle_assets(src)

    # 1. Core Variant A: ctrl_bicycle_occ (40 frames, 5-frame occlusion)
    n_occ, reap = _generate_occ_sequence(
        ctrl_root / "bicycle_occ_seq",
        CSV_DIR / "ctrl_bicycle_occ_gt.csv",
        bg_base, target_base, shifter_dist, col_strip,
        col_x0=258, col_x1=398, n_frames=40,
        x0=52.0, vx=11.8, scale_amp=0.18, y_amp=24.0,
        distractor_x0=520.0, distractor_vx=-11.0, distractor_y=290,
    )
    print(f"ctrl_bicycle_occ: full_occ={n_occ} frames, reappear={reap}")

    # 2. Occlusion Ladder: identical trajectory (x0=42, vx=8.8, scale_amp=0.12, y_amp=14, 58 frames)
    ladder_specs = [
        ("ctrl_occ_ladder_3f", "occ_ladder_3f", 268, 384, shifter_dist, 560.0, -9.0, 310),
        ("ctrl_occ_ladder_10f", "occ_ladder_10f", 236, 416, shifter_dist, 560.0, -9.0, 310),
        ("ctrl_occ_ladder_20f", "occ_ladder_20f", 192, 460, shifter_dist, 560.0, -9.0, 310),
        # Similar distractor (bike_dist) crosses right outside exit (x~420..480) during reappearance!
        ("ctrl_occ_ladder_10f_distractor", "occ_ladder_10f_distractor", 236, 416, bike_dist, 560.0, -4.5, 165),
    ]
    for csv_stem, dir_name, cx0, cx1, dpatch, dx0, dvx, dy in ladder_specs:
        cnt, rp = _generate_occ_sequence(
            ctrl_root / dir_name,
            CSV_DIR / f"{csv_stem}_gt.csv",
            bg_base, target_base, dpatch, col_strip,
            col_x0=cx0, col_x1=cx1, n_frames=58,
            x0=42.0, vx=8.8, scale_amp=0.12, y_amp=14.0,
            distractor_x0=dx0, distractor_vx=dvx, distractor_y=dy,
        )
        print(f"{csv_stem}: full_occ={cnt} frames, reappear={rp}")

    # 3. Variant B: Whole-frame camera shift on CC0 photo (24 frames, analytical u=-6, v=-1)
    seq_b_dir = ctrl_root / "bicycle_cam_shift_seq"
    seq_b_dir.mkdir(parents=True, exist_ok=True)
    col_patch_70 = cv2.resize(col_strip, (70, h))
    roi0 = (268, 220, 108, 170)
    dx_cam, dy_cam = 6, 1
    with open(CSV_DIR / "ctrl_bicycle_cam_shift_gt.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "image_file", "present", "x", "y", "w", "h", "u_gt", "v_gt"])
        for i in range(24):
            ox, oy = 120 + i * dx_cam, 420 + i * dy_cam
            frame = pan_src[oy : oy + h, ox : ox + w].copy()
            col_world_x = 430 - i * dx_cam
            if 0 < col_world_x < w:
                c0, c1 = max(0, col_world_x), min(w, col_world_x + 70)
                if c0 < c1:
                    frame[:, c0:c1] = col_patch_70[:, : c1 - c0]
            bx = roi0[0] - i * dx_cam
            by = roi0[1] - i * dy_cam
            fname = f"frame_{i:04d}.png"
            cv2.imwrite(str(seq_b_dir / fname), frame)
            wcsv.writerow([i, fname, 1, bx, by, roi0[2], roi0[3], -float(dx_cam), -float(dy_cam)])

    # 4. Controlled Optical Flow Pairs (cam_shift, obj_occ, cam_plus_obj)
    pair_dir = ctrl_root / "flow_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    cam_f0 = pan_src[420 : 420 + h, 120 : 120 + w].copy()
    cam_f1 = pan_src[420 + dy_cam : 420 + dy_cam + h, 120 + dx_cam : 120 + dx_cam + w].copy()
    cv2.imwrite(str(pair_dir / "cam_shift_f0.png"), cam_f0)
    cv2.imwrite(str(pair_dir / "cam_shift_f1.png"), cam_f1)
    flow_cam = np.zeros((h, w, 2), dtype=np.float32)
    flow_cam[..., 0], flow_cam[..., 1] = -float(dx_cam), -float(dy_cam)
    valid_cam = np.ones((h, w), dtype=bool)
    valid_cam[:, :dx_cam] = False
    valid_cam[:dy_cam, :] = False
    write_flo(pair_dir / "cam_shift_gt.flo", flow_cam, valid_cam)

    # Object motion + column occlusion pair
    col_x0, col_x1 = 258, 398
    col_patch = cv2.resize(col_strip, (col_x1 - col_x0, h))
    cv2.rectangle(col_patch, (0, 0), (col_x1 - col_x0 - 1, h - 1), (45, 45, 45), 2)
    obj_f0, obj_f1 = bg_base.copy(), bg_base.copy()
    tx0, ty0, dx_o, dy_o = 190, 155, 12, 2
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
    flow_obj[vis_t0, 0], flow_obj[vis_t0, 1] = float(dx_o), float(dy_o)
    occluded_t0_target = vis_t0 & np.roll(mask_col, -dx_o, axis=1)
    covered_bg_at_t1 = (~mask_t0) & (~mask_col) & vis_t1
    valid_obj[occluded_t0_target | covered_bg_at_t1] = False
    write_flo(pair_dir / "obj_occ_gt.flo", flow_obj, valid_obj)

    # Camera pan (-5, -1) + moving bicycle target (+10, 0) for ego-motion compensation test
    cam_bg0 = pan_src[420 : 420 + h, 140 : 140 + w].copy()
    cam_bg1 = pan_src[421 : 421 + h, 145 : 145 + w].copy()
    _clipped_paste(cam_bg0, target_base, 180, 160)
    _clipped_paste(cam_bg1, target_base, 190, 160)  # +10 px net image motion while bg shifts (-5, -1)
    cv2.imwrite(str(pair_dir / "cam_plus_obj_f0.png"), cam_bg0)
    cv2.imwrite(str(pair_dir / "cam_plus_obj_f1.png"), cam_bg1)
    flow_cpo = np.zeros((h, w, 2), dtype=np.float32)
    flow_cpo[..., 0], flow_cpo[..., 1] = -5.0, -1.0
    flow_cpo[160 : 160 + th_o, 180 : 180 + tw_o, 0] = 10.0
    flow_cpo[160 : 160 + th_o, 180 : 180 + tw_o, 1] = 0.0
    valid_cpo = np.ones((h, w), dtype=bool)
    valid_cpo[:, :5] = False
    valid_cpo[:1, :] = False
    # Mask background covered by target at f1
    bg_covered = np.zeros((h, w), dtype=bool)
    bg_covered[160 : 160 + th_o, 190 : 190 + tw_o] = True
    bg_covered[160 : 160 + th_o, 180 : 180 + tw_o] = False
    valid_cpo[bg_covered] = False
    write_flo(pair_dir / "cam_plus_obj_gt.flo", flow_cpo, valid_cpo)

    print("Generated controlled CC0 experiments & occlusion ladder in", ctrl_root)


def write_dataset_manifest() -> Path:
    """Record source URL, ID, frame range, license, and SHA-256 for every dataset asset."""
    rows: List[List[str]] = []

    def add(source: str, item_id: str, frames_used: str, license_terms: str, path: Optional[Path]) -> None:
        digest = sha256_file(path) if path is not None and path.exists() else "-"
        rel = str(path.relative_to(DATA_DIR.parent)) if path is not None and path.exists() else "-"
        rows.append([source, item_id, frames_used, license_terms, digest, rel])

    add(WIKIMEDIA_BIKE_PAGE, "Wikimedia:Front_Side_of_Bicyle.jpg", "1 image (1200x1600) -> controlled suites",
        "CC0 1.0 Public Domain Dedication", DATA_DIR / "wikimedia" / "Front_Side_of_Bicyle.jpg")

    for seq in ("RubberWhale", "Grove2"):
        add("https://vision.middlebury.edu/flow/data/", f"Middlebury:{seq}:frame10", "frame10-frame11",
            "Middlebury Flow Evaluation (research/eval; raw excluded)", DATA_DIR / "middlebury" / "other-data" / seq / "frame10.png")
        add("https://vision.middlebury.edu/flow/data/", f"Middlebury:{seq}:flow10.flo", "frame10->frame11 GT",
            "Middlebury Flow Evaluation (research/eval; raw excluded)", DATA_DIR / "middlebury" / "other-gt-flow" / seq / "flow10.flo")

    for pass_name, seq, fnum in (("clean", "alley_1", "0001"), ("final", "ambush_5", "0010")):
        add("https://sintel.is.tue.mpg.de/", f"MPI-Sintel:{pass_name}/{seq}:frame_{fnum}", f"frame_{fnum}->next",
            "MPI Sintel Training (research/eval; raw excluded)", DATA_DIR / "sintel" / "training" / pass_name / seq / f"frame_{fnum}.png")
        add("https://sintel.is.tue.mpg.de/", f"MPI-Sintel:flow/{seq}/frame_{fnum}.flo", f"frame_{fnum} GT",
            "MPI Sintel Training (research/eval; raw excluded)", DATA_DIR / "sintel" / "training" / "flow" / seq / f"frame_{fnum}.flo")

    for seq, fcount in (("bmx-trees", "00000-00079 (80 frames)"), ("car-shadow", "00000-00039 (40 frames)"), ("scooter-black", "00000-00042 (43 frames)")):
        add("https://davischallenge.org/davis2017/code.html", f"DAVIS-2017:{seq}:00000.jpg", fcount,
            "CC BY-NC 4.0 (raw excluded)", DATA_DIR / "davis" / "DAVIS" / "JPEGImages" / "480p" / seq / "00000.jpg")
        add("https://davischallenge.org/davis2017/code.html", f"DAVIS-2017:{seq}:00000.png", fcount,
            "CC BY-NC 4.0 (raw excluded)", DATA_DIR / "davis" / "DAVIS" / "Annotations" / "480p" / seq / "00000.png")

    add("https://huggingface.co/datasets/l-lt/LaSOT", "LaSOT:bicycle-11:00000001.jpg", "00000001-00000120 (120 frames, full occ 74-78)",
        "LaSOT academic research license (raw excluded)", DATA_DIR / "lasot" / "bicycle-11" / "img" / "00000001.jpg")
    add("https://huggingface.co/datasets/l-lt/LaSOT", "LaSOT:bicycle-11:groundtruth.txt", "120 frames bbox GT",
        "LaSOT academic research license (raw excluded)", DATA_DIR / "lasot" / "bicycle-11" / "groundtruth.txt")

    for name, rel in (
        ("Controlled:obj_occ_gt.flo", DATA_DIR / "controlled" / "flow_pairs" / "obj_occ_gt.flo"),
        ("Controlled:cam_shift_gt.flo", DATA_DIR / "controlled" / "flow_pairs" / "cam_shift_gt.flo"),
        ("Controlled:cam_plus_obj_gt.flo", DATA_DIR / "controlled" / "flow_pairs" / "cam_plus_obj_gt.flo"),
    ):
        add("Photo-generated from CC0 Front_Side_of_Bicyle.jpg", name, "frame0->frame1",
            "CC0 1.0 derived controlled experiment", rel)

    out_csv = CSV_DIR / "dataset_manifest.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_url", "dataset_id", "frames_used", "license_or_terms", "sha256", "local_path"])
        w.writerows(rows)
    print("Dataset manifest written:", out_csv)
    return out_csv


def main() -> None:
    ensure_dirs()
    ap = argparse.ArgumentParser(description="Prepare datasets and controlled CC0 suites.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fetch-middlebury", action="store_true")
    ap.add_argument("--fetch-sintel", action="store_true")
    ap.add_argument("--fetch-lasot", action="store_true")
    ap.add_argument("--fetch-wikimedia", action="store_true")
    ap.add_argument("--fetch-davis", nargs="*", default=None)
    ap.add_argument("--generate-controlled", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    args = ap.parse_args()

    if args.all or args.fetch_middlebury:
        fetch_middlebury()
    if args.all or args.fetch_sintel:
        fetch_sintel_pairs()
    if args.all or args.fetch_lasot:
        fetch_lasot_subset()
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
