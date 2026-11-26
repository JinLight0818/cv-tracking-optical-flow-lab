"""Probe OpenCV tracker constructors across cv2 and cv2.legacy namespaces."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from config import CSV_DIR, PDF_TRACKERS, ensure_dirs


class StaticBaselineTracker:
    """Simple baseline that keeps the initial frame-0 ROI unchanged."""

    def __init__(self) -> None:
        self._roi: Optional[Tuple[float, float, float, float]] = None

    def init(self, image: np.ndarray, roi: Tuple[float, float, float, float]) -> bool:
        if image is None or image.size == 0 or roi[2] <= 0 or roi[3] <= 0:
            return False
        self._roi = tuple(float(v) for v in roi)
        return True

    def update(self, image: np.ndarray) -> Tuple[bool, Tuple[float, float, float, float]]:
        if self._roi is None or image is None or image.size == 0:
            return False, (0.0, 0.0, 0.0, 0.0)
        return True, self._roi


TRACKER_CONSTRUCTOR_MAP = {
    "BOOSTING": "TrackerBoosting_create",
    "MIL": "TrackerMIL_create",
    "KCF": "TrackerKCF_create",
    "TLD": "TrackerTLD_create",
    "MEDIANFLOW": "TrackerMedianFlow_create",
    "CSRT": "TrackerCSRT_create",
    "MOSSE": "TrackerMOSSE_create",
}


def resolve_tracker_factory(name: str, prefer_legacy: bool = False) -> Tuple[Optional[Callable], str]:
    """Return (constructor_callable, binding_label) for a tracker name."""
    key = name.upper()
    if key == "STATIC":
        return StaticBaselineTracker, "StaticBaselineTracker"

    attr = TRACKER_CONSTRUCTOR_MAP.get(key)
    if attr is None:
        return None, "UNSUPPORTED"

    legacy_ns = getattr(cv2, "legacy", None)
    order = [("cv2.legacy", legacy_ns), ("cv2", cv2)] if prefer_legacy else [("cv2", cv2), ("cv2.legacy", legacy_ns)]
    for ns_label, ns in order:
        if ns is None:
            continue
        fn = getattr(ns, attr, None)
        if fn is not None:
            return fn, f"{ns_label}.{attr}"
    return None, "MISSING"


def safe_tracker_init(tracker, frame: np.ndarray, roi: Tuple[float, float, float, float]) -> Tuple[bool, str]:
    """Initialize an OpenCV tracker while normalizing None vs bool return values.

    In OpenCV 4.x/5.x, cv2.Tracker*.init() returns None (void) while
    cv2.legacy.Tracker*.init() returns a boolean.
    """
    if frame is None or frame.size == 0:
        return False, "empty_frame"
    if len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
        return False, "empty_roi"
    try:
        ret = tracker.init(frame, roi)
    except Exception as exc:
        return False, f"exception:{type(exc).__name__}:{exc}"
    if ret is None:
        return True, "returned_None"
    return bool(ret), f"returned_{ret!r}"


def probe_all() -> list[dict]:
    """Probe cv2 and cv2.legacy constructors, init return types, and empty-ROI handling."""
    rng = np.random.default_rng(42)
    img = rng.integers(30, 90, (160, 200, 3), dtype=np.uint8)
    # Textured target inside ROI so optical-flow trackers (MedianFlow) can match features
    for yy in range(40, 90, 8):
        for xx in range(40, 90, 8):
            color = (220, 220, 220) if ((xx + yy) // 8) % 2 == 0 else (60, 140, 230)
            cv2.rectangle(img, (xx, yy), (xx + 6, yy + 6), color, -1)
    roi = (40, 40, 50, 50)

    records = []
    for name in PDF_TRACKERS + ["CSRT", "MOSSE"]:
        attr = TRACKER_CONSTRUCTOR_MAP[name]
        for ns_label, ns in [("cv2", cv2), ("cv2.legacy", getattr(cv2, "legacy", None))]:
            fn = getattr(ns, attr, None) if ns is not None else None
            if fn is None:
                records.append({
                    "tracker": name,
                    "binding": f"{ns_label}.{attr}",
                    "available": 0,
                    "class_name": "-",
                    "init_return": "-",
                    "update_ok": 0,
                    "empty_roi_handled": "-",
                    "opencv_version": cv2.__version__,
                })
                continue
            try:
                t = fn()
                ret = t.init(img, roi)
                upd_ok, _ = t.update(img)
                t_empty = fn()
                ok_empty, msg_empty = safe_tracker_init(t_empty, img, (0, 0, 0, 0))
                records.append({
                    "tracker": name,
                    "binding": f"{ns_label}.{attr}",
                    "available": 1,
                    "class_name": type(t).__name__,
                    "init_return": repr(ret),
                    "update_ok": int(bool(upd_ok)),
                    "empty_roi_handled": f"rejected({msg_empty})" if not ok_empty else "accepted",
                    "opencv_version": cv2.__version__,
                })
            except Exception as exc:
                records.append({
                    "tracker": name,
                    "binding": f"{ns_label}.{attr}",
                    "available": 0,
                    "class_name": type(exc).__name__,
                    "init_return": str(exc),
                    "update_ok": 0,
                    "empty_roi_handled": "-",
                    "opencv_version": cv2.__version__,
                })
    return records


def main() -> None:
    ensure_dirs()
    print("Python:", sys.version.split()[0])
    print("OpenCV:", cv2.__version__)
    print("Has cv2.legacy:", hasattr(cv2, "legacy"))
    rows = probe_all()
    for r in rows:
        flag = "OK  " if r["available"] else "MISS"
        print(f"[{flag}] {r['binding']:38s} cls={r['class_name']:20s} init={r['init_return']:6s} upd={r['update_ok']}")

    out_csv = CSV_DIR / "tracker_bindings.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
