"""Probe OpenCV tracker constructors across cv2 and cv2.legacy namespaces."""

from __future__ import annotations

import csv
import sys

import cv2
import numpy as np

from config import CSV_DIR, PDF_TRACKERS, ensure_dirs
from trackers import (
    TRACKER_CONSTRUCTOR_MAP,
    StaticBaselineTracker,
    resolve_tracker_factory,
    safe_tracker_init,
)


def probe_all() -> list[dict]:
    """Probe cv2 and cv2.legacy constructors, init return types, and empty-ROI handling."""
    rng = np.random.default_rng(42)
    img = rng.integers(30, 90, (160, 200, 3), dtype=np.uint8)
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
