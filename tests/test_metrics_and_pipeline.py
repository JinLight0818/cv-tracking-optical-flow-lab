"""Unit and integration verification tests for metrics, bindings, and CSV integrity."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import CSV_DIR
from metrics import epe, epe_map, iou, mask_to_bbox, read_flo, validate_box, write_flo
from tracker_binding_check import StaticBaselineTracker, resolve_tracker_factory, safe_tracker_init


class MetricsAndBindingTests(unittest.TestCase):
    def test_iou_exact_and_disjoint(self) -> None:
        self.assertAlmostEqual(iou((10, 10, 20, 20), (10, 10, 20, 20)), 1.0)
        self.assertAlmostEqual(iou((10, 10, 20, 20), (20, 10, 20, 20)), 1.0 / 3.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (50, 50, 10, 10)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 0, 10), (0, 0, 10, 10)), 0.0)

    def test_epe_and_map(self) -> None:
        self.assertAlmostEqual(epe((3.0, 4.0), (0.0, 0.0)), 5.0)
        pred = np.zeros((4, 4, 2), dtype=np.float32)
        gt = np.zeros((4, 4, 2), dtype=np.float32)
        gt[..., 0] = 3.0
        gt[..., 1] = 4.0
        valid = np.ones((4, 4), dtype=bool)
        valid[0, 0] = False
        err, count = epe_map(pred, gt, valid)
        self.assertEqual(count, 15)
        self.assertAlmostEqual(err, 5.0, places=5)

    def test_flo_roundtrip_and_invalid_mask(self) -> None:
        flo = np.zeros((8, 10, 2), dtype=np.float32)
        flo[..., 0] = -2.5
        flo[..., 1] = 1.25
        valid = np.ones((8, 10), dtype=bool)
        valid[2:4, 3:5] = False
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.flo"
            write_flo(p, flo, valid)
            loaded, loaded_valid = read_flo(p)
            self.assertEqual(loaded.shape, (8, 10, 2))
            self.assertEqual(int(loaded_valid.sum()), 76)
            self.assertTrue(np.allclose(loaded[loaded_valid], flo[loaded_valid]))

    def test_mask_to_bbox_and_empty_mask(self) -> None:
        m = np.zeros((20, 30), dtype=np.uint8)
        self.assertIsNone(mask_to_bbox(m, 1))
        m[5:10, 8:14] = 2
        self.assertEqual(mask_to_bbox(m, 2), (8, 5, 6, 5))

    def test_tracker_init_handles_none_bool_and_empty_roi(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:60, 20:60] = 255
        st = StaticBaselineTracker()
        ok, _ = safe_tracker_init(st, img, (0, 0, 0, 0))
        self.assertFalse(ok)
        ok_good, _ = safe_tracker_init(st, img, (20, 20, 40, 40))
        self.assertTrue(ok_good)

        fn, _ = resolve_tracker_factory("CSRT")
        self.assertIsNotNone(fn)
        ok_csrt, note = safe_tracker_init(fn(), img, (20, 20, 40, 40))
        self.assertTrue(ok_csrt)
        self.assertIn(note, ("returned_None", "returned_True"))

    def test_validate_box(self) -> None:
        self.assertTrue(validate_box((10, 10, 20, 20), (100, 100, 3)))
        self.assertFalse(validate_box((120, 10, 20, 20), (100, 100, 3)))
        self.assertFalse(validate_box((10, 10, -5, 20), (100, 100, 3)))

    def test_generated_csv_integrity(self) -> None:
        ts_path = CSV_DIR / "track_summary.csv"
        fs_path = CSV_DIR / "flow_summary.csv"
        if ts_path.exists():
            with open(ts_path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 28)
            for r in rows:
                self.assertGreaterEqual(float(r["mean_iou"]), 0.0)
                self.assertLessEqual(float(r["mean_iou"]), 1.0)
        if fs_path.exists():
            with open(fs_path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 12)


if __name__ == "__main__":
    unittest.main()
