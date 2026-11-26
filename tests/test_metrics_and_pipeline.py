"""Unit and integration tests for metrics, bindings, stem matching, LK point IDs, and CSRT_ReDetect."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from config import CSV_DIR
from custom_video_lab import load_verified_gt
from dataset_prep import match_frames_and_masks
from metrics import compute_flow_regions, compute_recovery_stats, epe, epe_map, iou, mask_to_bbox, read_flo, validate_box, write_flo
from trackers import CSRTReDetectTracker, StaticBaselineTracker, resolve_tracker_factory, safe_tracker_init


class MetricsAndPipelineTests(unittest.TestCase):
    def test_iou_exact_and_disjoint(self) -> None:
        self.assertAlmostEqual(iou((10, 10, 20, 20), (10, 10, 20, 20)), 1.0)
        self.assertAlmostEqual(iou((10, 10, 20, 20), (20, 10, 20, 20)), 1.0 / 3.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (50, 50, 10, 10)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 0, 10), (0, 0, 10, 10)), 0.0)

    def test_epe_and_region_masks(self) -> None:
        self.assertAlmostEqual(epe((3.0, 4.0), (0.0, 0.0)), 5.0)
        pred = np.zeros((64, 64, 2), dtype=np.float32)
        gt = np.zeros((64, 64, 2), dtype=np.float32)
        gt[..., 0], gt[..., 1] = 3.0, 4.0
        valid = np.ones((64, 64), dtype=bool)
        valid[0, 0] = False
        err, count = epe_map(pred, gt, valid)
        self.assertEqual(count, 64 * 64 - 1)
        self.assertAlmostEqual(err, 5.0, places=5)

        img = np.random.default_rng(7).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        regs = compute_flow_regions(img, gt, valid)
        self.assertGreater(int(regs["border_band"].sum()), 0)
        self.assertGreater(int(regs["low_texture"].sum()), 0)
        self.assertGreater(int(regs["high_texture"].sum()), 0)

    def test_flo_roundtrip_and_invalid_mask(self) -> None:
        flo = np.zeros((8, 10, 2), dtype=np.float32)
        flo[..., 0], flo[..., 1] = -2.5, 1.25
        valid = np.ones((8, 10), dtype=bool)
        valid[2:4, 3:5] = False
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.flo"
            write_flo(p, flo, valid)
            loaded, loaded_valid = read_flo(p)
            self.assertEqual(loaded.shape, (8, 10, 2))
            self.assertEqual(int(loaded_valid.sum()), 76)
            self.assertTrue(np.allclose(loaded[loaded_valid], flo[loaded_valid]))

    def test_mask_to_bbox_and_stem_matching_detects_missing_frame(self) -> None:
        m = np.zeros((20, 30), dtype=np.uint8)
        self.assertIsNone(mask_to_bbox(m, 1))
        m[5:10, 8:14] = 2
        self.assertEqual(mask_to_bbox(m, 2), (8, 5, 6, 5))

        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp) / "imgs"
            msk_dir = Path(tmp) / "masks"
            img_dir.mkdir()
            msk_dir.mkdir()
            cv2.imwrite(str(img_dir / "00000.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))
            cv2.imwrite(str(img_dir / "00001.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))
            cv2.imwrite(str(msk_dir / "00000.png"), np.zeros((20, 20), dtype=np.uint8))
            cv2.imwrite(str(msk_dir / "00002.png"), np.zeros((20, 20), dtype=np.uint8))
            matched, rep = match_frames_and_masks(img_dir, msk_dir)
            self.assertEqual(len(matched), 1)
            self.assertEqual(rep["missing_masks"], ["00001"])
            self.assertEqual(rep["missing_imgs"], ["00002"])

    def test_recovery_start_and_confirmed_frames(self) -> None:
        rows = [
            {"frame": 23, "gt_present": 1, "iou": 0.30},
            {"frame": 28, "gt_present": 1, "iou": 0.45},
            {"frame": 29, "gt_present": 1, "iou": 0.74},
            {"frame": 30, "gt_present": 1, "iou": 0.52},
            {"frame": 31, "gt_present": 1, "iou": 0.53},
        ]
        s_off, c_off, f_start, f_conf = compute_recovery_stats(rows, reappear_frame=23, iou_threshold=0.5, consecutive=3)
        self.assertEqual((s_off, c_off, f_start, f_conf), (6, 8, 29, 31))
        # Zero eval frames / never recovered
        self.assertEqual(compute_recovery_stats([], reappear_frame=23), (None, None, None, None))
        self.assertEqual(compute_recovery_stats(rows[:2], reappear_frame=None), (None, None, None, None))

    def test_tracker_init_handles_none_bool_empty_and_redetect(self) -> None:
        rng = np.random.default_rng(12)
        bg = rng.integers(30, 80, (160, 220, 3), dtype=np.uint8)
        patch = rng.integers(120, 250, (44, 44, 3), dtype=np.uint8)
        cv2.rectangle(patch, (6, 6), (38, 38), (0, 220, 255), 3)
        f0 = bg.copy()
        f0[40:84, 30:74] = patch

        st = StaticBaselineTracker()
        self.assertFalse(safe_tracker_init(st, f0, (0, 0, 0, 0))[0])
        self.assertTrue(safe_tracker_init(st, f0, (30, 40, 44, 44))[0])

        fn, _ = resolve_tracker_factory("CSRT")
        self.assertIsNotNone(fn)
        ok_csrt, note = safe_tracker_init(fn(), f0, (30, 40, 44, 44))
        self.assertTrue(ok_csrt)
        self.assertIn(note, ("returned_None", "returned_True"))

        # Test CSRTReDetectTracker recovers after blank occlusion frames
        rd = CSRTReDetectTracker(fail_streak_trigger=1, confirm_streak=2, score_thresh=0.45)
        self.assertTrue(rd.init(f0, (30, 40, 44, 44)))
        blank = bg.copy()
        for _ in range(2):
            ok, _ = rd.update(blank)
            self.assertFalse(ok)
        f_reapp = bg.copy()
        f_reapp[60:104, 140:184] = patch
        rd.update(f_reapp)
        ok2, box2 = rd.update(f_reapp)
        self.assertTrue(ok2)
        self.assertGreater(iou(box2, (140, 60, 44, 44)), 0.7)

    def test_validate_box_and_csv_integrity(self) -> None:
        self.assertTrue(validate_box((10, 10, 20, 20), (100, 100, 3)))
        self.assertFalse(validate_box((120, 10, 20, 20), (100, 100, 3)))
        self.assertFalse(validate_box((10, 10, -5, 20), (100, 100, 3)))
        ts_path = CSV_DIR / "track_summary.csv"
        if ts_path.exists():
            with open(ts_path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 40)
            for r in rows:
                if r["absent_frames"] == "0":
                    self.assertEqual(r["absent_fp_pct"], "N/A (0 кадр)")

        ab_path = CSV_DIR / "flow_ablation.csv"
        if ab_path.exists():
            with open(ab_path, newline="") as f:
                ab_rows = list(csv.DictReader(f))
            for r in ab_rows:
                self.assertEqual(int(r["final_active_pts"]), int(r["retained_initial_pts"]) + int(r["redetected_pts"]))
                self.assertLessEqual(int(r["retained_initial_pts"]), int(r["init_pts"]))

        cc_path = CSV_DIR / "camera_compensation.csv"
        if cc_path.exists():
            with open(cc_path, newline="") as f:
                cc_rows = list(csv.DictReader(f))
            dis_bmx = next(r for r in cc_rows if r["pair"] == "DAVIS:bmx_trees_f10" and r["flow_method"] == "dis_medium")
            self.assertGreater(float(dis_bmx["bg_reduction_pct"]), 80.0)

    def test_load_verified_gt_rejects_mismatches_and_checks_phone_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gt.csv"
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "orig_frame", "timestamp_s", "present", "gt_state", "x", "y", "w", "h"])
                w.writerow([0, 42, 1.75, 1, "visible", 10, 20, 30, 40])
                w.writerow([1, 43, 1.79, 1, "visible", 12, 20, 30, 40])

            loaded = load_verified_gt(p, [42, 43])
            self.assertEqual(len(loaded), 2)
            with self.assertRaises(ValueError):
                load_verified_gt(p, [42, 43, 44])
            with self.assertRaises(ValueError):
                load_verified_gt(p, [42, 99])

        pv_path = CSV_DIR / "phone_videos_summary.csv"
        if pv_path.exists():
            with open(pv_path, newline="") as f:
                pv_rows = list(csv.DictReader(f))
            self.assertEqual(len(pv_rows), 24)
            for r in pv_rows:
                self.assertEqual(r["absent_frames"], "0")
                self.assertEqual(r["absent_fp_pct"], "N/A (0 кадр)")

        lm_path = CSV_DIR / "phone_commons_leman_express_summary.csv"
        if lm_path.exists():
            with open(lm_path, newline="") as f:
                lm_rows = list(csv.DictReader(f))
            self.assertEqual(len(lm_rows), 8)
            gif_path = CSV_DIR.parent / "figures" / "custom_videos" / "tracking_demo.gif"
            self.assertTrue(gif_path.exists())
            self.assertLess(gif_path.stat().st_size, 5_500_000)


if __name__ == "__main__":
    unittest.main()
