"""Tracker constructors, safe initialization, and hybrid CSRT_ReDetect tracker."""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from metrics import iou


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


class CSRTReDetectTracker:
    """Hybrid tracker: CSRT for local tracking + multi-scale template/HSV re-detection.

    When CSRT reports failure (or appearance similarity collapses) for `fail_streak_trigger`
    consecutive frames, searches the frame using a multi-scale template + HSV histogram
    verifier derived from the initial confirmed ROI. Once a candidate passes the similarity
    threshold consistently across `confirm_streak` frames, re-initializes a fresh CSRT.
    """

    def __init__(
        self,
        fail_streak_trigger: int = 2,
        confirm_streak: int = 2,
        score_thresh: float = 0.50,
        scales: Tuple[float, ...] = (0.75, 0.88, 1.0, 1.15, 1.30),
    ) -> None:
        self.fail_streak_trigger = fail_streak_trigger
        self.confirm_streak = confirm_streak
        self.score_thresh = score_thresh
        self.scales = scales

        self._csrt = None
        self._tmpl_gray: Optional[np.ndarray] = None
        self._tmpl_hist: Optional[np.ndarray] = None
        self._base_wh: Tuple[int, int] = (0, 0)
        self._fail_streak = 0
        self._lost_mode = False
        self._cand_boxes: List[Tuple[float, float, float, float]] = []

        # Telemetry
        self.redetect_calls = 0
        self.redetect_ms_total = 0.0
        self.reinits_count = 0
        self.last_mode = "init"

    @staticmethod
    def _hsv_hist(bgr_patch: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def _crop_clamped(image: np.ndarray, roi: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        fh, fw = image.shape[:2]
        x, y, w, h = (int(round(v)) for v in roi)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return image[y0:y1, x0:x1]

    def init(self, image: np.ndarray, roi: Tuple[float, float, float, float]) -> bool:
        if image is None or image.size == 0 or roi[2] <= 0 or roi[3] <= 0:
            return False
        patch = self._crop_clamped(image, roi)
        if patch is None:
            return False
        self._tmpl_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        self._tmpl_hist = self._hsv_hist(patch)
        self._base_wh = (patch.shape[1], patch.shape[0])
        self._csrt = cv2.TrackerCSRT_create()
        self._csrt.init(image, tuple(int(round(v)) for v in roi))
        self._fail_streak = 0
        self._lost_mode = False
        self._cand_boxes = []
        self.redetect_calls = 0
        self.redetect_ms_total = 0.0
        self.reinits_count = 0
        self.last_mode = "track"
        return True

    def _appearance_score(self, patch: np.ndarray) -> float:
        if patch is None or self._tmpl_gray is None or self._tmpl_hist is None:
            return 0.0
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self._tmpl_gray.shape[1], self._tmpl_gray.shape[0]))
        ncc = float(cv2.matchTemplate(resized, self._tmpl_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
        hist = self._hsv_hist(patch)
        hist_cor = float(cv2.compareHist(self._tmpl_hist, hist, cv2.HISTCMP_CORREL))
        return 0.7 * ncc + 0.3 * max(0.0, hist_cor)

    def _search_frame(self, image: np.ndarray) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
        """Run multi-scale normalized cross-correlation + HSV verification over image."""
        assert self._tmpl_gray is not None and self._tmpl_hist is not None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ih, iw = gray.shape
        bw, bh = self._base_wh

        # Downscale search 0.5x for speed, then verify HSV on full-res candidate
        ds = 0.5
        gray_s = cv2.resize(gray, (max(16, int(iw * ds)), max(16, int(ih * ds))), interpolation=cv2.INTER_AREA)

        best_score = -1.0
        best_box: Optional[Tuple[float, float, float, float]] = None

        for sc in self.scales:
            tw = max(8, int(round(bw * sc * ds)))
            th = max(8, int(round(bh * sc * ds)))
            if tw >= gray_s.shape[1] or th >= gray_s.shape[0]:
                continue
            tmpl_s = cv2.resize(self._tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray_s, tmpl_s, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val < self.score_thresh * 0.75:
                continue
            cand = (max_loc[0] / ds, max_loc[1] / ds, bw * sc, bh * sc)
            patch = self._crop_clamped(image, cand)
            if patch is None:
                continue
            hist_cor = max(0.0, float(cv2.compareHist(self._tmpl_hist, self._hsv_hist(patch), cv2.HISTCMP_CORREL)))
            combined = 0.7 * float(max_val) + 0.3 * hist_cor
            if combined > best_score:
                best_score = combined
                best_box = cand
        return best_score, best_box

    def update(self, image: np.ndarray) -> Tuple[bool, Tuple[float, float, float, float]]:
        if self._csrt is None or image is None or image.size == 0:
            self.last_mode = "uninitialized"
            return False, (0.0, 0.0, 0.0, 0.0)

        if not self._lost_mode:
            ok, box = self._csrt.update(image)
            box_f = tuple(float(v) for v in box)
            app_ok = False
            if ok:
                patch = self._crop_clamped(image, box_f)
                app_ok = patch is not None and self._appearance_score(patch) >= 0.22
            if ok and app_ok:
                self._fail_streak = 0
                self.last_mode = "track"
                return True, box_f

            self._fail_streak += 1
            if self._fail_streak < self.fail_streak_trigger:
                self.last_mode = "track_uncertain" if ok else "csrt_failed"
                return bool(ok), box_f
            self._lost_mode = True
            self._cand_boxes = []

        # Re-detection mode
        t0 = time.perf_counter()
        self.redetect_calls += 1
        score, cand = self._search_frame(image)
        self.redetect_ms_total += (time.perf_counter() - t0) * 1000.0

        if cand is not None and score >= self.score_thresh:
            if self._cand_boxes and iou(self._cand_boxes[-1], cand) >= 0.35:
                self._cand_boxes.append(cand)
            else:
                self._cand_boxes = [cand]
            if len(self._cand_boxes) >= self.confirm_streak:
                self._csrt = cv2.TrackerCSRT_create()
                self._csrt.init(image, tuple(int(round(v)) for v in cand))
                self._lost_mode = False
                self._fail_streak = 0
                self._cand_boxes = []
                self.reinits_count += 1
                self.last_mode = "reinit"
                return True, cand
            self.last_mode = "redetect_candidate"
            return False, (0.0, 0.0, 0.0, 0.0)

        self._cand_boxes = []
        self.last_mode = "lost_searching"
        return False, (0.0, 0.0, 0.0, 0.0)


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
    if key == "CSRT_REDETECT":
        return CSRTReDetectTracker, "CSRTReDetectTracker(cv2.TrackerCSRT_create)"

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
    """Initialize an OpenCV tracker while normalizing None vs bool return values."""
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
