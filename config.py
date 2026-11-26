"""Shared configuration, PDF hyperparameters, and directory paths."""

from __future__ import annotations

import os
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"
LOCAL_VIZ_DIR = RESULTS_DIR / "local_only"
VIDEOS_DIR = RESULTS_DIR / "videos"

# Single-thread CPU fairness settings
NUM_THREADS = 1


def enforce_single_thread() -> None:
    """Pin OpenCV and BLAS thread counts for fair single-CPU timing."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(NUM_THREADS)
    cv2.setNumThreads(NUM_THREADS)


# --- Optical Flow parameters (06-Optical-Flow.ipynb - Colaboratory.pdf, pp. 1-3) ---
SHI_TOMASI_PDF = {
    "maxCorners": 10,
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7,
}

LK_PDF = {
    "winSize": (200, 200),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
}

LK_SMALL = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
}

FARNEBACK_PDF = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

# --- Tracking algorithms & evaluation thresholds ---
PDF_TRACKERS = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW"]
ALL_TRACKERS = ["STATIC", "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "CSRT", "CSRT_REDETECT"]

IOU_THRESHOLD = 0.5
RECOVERY_CONSECUTIVE = 3
WARMUP_RUNS = 1
TIMING_REPEATS = 5
FB_THRESH_PX = 1.0
FARNEBACK_VIS_MAX_MAG = 12.0


def ensure_dirs() -> None:
    """Create standard output directories if they do not exist."""
    for path in (DATA_DIR, RESULTS_DIR, CSV_DIR, FIGURES_DIR, LOCAL_VIZ_DIR, VIDEOS_DIR):
        path.mkdir(parents=True, exist_ok=True)
