#!/usr/bin/env python3
"""
QuackOps — Phase C: qpsMarkerDetector Live Hardware Test

Validates the production qpsMarkerDetector class against real Pi Camera
Module 3 frames with a physical ArUco marker (DICT_4X4_50, ID 0).

Test flow:
  1. Construct qpsConfig (640×480 @ 30fps, DICT_4X4_50, target_marker_id=0)
  2. Start qpsPiCameraManager (cam0, the bottom-facing wide camera)
  3. Construct qpsMarkerDetector
  4. For 10 seconds, grab frames and run detection
  5. Per frame: log detection result with pixel center
  6. Save annotated frames showing marker detection (raw + bounding box)
  7. Compute and report statistics: detection rate, mean center, FPS

Run on the Pi from project root:
    cd ~/SeniorD/QuackOpsPi
    source venv/bin/activate
    python3 cameratests/test_phase_c_marker_detector.py

Pre-flight:
  - Print ArUco DICT_4X4_50 marker, ID 0 (use the larger 8-9 inch one)
  - Place marker flat in cam0's field of view (bottom-facing)
  - Lighting should be reasonable; no glare/reflections
  - Camera roughly 30-100cm from marker
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Make project importable when run from cameratests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_pi_camera_manager import qpsPiCameraManager
from quackops_pi.vision.qps_marker_detector import qpsMarkerDetector


# ── Config ────────────────────────────────────────────────────────────
TEST_DURATION_S = 10.0
DETECTION_LOG_INTERVAL = 5      # log every Nth detection result
SAMPLE_FRAMES_TO_SAVE = 5
OUTPUT_DIR = Path(__file__).parent / "captures"


# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qps.phase_c")


def annotate_frame(frame: np.ndarray, detections, frame_idx: int) -> np.ndarray:
    """Overlay detection results on a frame for visual verification."""
    annotated = frame.copy()
    h, w = frame.shape[:2]
    cx_img, cy_img = w // 2, h // 2

    # Draw image center crosshair (where we're aiming for landing)
    cv2.drawMarker(annotated, (cx_img, cy_img), (0, 255, 255),
                   cv2.MARKER_CROSS, 30, 2)

    # Draw detection overlays
    for det in detections:
        # Draw the marker boundary
        corners = det.corners.astype(int)
        cv2.polylines(annotated, [corners], isClosed=True,
                      color=(0, 255, 0), thickness=3)

        # Draw the marker's pixel center
        cx, cy = int(det.center_px[0]), int(det.center_px[1])
        cv2.drawMarker(annotated, (cx, cy), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)

        # Label with marker ID
        cv2.putText(annotated, f"ID:{det.marker_id}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Line from image center to marker center (offset visualization)
        cv2.line(annotated, (cx_img, cy_img), (cx, cy), (255, 0, 255), 2)

    # Frame metadata in corner
    status = "DETECTED" if detections else "no marker"
    color = (0, 255, 0) if detections else (0, 0, 255)
    cv2.putText(annotated, f"Frame {frame_idx}: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return annotated


async def main():
    log.info("=" * 60)
    log.info("QuackOps — Phase C: qpsMarkerDetector Live Test")
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = qpsConfig()
    log.info(
        f"Config: {config.camera_resolution[0]}x{config.camera_resolution[1]} "
        f"@ {config.camera_fps}fps  dict={config.aruco_dictionary}  "
        f"target_id={config.target_marker_id}"
    )

    # ── Step 1: Start camera ──
    log.info("[1/4] Starting camera...")
    cam = qpsPiCameraManager(config)
    await cam.start()
    log.info("  ✓ Camera started")

    # Wait for first frame
    t0 = time.time()
    while time.time() - t0 < 5.0:
        if await cam.get_frame() is not None:
            break
        await asyncio.sleep(0.05)
    else:
        log.error("✗ No frame from camera within 5s")
        await cam.stop()
        sys.exit(1)
    log.info(f"  ✓ First frame ready in {time.time() - t0:.2f}s")

    # ── Step 2: Construct detector ──
    log.info("[2/4] Constructing detector...")
    detector = qpsMarkerDetector(config)
    log.info(f"  ✓ Detector ready (dict={config.aruco_dictionary})")

    # ── Step 3: Detection loop ──
    log.info(f"[3/4] Running detection for {TEST_DURATION_S}s...")
    log.info("    Place marker (DICT_4X4_50, ID 0) in cam0's view")

    stats = {
        "frames_processed": 0,
        "frames_with_target": 0,           # detections containing target_marker_id
        "frames_with_any_marker": 0,       # detections with anything
        "centers_x": [],
        "centers_y": [],
        "detection_durations_ms": [],
    }
    saved_frames = []

    t_start = time.time()
    last_log_idx = -1

    while time.time() - t_start < TEST_DURATION_S:
        frame = await cam.get_frame()
        if frame is None:
            await asyncio.sleep(0.01)
            continue

        # Run detection (offloaded to thread by qpsMarkerDetector)
        t_det_start = time.time()
        detections = await detector.detect(frame)
        t_det_ms = (time.time() - t_det_start) * 1000

        stats["frames_processed"] += 1
        stats["detection_durations_ms"].append(t_det_ms)

        target_detections = [d for d in detections if d.marker_id == config.target_marker_id]

        if detections:
            stats["frames_with_any_marker"] += 1
        if target_detections:
            stats["frames_with_target"] += 1
            for d in target_detections:
                stats["centers_x"].append(d.center_px[0])
                stats["centers_y"].append(d.center_px[1])

            # Log every Nth detection
            if stats["frames_with_target"] % DETECTION_LOG_INTERVAL == 1:
                d = target_detections[0]
                log.info(
                    f"    Frame {stats['frames_processed']:4d}: ID={d.marker_id} "
                    f"center=({d.center_px[0]:6.1f}, {d.center_px[1]:6.1f}) "
                    f"detect={t_det_ms:.1f}ms"
                )

        # Save up to N annotated frames spread across the test
        save_every = max(1, int(config.camera_fps * TEST_DURATION_S / SAMPLE_FRAMES_TO_SAVE))
        if (stats["frames_processed"] % save_every == 0
                and len(saved_frames) < SAMPLE_FRAMES_TO_SAVE):
            annotated = annotate_frame(frame, detections, stats["frames_processed"])
            path = OUTPUT_DIR / f"phase_c_frame_{len(saved_frames):02d}.jpg"
            cv2.imwrite(str(path), annotated)
            saved_frames.append(path.name)

        # Don't burn CPU
        await asyncio.sleep(1.0 / config.camera_fps / 2)

    elapsed = time.time() - t_start
    log.info(f"  ✓ Detection loop complete ({elapsed:.2f}s)")

    # ── Step 4: Cleanup + report ──
    log.info("[4/4] Stopping camera...")
    await cam.stop()

    log.info("")
    log.info("=" * 60)
    log.info("Results")
    log.info("=" * 60)

    fp = stats["frames_processed"]
    log.info(f"Frames processed:        {fp}")
    log.info(f"Effective FPS:           {fp / elapsed:.1f}")

    if fp > 0:
        any_pct = 100 * stats["frames_with_any_marker"] / fp
        target_pct = 100 * stats["frames_with_target"] / fp
        log.info(f"Frames w/ ANY marker:    {stats['frames_with_any_marker']:4d}  ({any_pct:.1f}%)")
        log.info(
            f"Frames w/ target ID={config.target_marker_id}: "
            f"{stats['frames_with_target']:4d}  ({target_pct:.1f}%)"
        )

    if stats["detection_durations_ms"]:
        durs = stats["detection_durations_ms"]
        log.info(f"Detect time mean/p95/max: "
                 f"{np.mean(durs):.1f} / {np.percentile(durs, 95):.1f} / {max(durs):.1f}  ms")

    if stats["centers_x"]:
        cx_arr = np.array(stats["centers_x"])
        cy_arr = np.array(stats["centers_y"])
        img_cx = config.camera_resolution[0] / 2
        img_cy = config.camera_resolution[1] / 2
        offset_x = cx_arr.mean() - img_cx
        offset_y = cy_arr.mean() - img_cy
        log.info(
            f"Marker center mean:      ({cx_arr.mean():.1f}, {cy_arr.mean():.1f})  "
            f"offset from image center: ({offset_x:+.1f}, {offset_y:+.1f}) px"
        )
        log.info(
            f"Marker center stddev:    ({cx_arr.std():.1f}, {cy_arr.std():.1f}) px  "
            f"(low = stable detection)"
        )

    log.info("")
    log.info(f"Saved {len(saved_frames)} annotated frames to {OUTPUT_DIR}")
    for f in saved_frames:
        log.info(f"  {f}")

    log.info("=" * 60)
    log.info("Phase C complete ✓")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.warning("Interrupted by user")