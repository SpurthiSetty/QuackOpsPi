#!/usr/bin/env python3
"""
QuackOps — Phase B: qpsPiCameraManager Hardware Bring-up

Validates the production qpsPiCameraManager class against real Pi Camera
Module 3 hardware (cam0, the bottom-facing wide-angle camera used for
landing).

Test flow:
  1. Construct qpsConfig with camera settings (640×480 @ 30fps from defaults)
  2. Instantiate qpsPiCameraManager
  3. Start camera + verify capture thread is running
  4. Wait for first frame (cold-start sanity check)
  5. Capture 30 frames over ~3 seconds, measure actual FPS
  6. Save first/middle/last frames to disk for visual verification
  7. Clean shutdown

Run on the Pi from project root:
    cd ~/SeniorD/QuackOpsPi
    source venv/bin/activate
    python3 cameratests/test_phase_b_pi_camera_manager.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).parent.parent))

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_pi_camera_manager import qpsPiCameraManager


# ── Config ────────────────────────────────────────────────────────────
NUM_FRAMES_TO_CAPTURE = 30
FIRST_FRAME_TIMEOUT_S = 5.0
OUTPUT_DIR = Path(__file__).parent / "captures"


# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qps.phase_b")


async def main():
    log.info("=" * 60)
    log.info("QuackOps — Phase B: qpsPiCameraManager Bring-up")
    log.info("=" * 60)

    # ── Step 1: Setup ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {OUTPUT_DIR}")

    config = qpsConfig()
    log.info(
        f"Camera config: {config.camera_resolution[0]}x"
        f"{config.camera_resolution[1]} @ {config.camera_fps}fps"
    )

    cam = qpsPiCameraManager(config)

    # ── Step 2: Start camera ──
    log.info("[1/5] Starting camera...")
    await cam.start()
    log.info("  ✓ Camera started")

    if not cam.is_running():
        log.error("✗ Camera reports start succeeded but is_running() is False")
        await cam.stop()
        sys.exit(1)
    log.info("  ✓ Capture thread running")

    try:
        # ── Step 3: Wait for first frame ──
        log.info("[2/5] Waiting for first frame...")
        t_start_wait = time.time()
        first_frame = None
        while time.time() - t_start_wait < FIRST_FRAME_TIMEOUT_S:
            frame = await cam.get_frame()
            if frame is not None:
                first_frame = frame
                break
            await asyncio.sleep(0.05)

        if first_frame is None:
            log.error(
                f"✗ No frame received within {FIRST_FRAME_TIMEOUT_S}s. "
                "Camera started but capture thread is not producing frames."
            )
            await cam.stop()
            sys.exit(1)

        cold_start_s = time.time() - t_start_wait
        log.info(
            f"  ✓ First frame received in {cold_start_s:.2f}s "
            f"shape={first_frame.shape}, dtype={first_frame.dtype}"
        )

        # Sanity-check the frame has variation, not just black/uniform
        std = float(first_frame.std())
        if std < 1.0:
            log.warning(
                f"  ⚠ First frame std={std:.2f} — image appears uniform. "
                "Check lens cap and lighting."
            )
        else:
            log.info(f"  ✓ Frame has visual content (std={std:.1f})")

        # ── Step 4: FPS measurement ──
        log.info(f"[3/5] Capturing {NUM_FRAMES_TO_CAPTURE} frames for FPS test...")
        frames_captured = []
        timestamps = []
        t_capture_start = time.time()

        last_frame_id = id(first_frame)
        attempts = 0
        max_attempts = NUM_FRAMES_TO_CAPTURE * 10  # safety bound

        while len(frames_captured) < NUM_FRAMES_TO_CAPTURE and attempts < max_attempts:
            frame = await cam.get_frame()
            attempts += 1

            if frame is not None:
                frames_captured.append(frame)
                timestamps.append(time.time())
            await asyncio.sleep(1.0 / config.camera_fps / 2)  # poll at 2x target fps

        t_capture_end = time.time()
        elapsed = t_capture_end - t_capture_start
        actual_fps = len(frames_captured) / elapsed if elapsed > 0 else 0

        log.info(f"  ✓ Captured {len(frames_captured)} frames in {elapsed:.2f}s")
        log.info(f"  ✓ Effective polling rate: {actual_fps:.1f} fps")
        log.info(
            f"    (Note: this measures get_frame() polling rate, not the "
            f"underlying picamera2 capture rate)"
        )

        if len(frames_captured) < NUM_FRAMES_TO_CAPTURE:
            log.warning(
                f"  ⚠ Only got {len(frames_captured)}/{NUM_FRAMES_TO_CAPTURE} "
                "frames. Capture thread may be stalling."
            )

        # ── Step 5: Save sample frames for visual inspection ──
        log.info("[4/5] Saving sample frames for visual inspection...")
        if len(frames_captured) > 0:
            samples = {
                "first": frames_captured[0],
                "middle": frames_captured[len(frames_captured) // 2],
                "last": frames_captured[-1],
            }
            for label, frame in samples.items():
                path = OUTPUT_DIR / f"phase_b_{label}.jpg"
                cv2.imwrite(str(path), frame)
                log.info(f"  ✓ Saved {path.name} ({frame.shape[1]}×{frame.shape[0]})")
        else:
            log.warning("  ⚠ No frames to save")

        # ── Step 6: Stop camera cleanly ──
        log.info("[5/5] Stopping camera...")
        await cam.stop()
        if cam.is_running():
            log.warning("  ⚠ Camera reports still running after stop()")
        else:
            log.info("  ✓ Camera stopped cleanly")

    except KeyboardInterrupt:
        log.warning("Interrupted by user — stopping camera")
        await cam.stop()
        sys.exit(130)
    except Exception:
        log.exception("Unexpected error — stopping camera")
        await cam.stop()
        sys.exit(1)

    log.info("=" * 60)
    log.info("Phase B complete ✓")
    log.info(f"Sample frames in: {OUTPUT_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())