#!/usr/bin/env python3
"""
Test camera stream with ArUco detection overlay.

Usage on Pi:
    python3 tests/hardware/test_camera_stream.py

Usage on laptop (USB webcam):
    python3 tests/hardware/test_camera_stream.py --opencv

Then open http://<pi-ip>:8080 in your browser.
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_marker_detector import qpsMarkerDetector
from quackops_pi.vision.qps_stream_server import qpsStreamServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera stream with ArUco overlay")
    parser.add_argument(
        "--opencv", action="store_true",
        help="Use OpenCV VideoCapture instead of PiCamera2",
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument(
        "--camera-id", type=int, default=0,
        help="Camera device ID for --opencv mode (default: 0)",
    )
    args = parser.parse_args()

    config = qpsConfig()

    # Select camera implementation
    if args.opencv:
        from quackops_pi.vision.qps_cv_camera_manager import qpsCVCameraManager
        camera = qpsCVCameraManager(config, camera_id=args.camera_id)
        log.info("Using OpenCV camera (device %d)", args.camera_id)
        asyncio.run(camera.start())
    else:
        from quackops_pi.vision.qps_pi_camera_manager import qpsPiCameraManager
        camera = qpsPiCameraManager(config)
        log.info("Using PiCamera2")
        if not camera.start():
            log.error("Failed to start PiCamera2")
            sys.exit(1)

    # Let camera warm up
    time.sleep(1.0)

    detector = qpsMarkerDetector(config)
    stream = qpsStreamServer(camera, detector, config)

    stream.start(port=args.port)
    log.info("=" * 50)
    log.info("Stream running — open in browser:")
    log.info("  http://ssetty.local:%d", args.port)
    log.info("  (or http://localhost:%d for local test)", args.port)
    log.info("Press Ctrl+C to stop")
    log.info("=" * 50)

    def shutdown(sig, frame) -> None:
        log.info("Shutting down...")
        stream.stop()
        if args.opencv:
            asyncio.run(camera.stop())
        else:
            camera.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
