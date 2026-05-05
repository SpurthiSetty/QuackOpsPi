"""
outdoor_tests/camera_daemon.py

Camera recording daemon: delegates capture + detection + annotation + MJPEG
serving to qpsStreamServer, and records the annotated frames to disk via a
frame sink.

Architecture
------------
qpsStreamServer owns the producer thread that grabs frames from the camera,
runs ArUco detection, annotates, caches the latest frame, serves MJPEG over
HTTP, AND fires registered frame sinks once per produced frame.

This daemon registers a single frame sink that writes:
  - video.mp4 (or video.avi fallback)        — annotated frames at stream_fps
  - detections.csv                           — one row per detected marker
  - frame_timestamps.csv                     — frame_number → unix_timestamp

Usage
-----
    python3 outdoor_tests/camera_daemon.py --output-dir /path/to/logs \
        [--fps 30] [--resolution 1280x720] [--mjpeg-port 8080] \
        [--marker-dict DICT_4X4_50]

Stream URL: http://<pi-ip>:8080/
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import signal
import sys
import threading
import time
from pathlib import Path

# ── Repo-root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_pi_camera_manager import qpsPiCameraManager
from quackops_pi.vision.qps_marker_detector import qpsMarkerDetector
from quackops_pi.vision.qps_stream_server import qpsStreamServer
from test_common import open_recording_sink

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("qps.camera_daemon")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuackOps camera recording daemon")
    parser.add_argument("--output-dir", required=True, help="Directory for video + CSV output")
    parser.add_argument("--fps", type=int, default=None,
                        help="Capture FPS (default: from config)")
    parser.add_argument("--resolution", type=str, default=None,
                        help="Resolution as WxH e.g. 1280x720 (default: from config)")
    parser.add_argument("--mjpeg-port", type=int, default=None,
                        help="MJPEG HTTP server port (default: from config)")
    parser.add_argument("--marker-dict", type=str, default=None,
                        help="ArUco dictionary name e.g. DICT_4X4_50 (default: from config)")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "config" / "outdoor_production.json"))
    return parser.parse_args()


def _apply_cli_overrides(config: qpsConfig, args: argparse.Namespace) -> None:
    if args.resolution is not None:
        try:
            w, h = args.resolution.lower().split("x")
            config.camera_resolution = (int(w), int(h))
        except ValueError:
            raise SystemExit("Invalid --resolution format (expected WxH, e.g. 1280x720)")
    if args.fps is not None:
        config.camera_fps = args.fps
    if args.mjpeg_port is not None:
        config.stream_port = args.mjpeg_port
    if args.marker_dict is not None:
        config.aruco_dictionary = args.marker_dict


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = qpsConfig.from_file(args.config)
    _apply_cli_overrides(config, args)

    width, height = config.camera_resolution
    log.info(
        "Camera: %dx%d @ %dfps  stream: %dfps @ port %d  markers: %s",
        width, height, config.camera_fps, config.stream_fps,
        config.stream_port, config.aruco_dictionary,
    )

    camera = qpsPiCameraManager(config)
    detector = qpsMarkerDetector(config)
    stream = qpsStreamServer(camera, detector, config)

    sink, writer, det_file, ts_file = open_recording_sink(
        output_dir, width, height, config.stream_fps
    )
    stream.register_frame_sink(sink)

    try:
        asyncio.run(camera.start())
    except Exception as exc:
        log.error("Failed to start camera: %s", exc)
        writer.release()
        det_file.close()
        ts_file.close()
        return 1

    stream.start(port=config.stream_port)
    log.info("Recording annotated frames → %s", output_dir / "video.mp4")
    log.info("Browser: http://0.0.0.0:%d/", config.stream_port)

    shutdown = threading.Event()

    def _handle_signal(signum: int, _frame) -> None:
        log.info("Signal %d received — shutting down", signum)
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        shutdown.wait()
    finally:
        log.info("Stopping stream server...")
        stream.stop()
        log.info("Stopping camera...")
        asyncio.run(camera.stop())
        log.info("Closing output files (%d frames written)...", sink.frame_count)
        writer.release()
        det_file.close()
        ts_file.close()
        log.info("Camera daemon stopped cleanly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
