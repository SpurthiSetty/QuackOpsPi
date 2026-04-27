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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("qps.camera_daemon")


# ── VideoWriter helper ────────────────────────────────────────────────────────

def _open_video_writer(
    output_dir: Path, width: int, height: int, fps: int
) -> tuple[cv2.VideoWriter, str]:
    """Try mp4v codec first; fall back to MJPG .avi on failure."""
    mp4_path = str(output_dir / "video.mp4")
    writer = cv2.VideoWriter(
        mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if writer.isOpened():
        log.info("VideoWriter: mp4v → %s", mp4_path)
        return writer, mp4_path

    writer.release()
    avi_path = str(output_dir / "video.avi")
    writer = cv2.VideoWriter(
        avi_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open VideoWriter with mp4v or MJPG codec")
    log.warning("mp4v codec unavailable — falling back to MJPG AVI: %s", avi_path)
    return writer, avi_path


# ── Recording sink ────────────────────────────────────────────────────────────

class RecordingSink:
    """Frame sink that writes video, detections, and timestamps to disk.

    Invoked by qpsStreamServer's producer thread on each produced frame.
    All I/O is protected by a lock and done synchronously — must stay fast
    (< 10 ms) to avoid stalling stream fps.
    """

    def __init__(
        self,
        writer: cv2.VideoWriter,
        det_csv: csv.writer,
        det_file,
        ts_csv: csv.writer,
        ts_file,
    ) -> None:
        self._writer = writer
        self._det_csv = det_csv
        self._det_file = det_file
        self._ts_csv = ts_csv
        self._ts_file = ts_file
        self._lock = threading.Lock()
        self.frame_count = 0

    def __call__(self, annotated_frame: np.ndarray, detections: list) -> None:
        with self._lock:
            try:
                ts = time.time()
                self._writer.write(annotated_frame)
                self._ts_csv.writerow([self.frame_count, f"{ts:.6f}"])

                if detections:
                    for det in detections:
                        corners_flat = det.corners.flatten().tolist()
                        self._det_csv.writerow([
                            f"{ts:.6f}", det.marker_id,
                            f"{det.center_px[0]:.1f}", f"{det.center_px[1]:.1f}",
                            *[f"{v:.1f}" for v in corners_flat],
                        ])
                    self._det_file.flush()

                self._ts_file.flush()
                self.frame_count += 1
            except Exception as exc:
                log.warning("Recording sink write error: %s", exc)


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

    # Recording fps = stream_fps (the rate at which the sink fires)
    writer, video_path = _open_video_writer(
        output_dir, width, height, config.stream_fps
    )

    det_file = open(output_dir / "detections.csv", "w", newline="")
    det_csv = csv.writer(det_file)
    det_csv.writerow([
        "timestamp", "marker_id", "center_x", "center_y",
        "corner1_x", "corner1_y", "corner2_x", "corner2_y",
        "corner3_x", "corner3_y", "corner4_x", "corner4_y",
    ])

    ts_file = open(output_dir / "frame_timestamps.csv", "w", newline="")
    ts_csv = csv.writer(ts_file)
    ts_csv.writerow(["frame_number", "unix_timestamp"])

    sink = RecordingSink(writer, det_csv, det_file, ts_csv, ts_file)
    stream.register_frame_sink(sink)

    if not camera.start():
        log.error("Failed to start camera")
        writer.release()
        det_file.close()
        ts_file.close()
        return 1

    stream.start(port=config.stream_port)
    log.info("Recording annotated frames → %s", video_path)
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
        camera.stop()
        log.info("Closing output files (%d frames written)...", sink.frame_count)
        writer.release()
        det_file.close()
        ts_file.close()
        log.info("Camera daemon stopped cleanly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
