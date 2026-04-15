"""
qps_stream_server.py

MJPEG HTTP server that reads frames from a camera manager, runs ArUco
detection, draws overlay annotations, and streams the result to a browser.

The server runs Python's built-in HTTPServer in a background daemon thread.
All work inside the handler is synchronous — no asyncio event loop required.
"""

from __future__ import annotations

import asyncio
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Optional

import cv2
import numpy as np

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_camera_manager_interface import qpsCameraManagerInterface
from quackops_pi.vision.qps_marker_detector_interface import qpsMarkerDetectorInterface

logger = logging.getLogger("qps.stream_server")


class _MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MJPEG streaming."""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # suppress per-request logging at stream fps

    def do_GET(self) -> None:
        if self.path == "/":
            self._serve_index()
        elif self.path == "/video":
            self._serve_stream()
        else:
            self.send_error(404)

    def _serve_index(self) -> None:
        html = b"""<html><body style="margin:0;background:#111">
        <img src="/video" style="width:100%;height:auto">
        </body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()

        stream_server: qpsStreamServer = self.server.stream_server
        fps = getattr(stream_server._config, "stream_fps", 15)
        interval = 1.0 / fps

        while True:
            try:
                frame = stream_server._get_annotated_frame()
                if frame is None:
                    time.sleep(interval)
                    continue

                ret, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if not ret:
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b"\r\n")
                time.sleep(interval)

            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception as exc:
                logger.warning("Stream error: %s", exc)
                break


class qpsStreamServer:
    """MJPEG HTTP server with live ArUco detection overlay.

    Runs in a background daemon thread. Access the stream at
    http://<host>:<port>/ in any browser.
    """

    def __init__(
        self,
        camera_manager: qpsCameraManagerInterface,
        marker_detector: qpsMarkerDetectorInterface,
        config: qpsConfig,
    ) -> None:
        self._camera = camera_manager
        self._detector = marker_detector
        self._config = config
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[Thread] = None
        self._running: bool = False

    # ── Public API ────────────────────────────────────────────────────

    def start(self, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        """Start the MJPEG server in a background daemon thread."""
        if port is None:
            port = getattr(self._config, "stream_port", 8080)

        server = HTTPServer((host, port), _MJPEGHandler)
        server.stream_server = self  # give handler access to this instance

        self._server = server
        self._running = True
        self._thread = Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Stream server started at http://%s:%d", host, port)

    def stop(self) -> None:
        """Shutdown the server and join the background thread."""
        if self._server is not None:
            self._server.shutdown()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        logger.info("Stream server stopped")

    # ── Frame pipeline ────────────────────────────────────────────────

    def _get_annotated_frame(self) -> Optional[np.ndarray]:
        """Grab a frame, run detection, and return the annotated result."""
        frame = self._grab_frame()
        if frame is None:
            return None
        detections = self._run_detection(frame)
        return self._annotate_frame(frame, detections)

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Get a frame from the camera manager, handling sync or async impls."""
        result = self._camera.get_frame()
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(result)
            finally:
                loop.close()
        return result

    def _run_detection(self, frame: np.ndarray) -> list:
        """Run ArUco detection, handling sync or async implementations."""
        result = self._detector.detect(frame)
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(result)
            finally:
                loop.close()
        return result

    def _annotate_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw detection overlay and status annotations onto a copy of frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        frame_center = (w // 2, h // 2)

        # 1. Frame center crosshair (white, always visible)
        cross_size = 20
        cv2.line(
            annotated,
            (frame_center[0] - cross_size, frame_center[1]),
            (frame_center[0] + cross_size, frame_center[1]),
            (255, 255, 255), 1,
        )
        cv2.line(
            annotated,
            (frame_center[0], frame_center[1] - cross_size),
            (frame_center[0], frame_center[1] + cross_size),
            (255, 255, 255), 1,
        )

        # 2. Per-marker annotations
        for det in detections:
            corners_int = det.corners.astype(int)

            # Green bounding polygon
            cv2.polylines(annotated, [corners_int], True, (0, 255, 0), 2)

            # Red center dot
            center = (int(det.center_px[0]), int(det.center_px[1]))
            cv2.circle(annotated, center, 5, (0, 0, 255), -1)

            # Yellow line from frame center to marker center
            cv2.line(annotated, frame_center, center, (0, 255, 255), 1)

            # Marker ID label (above top-left corner)
            label_pos = (corners_int[0][0], corners_int[0][1] - 10)
            cv2.putText(
                annotated, f"ID:{det.marker_id}", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            # Confidence
            conf_pos = (corners_int[0][0], corners_int[0][1] - 30)
            cv2.putText(
                annotated, f"conf:{det.confidence:.2f}", conf_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

            # Pixel offset from frame center (bottom-left of frame)
            offset_x = int(det.center_px[0]) - frame_center[0]
            offset_y = int(det.center_px[1]) - frame_center[1]
            cv2.putText(
                annotated, f"offset:({offset_x},{offset_y})",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

            # Distance if available
            if det.distance_m is not None:
                dist_pos = (corners_int[0][0], corners_int[0][1] - 50)
                cv2.putText(
                    annotated, f"dist:{det.distance_m:.2f}m", dist_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
                )

        # 3. Marker count status (top-left)
        cv2.putText(
            annotated, f"Markers: {len(detections)}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        return annotated
