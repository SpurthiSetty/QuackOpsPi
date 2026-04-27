"""
qps_stream_server.py

MJPEG HTTP server that streams annotated camera frames to a browser.

Architecture
------------
- One producer thread grabs frames, runs detection, annotates, caches the
  latest annotated frame, and fires all registered sinks. Runs at
  config.stream_fps regardless of HTTP client connections.
- HTTP handlers are pure consumers — they read the cached latest frame,
  JPEG-encode, and push to their socket. No detection work in the handler.
- Frame sinks (register_frame_sink) fire once per produced frame. Used by
  the recording daemon to capture MP4 + CSV.
- Connection-limited: only MAX_MJPEG_CLIENTS concurrent viewers.

Endpoints
---------
- GET /           → HTML page that embeds the stream
- GET /stream     → MJPEG multipart stream (preferred)
- GET /video      → MJPEG multipart stream (legacy alias for /stream)
"""

from __future__ import annotations

import asyncio
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from threading import Event, Lock, Thread
from typing import Callable, Optional

import cv2
import numpy as np

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_camera_manager_interface import qpsCameraManagerInterface
from quackops_pi.vision.qps_marker_detector_interface import qpsMarkerDetectorInterface

logger = logging.getLogger("qps.stream_server")


# Sink signature: (annotated_frame, detections) -> None
FrameSink = Callable[[np.ndarray, list], None]

MAX_MJPEG_CLIENTS: int = 4


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that serves each client in its own thread."""
    allow_reuse_address = True
    daemon_threads = True


class _MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MJPEG streaming."""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # suppress per-request logging at stream fps

    def do_GET(self) -> None:
        if self.path == "/":
            self._serve_index()
        elif self.path in ("/stream", "/video"):
            self._serve_stream()
        else:
            self.send_error(404)

    def _serve_index(self) -> None:
        html = b"""<html><body style="margin:0;background:#111">
        <img src="/stream" style="width:100%;height:auto">
        </body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html)

    def _serve_stream(self) -> None:
        stream_server: qpsStreamServer = self.server.stream_server

        if not stream_server._acquire_client_slot():
            self.send_error(503, f"Too many MJPEG clients (max {MAX_MJPEG_CLIENTS})")
            return

        try:
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()

            fps = getattr(stream_server._config, "stream_fps", 15)
            interval = 1.0 / fps

            while stream_server._running:
                try:
                    frame = stream_server.get_latest_annotated()
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
        finally:
            stream_server._release_client_slot()


class qpsStreamServer:
    """MJPEG HTTP server with live ArUco detection overlay.

    Runs a producer thread that continuously grabs, detects, and annotates
    frames — independent of HTTP client connections. Access the browser
    stream at http://<host>:<port>/ (or /stream for raw multipart).

    External consumers can register frame sinks via register_frame_sink()
    to tap the same pipeline (e.g. for disk recording).
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

        # HTTP server
        self._server: Optional[_ThreadingHTTPServer] = None
        self._http_thread: Optional[Thread] = None

        # Producer thread
        self._producer_thread: Optional[Thread] = None
        self._stop_event: Event = Event()
        self._running: bool = False

        # Latest annotated frame cache (thread-safe)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: list = []
        self._frame_lock: Lock = Lock()

        # Frame sinks (external consumers — e.g. MP4 recorder)
        self._sinks: list[FrameSink] = []
        self._sinks_lock: Lock = Lock()

        # Client slot tracking for connection limiting
        self._client_count: int = 0
        self._client_count_lock: Lock = Lock()

        # Single reusable asyncio event loop for async camera/detector impls
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Public API ────────────────────────────────────────────────────

    def start(self, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        """Start the producer thread and the HTTP server."""
        if port is None:
            port = getattr(self._config, "stream_port", 8080)

        self._running = True
        self._stop_event.clear()

        # Start producer BEFORE HTTP server so clients immediately find frames
        self._producer_thread = Thread(
            target=self._producer_loop,
            name="qps-stream-producer",
            daemon=True,
        )
        self._producer_thread.start()

        server = _ThreadingHTTPServer((host, port), _MJPEGHandler)
        server.stream_server = self
        self._server = server
        self._http_thread = Thread(
            target=server.serve_forever,
            name="qps-stream-http",
            daemon=True,
        )
        self._http_thread.start()
        logger.info("Stream server started at http://%s:%d", host, port)

    def stop(self) -> None:
        """Stop the producer thread, shutdown HTTP server, join both threads."""
        self._running = False
        self._stop_event.set()

        if self._server is not None:
            self._server.shutdown()
        if self._http_thread is not None:
            self._http_thread.join(timeout=3.0)

        if self._producer_thread is not None:
            self._producer_thread.join(timeout=3.0)

        if self._async_loop is not None:
            try:
                self._async_loop.close()
            except Exception:
                pass
            self._async_loop = None

        logger.info("Stream server stopped")

    def register_frame_sink(self, sink: FrameSink) -> None:
        """Register a callback invoked with (annotated_frame, detections) each
        time the producer thread produces a new frame.

        Sinks execute synchronously in the producer loop — keep them fast
        (< 10 ms) to avoid stalling the stream fps. Exceptions are logged
        and swallowed to protect the pipeline.
        """
        with self._sinks_lock:
            self._sinks.append(sink)
        logger.info("Frame sink registered (total: %d)", len(self._sinks))

    def unregister_frame_sink(self, sink: FrameSink) -> None:
        with self._sinks_lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def get_latest_annotated(self) -> Optional[np.ndarray]:
        """Return a copy of the most recently produced annotated frame, or None."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest_detections(self) -> list:
        """Return the detections from the most recent produced frame."""
        with self._frame_lock:
            return list(self._latest_detections)

    # ── Client slot tracking ──────────────────────────────────────────

    def _acquire_client_slot(self) -> bool:
        with self._client_count_lock:
            if self._client_count >= MAX_MJPEG_CLIENTS:
                return False
            self._client_count += 1
            return True

    def _release_client_slot(self) -> None:
        with self._client_count_lock:
            self._client_count = max(0, self._client_count - 1)

    # ── Producer loop ─────────────────────────────────────────────────

    def _producer_loop(self) -> None:
        """Background thread: grab → detect → annotate → cache → fan-out."""
        fps = getattr(self._config, "stream_fps", 15)
        interval = 1.0 / fps
        logger.info("Producer loop running at %d fps", fps)

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            try:
                frame = self._grab_frame()
                if frame is None:
                    self._stop_event.wait(interval)
                    continue

                detections = self._run_detection(frame)
                annotated = self._annotate_frame(frame, detections)

                with self._frame_lock:
                    self._latest_frame = annotated
                    self._latest_detections = detections

                # Copy sink list under lock; call sinks without holding it
                with self._sinks_lock:
                    sinks_snapshot = list(self._sinks)
                for sink in sinks_snapshot:
                    try:
                        sink(annotated, detections)
                    except Exception as exc:
                        logger.warning("Frame sink error: %s", exc)

            except Exception as exc:
                logger.warning("Producer loop error: %s", exc)

            elapsed = time.monotonic() - loop_start
            sleep_for = interval - elapsed
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)

        logger.info("Producer loop exited")

    # ── Frame pipeline ────────────────────────────────────────────────

    def _grab_frame(self) -> Optional[np.ndarray]:
        result = self._camera.get_frame()
        if asyncio.iscoroutine(result):
            return self._run_coro(result)
        return result

    def _run_detection(self, frame: np.ndarray) -> list:
        result = self._detector.detect(frame)
        if asyncio.iscoroutine(result):
            return self._run_coro(result)
        return result

    def _run_coro(self, coro):
        """Run a coroutine from the producer thread using a reusable event loop."""
        if self._async_loop is None:
            self._async_loop = asyncio.new_event_loop()
        return self._async_loop.run_until_complete(coro)

    def _annotate_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw detection overlay and status annotations onto a copy of frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        frame_center = (w // 2, h // 2)

        # Frame center crosshair (white, always visible)
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

        for det in detections:
            corners_int = det.corners.astype(int)

            cv2.polylines(annotated, [corners_int], True, (0, 255, 0), 2)

            center = (int(det.center_px[0]), int(det.center_px[1]))
            cv2.circle(annotated, center, 5, (0, 0, 255), -1)
            cv2.line(annotated, frame_center, center, (0, 255, 255), 1)

            label_pos = (corners_int[0][0], corners_int[0][1] - 10)
            cv2.putText(
                annotated, f"ID:{det.marker_id}", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            conf_pos = (corners_int[0][0], corners_int[0][1] - 30)
            cv2.putText(
                annotated, f"conf:{det.confidence:.2f}", conf_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

            offset_x = int(det.center_px[0]) - frame_center[0]
            offset_y = int(det.center_px[1]) - frame_center[1]
            cv2.putText(
                annotated, f"offset:({offset_x},{offset_y})",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

            if det.distance_m is not None:
                dist_pos = (corners_int[0][0], corners_int[0][1] - 50)
                cv2.putText(
                    annotated, f"dist:{det.distance_m:.2f}m", dist_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
                )

        cv2.putText(
            annotated, f"Markers: {len(detections)}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        return annotated
