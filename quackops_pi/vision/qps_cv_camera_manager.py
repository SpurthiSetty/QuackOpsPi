"""
qps_cv_camera_manager.py

OpenCV-based camera manager for simulation / desktop testing.

Runs a daemon thread that continuously grabs frames from a VideoCapture device
(laptop webcam, USB camera, or virtual device).  The latest frame is exposed
thread-safely via get_frame().  The async start/stop methods satisfy the
qpsCameraManagerInterface contract while keeping all blocking I/O off the
event loop.
"""

from __future__ import annotations

import logging
from threading import Thread, Lock
from typing import Optional

import cv2
import numpy as np

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_camera_manager_interface import qpsCameraManagerInterface

logger = logging.getLogger("qps.cv_camera_manager")


class qpsCVCameraManager(qpsCameraManagerInterface):
    """Camera manager backed by OpenCV VideoCapture.

    The capture loop runs in a daemon Thread so it never blocks the asyncio
    event loop.  start() / stop() are async to satisfy the interface but do
    not themselves perform any async I/O — they just manage the thread.

    Usage:
        await camera.start()
        frame = await camera.get_frame()   # returns None until first frame
        await camera.stop()
    """

    def __init__(self, config: qpsConfig, camera_id: int = 0) -> None:
        """Initialise the OpenCV camera manager.

        Args:
            config:    Application configuration (camera_resolution, camera_fps).
            camera_id: OpenCV device index (0 = first/default webcam).
        """
        self._config = config
        self._camera_id = camera_id

        self._capture: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock: Lock = Lock()
        self._running: bool = False
        self._thread: Optional[Thread] = None

    # ── qpsCameraManagerInterface ─────────────────────────────────────

    async def start(self) -> None:
        """Open the camera device and launch the background capture thread.

        Raises RuntimeError if the camera cannot be opened.
        """
        if self._running:
            logger.debug("Camera already running — ignoring start()")
            return

        self._capture = cv2.VideoCapture(self._camera_id)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Failed to open camera device {self._camera_id}"
            )

        width, height = self._config.camera_resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, self._config.camera_fps)

        self._running = True
        self._thread = Thread(
            target=self._capture_loop,
            daemon=True,
            name="cv-capture",
        )
        self._thread.start()
        logger.info(
            "Camera started (device=%d  res=%dx%d  fps=%d)",
            self._camera_id, width, height, self._config.camera_fps,
        )

    async def stop(self) -> None:
        """Stop the capture thread and release the camera device."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        logger.info("Camera stopped")

    async def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recently captured frame, or None.

        Thread-safe: acquires frame_lock before reading latest_frame.
        Returns None until the first frame has been captured.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def is_running(self) -> bool:
        """Return True if the capture thread is active."""
        return self._running

    # ── Background capture thread ─────────────────────────────────────

    def _capture_loop(self) -> None:
        """Continuously read frames from VideoCapture and store the latest.

        Runs on the daemon Thread started by start().  Stops when self._running
        is set to False by stop().
        """
        logger.debug("Capture loop started")
        while self._running:
            if self._capture is None:
                break
            ret, frame = self._capture.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                logger.warning("Camera read() returned False — retrying")
        logger.debug("Capture loop ended")
