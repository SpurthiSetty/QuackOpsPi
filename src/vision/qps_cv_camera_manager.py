from threading import Thread, Lock

import cv2
import numpy

from src.config.qps_config import qpsConfig
from src.interfaces.qps_i_camera_manager import qpsICameraManager


class qpsCVCameraManager(qpsICameraManager):
    """Camera manager using OpenCV VideoCapture for simulation / desktop.

    Runs a background capture thread to continuously grab frames from an
    OpenCV-compatible camera device (USB webcam or virtual camera).  The
    latest frame is available via get_frame() in a thread-safe manner.
    """

    def __init__(self, config: qpsConfig) -> None:
        """Initialise the OpenCV-based camera manager.

        Args:
            config: Application configuration with camera ID, resolution,
                and FPS settings.
        """
        self.config: qpsConfig = config
        self.capture: cv2.VideoCapture | None = None
        self.latest_frame: numpy.ndarray | None = None
        self.running: bool = False
        self.capture_thread: Thread | None = None
        self.frame_lock: Lock = Lock()

    def start(self) -> bool:
        """Open the OpenCV camera and launch the background capture thread.

        Returns:
            bool: True if the camera was opened and started successfully.
        """
        # TODO: Create cv2.VideoCapture(self.config.bottom_camera_id).
        #  Set resolution and FPS properties. Verify isOpened().
        #  Launch self._capture_loop in a daemon thread.
        pass

    def stop(self) -> None:
        """Stop the camera and release OpenCV resources."""
        # TODO: Set self.running = False. Join capture_thread. Release
        #  self.capture.
        pass

    def get_frame(self) -> numpy.ndarray:
        """Return the most recently captured frame in a thread-safe manner.

        Returns:
            numpy.ndarray: The latest BGR image frame.
        """
        # TODO: Acquire self.frame_lock and return a copy of self.latest_frame.
        pass

    def is_running(self) -> bool:
        """Check whether the camera capture loop is active.

        Returns:
            bool: True if the capture thread is running.
        """
        # TODO: Return self.running.
        pass

    def _capture_loop(self) -> None:
        """Background thread loop that continuously captures frames.

        Reads frames from OpenCV VideoCapture and stores them behind
        frame_lock.
        """
        # TODO: While self.running, call self.capture.read(). On success,
        #  acquire self.frame_lock and update self.latest_frame.
        pass
