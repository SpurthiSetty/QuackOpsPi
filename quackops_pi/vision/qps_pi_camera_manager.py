import logging
from threading import Thread, Lock

import cv2
import numpy

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_camera_manager_interface import qpsCameraManagerInterface

logger = logging.getLogger(__name__)


class qpsPiCameraManager(qpsCameraManagerInterface):
    """Camera manager for the Raspberry Pi Camera Module via PiCamera2.

    Runs a background capture thread to continuously grab frames from
    the PiCamera2 hardware interface.  The latest frame is available via
    get_frame() in a thread-safe manner.
    """

    def __init__(self, config: qpsConfig) -> None:
        """Initialise the PiCamera2-based camera manager.

        Args:
            config: Application configuration with camera resolution and FPS.
        """
        self.config: qpsConfig = config
        self.camera = None  # Will be a picamera2.Picamera2 instance
        self.latest_frame: numpy.ndarray | None = None
        self.running: bool = False
        self.capture_thread: Thread | None = None
        self.frame_lock: Lock = Lock()

    def start(self) -> bool:
        """Start the PiCamera2 and launch the background capture thread.

        Returns:
            bool: True if the camera was started successfully.
        """
        try:
            from picamera2 import Picamera2

            self.camera = Picamera2()
            width, height = self.config.camera_resolution
            camera_config = self.camera.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameRate": self.config.camera_fps},
            )
            self.camera.configure(camera_config)
            self.camera.start()

            self.running = True
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            logger.info("PiCamera2 started at %dx%d @ %d fps", width, height, self.config.camera_fps)
            return True
        except Exception as e:
            logger.error("Failed to start PiCamera2: %s", e)
            return False

    def stop(self) -> None:
        """Stop the camera and join the capture thread."""
        self.running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None
        logger.info("PiCamera2 stopped")

    def get_frame(self) -> numpy.ndarray | None:
        """Return the most recently captured frame in a thread-safe manner.

        Returns:
            numpy.ndarray: The latest BGR image frame, or None if no frame yet.
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def is_running(self) -> bool:
        """Check whether the camera capture loop is active.

        Returns:
            bool: True if the capture thread is running.
        """
        return self.running

    def _capture_loop(self) -> None:
        """Background thread loop that continuously captures frames.

        Reads RGB frames from PiCamera2, converts to BGR for OpenCV
        consistency, and stores them behind frame_lock.
        """
        while self.running:
            try:
                frame_rgb = self.camera.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                with self.frame_lock:
                    self.latest_frame = frame_bgr
            except Exception as e:
                logger.error("Frame capture error: %s", e)
                self.running = False
