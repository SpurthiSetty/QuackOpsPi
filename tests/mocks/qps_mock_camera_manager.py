import numpy

from src.config.qps_config import qpsConfig
from src.interfaces.qps_i_camera_manager import qpsICameraManager


class qpsMockCameraManager(qpsICameraManager):
    """Mock camera manager that serves pre-loaded test frames.

    Cycles through a list of frames loaded via load_test_frames().
    """

    def __init__(self, config: qpsConfig) -> None:
        """Initialise the mock camera manager.

        Args:
            config: Application configuration (stored but largely unused).
        """
        self.config: qpsConfig = config
        self.frames: list[numpy.ndarray] = []
        self.frame_index: int = 0
        self.running: bool = False

    def start(self) -> bool:
        """Simulate starting the camera.

        Returns:
            bool: True if frames have been loaded.
        """
        self.running = True
        self.frame_index = 0
        return True

    def stop(self) -> None:
        """Simulate stopping the camera."""
        self.running = False

    def get_frame(self) -> numpy.ndarray:
        """Return the next pre-loaded frame, cycling through the list.

        Returns:
            numpy.ndarray: The next test frame, or an empty array if none
                are loaded.
        """
        if not self.frames:
            return numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        frame = self.frames[self.frame_index % len(self.frames)]
        self.frame_index += 1
        return frame

    def is_running(self) -> bool:
        """Check whether the mock camera is running.

        Returns:
            bool: Current running state.
        """
        return self.running

    # ------------------------------------------------------------------
    # Test-helper methods
    # ------------------------------------------------------------------

    def load_test_frames(self, frames: list[numpy.ndarray]) -> None:
        """Load a list of frames to serve via get_frame().

        Args:
            frames: Ordered list of BGR image arrays.
        """
        self.frames = list(frames)
        self.frame_index = 0
