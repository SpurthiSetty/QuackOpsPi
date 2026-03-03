from abc import ABC, abstractmethod

import numpy


class qpsICameraManager(ABC):
    """Abstract interface for camera frame acquisition.

    Implementations may use PiCamera2 (production on Raspberry Pi) or
    OpenCV VideoCapture (simulation / desktop development).
    """

    @abstractmethod
    def start(self) -> bool:
        """Start the camera and begin capturing frames.

        Returns:
            bool: True if the camera was started successfully.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera and release resources."""
        ...

    @abstractmethod
    def get_frame(self) -> numpy.ndarray:
        """Return the most recent captured frame.

        Returns:
            numpy.ndarray: The latest BGR image frame.
        """
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Check whether the camera is currently capturing.

        Returns:
            bool: True if the camera capture loop is active.
        """
        ...
