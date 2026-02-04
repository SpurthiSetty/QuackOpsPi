"""
Camera Manager implementation for managing multiple camera modules.

Supports both PiCamera2 (for Raspberry Pi) and OpenCV VideoCapture
for flexibility across different hardware configurations.
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import Optional, Dict, Tuple
from enum import Enum, auto

from ..core.interfaces import ICameraManager
from ..core.config import CameraConfig

logger = logging.getLogger(__name__)


class CameraType(Enum):
    """Types of cameras supported."""
    FRONT = auto()   # Navigation camera
    BOTTOM = auto()  # Landing/marker detection camera


class CameraManager(ICameraManager):
    """
    Manager for camera modules on the Raspberry Pi.
    
    Supports both the front-facing camera (for navigation) and the
    bottom-facing camera (for ArUco marker detection during landing).
    Can use either PiCamera2 or OpenCV VideoCapture depending on
    configuration.
    
    Attributes:
        config: Camera configuration settings
        cameras: Dictionary of active camera instances
        
    Example:
        >>> manager = CameraManager(config)
        >>> manager.start()
        >>> frame = manager.get_frame(CameraType.BOTTOM)
        >>> manager.stop()
    """
    
    def __init__(self, config: CameraConfig):
        """
        Initialize the camera manager.
        
        Args:
            config: Camera configuration settings
        """
        self.config = config
        self._cameras: Dict[int, any] = {}
        self._frames: Dict[int, Optional[np.ndarray]] = {}
        self._frame_locks: Dict[int, threading.Lock] = {}
        self._capture_threads: Dict[int, Optional[threading.Thread]] = {}
        self._running = False
        
        # Camera calibration data (can be loaded from file)
        self._camera_matrices: Dict[int, np.ndarray] = {}
        self._distortion_coeffs: Dict[int, np.ndarray] = {}
        
        # Initialize default calibration for common camera modules
        self._init_default_calibration()
        
        logger.info(
            f"CameraManager initialized (use_picamera={config.use_picamera}, "
            f"resolution={config.resolution_width}x{config.resolution_height})"
        )
    
    def _init_default_calibration(self) -> None:
        """Initialize default camera calibration parameters."""
        # Default calibration for Raspberry Pi Camera Module 3
        # These are approximate values - should be replaced with actual calibration
        w, h = self.config.resolution_width, self.config.resolution_height
        
        # Approximate focal length based on resolution
        focal_length = w * 0.9  # Rough approximation
        
        # Default camera matrix
        default_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Default distortion coefficients (assuming minimal distortion)
        default_dist = np.zeros((5, 1), dtype=np.float32)
        
        # Set for both cameras
        for cam_id in [self.config.front_camera_id, self.config.bottom_camera_id]:
            self._camera_matrices[cam_id] = default_matrix.copy()
            self._distortion_coeffs[cam_id] = default_dist.copy()
    
    def load_calibration(
        self, 
        camera_id: int, 
        calibration_file: str
    ) -> bool:
        """
        Load camera calibration from a NumPy file.
        
        Args:
            camera_id: The camera to load calibration for
            calibration_file: Path to .npz file with calibration data
            
        Returns:
            True if calibration loaded successfully
        """
        try:
            data = np.load(calibration_file)
            self._camera_matrices[camera_id] = data['camera_matrix']
            self._distortion_coeffs[camera_id] = data['distortion_coeffs']
            logger.info(f"Loaded calibration for camera {camera_id} from {calibration_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start all camera captures.
        
        Returns:
            True if at least one camera started successfully
        """
        if self._running:
            logger.warning("Camera manager already running")
            return True
        
        success = False
        
        # Start front camera
        if self._start_camera(self.config.front_camera_id):
            success = True
            logger.info(f"Front camera {self.config.front_camera_id} started")
        
        # Start bottom camera
        if self._start_camera(self.config.bottom_camera_id):
            success = True
            logger.info(f"Bottom camera {self.config.bottom_camera_id} started")
        
        if success:
            self._running = True
            
        return success
    
    def _start_camera(self, camera_id: int) -> bool:
        """Start a specific camera."""
        if camera_id in self._cameras:
            return True  # Already started
        
        try:
            if self.config.use_picamera:
                return self._start_picamera(camera_id)
            else:
                return self._start_opencv_camera(camera_id)
        except Exception as e:
            logger.error(f"Failed to start camera {camera_id}: {e}")
            return False
    
    def _start_picamera(self, camera_id: int) -> bool:
        """Start camera using PiCamera2."""
        try:
            from picamera2 import Picamera2
            
            picam = Picamera2(camera_id)
            
            config = picam.create_preview_configuration(
                main={
                    "size": (self.config.resolution_width, self.config.resolution_height),
                    "format": "RGB888"
                }
            )
            picam.configure(config)
            picam.start()
            
            self._cameras[camera_id] = picam
            self._frames[camera_id] = None
            self._frame_locks[camera_id] = threading.Lock()
            
            # Start capture thread
            self._start_capture_thread(camera_id)
            
            return True
            
        except ImportError:
            logger.warning("PiCamera2 not available, falling back to OpenCV")
            return self._start_opencv_camera(camera_id)
        except Exception as e:
            logger.error(f"PiCamera2 error: {e}")
            return False
    
    def _start_opencv_camera(self, camera_id: int) -> bool:
        """Start camera using OpenCV VideoCapture."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open OpenCV camera {camera_id}")
            return False
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.framerate)
        
        self._cameras[camera_id] = cap
        self._frames[camera_id] = None
        self._frame_locks[camera_id] = threading.Lock()
        
        # Start capture thread
        self._start_capture_thread(camera_id)
        
        return True
    
    def _start_capture_thread(self, camera_id: int) -> None:
        """Start background thread for continuous frame capture."""
        thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_id,),
            daemon=True
        )
        self._capture_threads[camera_id] = thread
        thread.start()
    
    def _capture_loop(self, camera_id: int) -> None:
        """Background thread that continuously captures frames."""
        camera = self._cameras.get(camera_id)
        if camera is None:
            return
        
        while self._running and camera_id in self._cameras:
            try:
                if self.config.use_picamera:
                    # PiCamera2 capture
                    frame = camera.capture_array()
                    # Convert RGB to BGR for OpenCV compatibility
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    # OpenCV capture
                    ret, frame = camera.read()
                    if not ret:
                        continue
                
                with self._frame_locks[camera_id]:
                    self._frames[camera_id] = frame
                    
            except Exception as e:
                logger.error(f"Capture error on camera {camera_id}: {e}")
                time.sleep(0.1)
    
    def stop(self) -> None:
        """Stop all camera captures."""
        self._running = False
        
        # Wait for capture threads to finish
        for thread in self._capture_threads.values():
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        
        # Release cameras
        for camera_id, camera in self._cameras.items():
            try:
                if self.config.use_picamera:
                    camera.stop()
                else:
                    camera.release()
                logger.info(f"Camera {camera_id} stopped")
            except Exception as e:
                logger.error(f"Error stopping camera {camera_id}: {e}")
        
        self._cameras.clear()
        self._frames.clear()
        self._capture_threads.clear()
    
    def get_frame(self, camera_id: int = 0) -> Optional[np.ndarray]:
        """
        Get the latest frame from the specified camera.
        
        Args:
            camera_id: Camera ID (use config.bottom_camera_id for landing)
            
        Returns:
            BGR image as numpy array, or None if unavailable
        """
        if camera_id not in self._frame_locks:
            return None
        
        with self._frame_locks[camera_id]:
            frame = self._frames.get(camera_id)
            if frame is not None:
                return frame.copy()
            return None
    
    def get_bottom_frame(self) -> Optional[np.ndarray]:
        """Convenience method to get frame from bottom camera."""
        return self.get_frame(self.config.bottom_camera_id)
    
    def get_front_frame(self) -> Optional[np.ndarray]:
        """Convenience method to get frame from front camera."""
        return self.get_frame(self.config.front_camera_id)
    
    def get_camera_matrix(self, camera_id: int = 0) -> np.ndarray:
        """
        Get the camera intrinsic matrix for pose estimation.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            3x3 camera intrinsic matrix
        """
        return self._camera_matrices.get(
            camera_id, 
            self._camera_matrices[self.config.bottom_camera_id]
        )
    
    def get_distortion_coeffs(self, camera_id: int = 0) -> np.ndarray:
        """
        Get the camera distortion coefficients.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Distortion coefficients array
        """
        return self._distortion_coeffs.get(
            camera_id,
            self._distortion_coeffs[self.config.bottom_camera_id]
        )
    
    def set_calibration(
        self,
        camera_id: int,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray
    ) -> None:
        """
        Set camera calibration parameters.
        
        Args:
            camera_id: Camera ID
            camera_matrix: 3x3 intrinsic matrix
            distortion_coeffs: Distortion coefficients
        """
        self._camera_matrices[camera_id] = camera_matrix.copy()
        self._distortion_coeffs[camera_id] = distortion_coeffs.copy()
        logger.info(f"Calibration set for camera {camera_id}")
    
    def save_calibration(self, camera_id: int, filepath: str) -> None:
        """
        Save camera calibration to file.
        
        Args:
            camera_id: Camera ID
            filepath: Path to save .npz file
        """
        np.savez(
            filepath,
            camera_matrix=self._camera_matrices[camera_id],
            distortion_coeffs=self._distortion_coeffs[camera_id]
        )
        logger.info(f"Calibration saved to {filepath}")
    
    @property
    def is_running(self) -> bool:
        """Check if cameras are running."""
        return self._running
    
    @property
    def active_cameras(self) -> list:
        """Get list of active camera IDs."""
        return list(self._cameras.keys())
