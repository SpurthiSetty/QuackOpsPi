"""
Utility functions and helpers for the QuackOps Pi CV Module.
"""

import logging
import time
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate the great circle distance between two points.
    
    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees
        
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = (np.sin(delta_phi / 2) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def bearing_between(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate the bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: Starting point coordinates in degrees
        lat2, lon2: Ending point coordinates in degrees
        
    Returns:
        Bearing in degrees (0-360)
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)
    
    x = np.sin(delta_lambda) * np.cos(phi2)
    y = (np.cos(phi1) * np.sin(phi2) -
         np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda))
    
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def destination_point(
    lat: float, lon: float,
    bearing_deg: float,
    distance_m: float
) -> Tuple[float, float]:
    """
    Calculate destination point given start, bearing, and distance.
    
    Args:
        lat, lon: Starting point in degrees
        bearing_deg: Bearing in degrees
        distance_m: Distance in meters
        
    Returns:
        Tuple of (latitude, longitude) in degrees
    """
    R = 6371000  # Earth's radius in meters
    
    phi1 = np.radians(lat)
    lambda1 = np.radians(lon)
    bearing = np.radians(bearing_deg)
    
    phi2 = np.arcsin(
        np.sin(phi1) * np.cos(distance_m / R) +
        np.cos(phi1) * np.sin(distance_m / R) * np.cos(bearing)
    )
    
    lambda2 = lambda1 + np.arctan2(
        np.sin(bearing) * np.sin(distance_m / R) * np.cos(phi1),
        np.cos(distance_m / R) - np.sin(phi1) * np.sin(phi2)
    )
    
    return np.degrees(phi2), np.degrees(lambda2)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * clamp(t, 0, 1)


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """Smooth interpolation function."""
    t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


class Timer:
    """Simple timer utility for measuring elapsed time."""
    
    def __init__(self):
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0
        self._running = False
    
    def start(self) -> None:
        """Start the timer."""
        if not self._running:
            self._start_time = time.time()
            self._running = True
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self._running:
            self._elapsed += time.time() - self._start_time
            self._running = False
        return self._elapsed
    
    def reset(self) -> None:
        """Reset the timer."""
        self._elapsed = 0.0
        self._start_time = None
        self._running = False
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._running:
            return self._elapsed + (time.time() - self._start_time)
        return self._elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class RateLimit:
    """Rate limiter utility."""
    
    def __init__(self, rate_hz: float):
        """
        Initialize rate limiter.
        
        Args:
            rate_hz: Maximum rate in Hz
        """
        self._min_interval = 1.0 / rate_hz
        self._last_time = 0.0
    
    def ready(self) -> bool:
        """Check if enough time has passed for the next event."""
        current = time.time()
        if current - self._last_time >= self._min_interval:
            self._last_time = current
            return True
        return False
    
    async def wait(self) -> None:
        """Wait until rate limit allows next event."""
        import asyncio
        
        current = time.time()
        remaining = self._min_interval - (current - self._last_time)
        
        if remaining > 0:
            await asyncio.sleep(remaining)
        
        self._last_time = time.time()


class MovingAverage:
    """Exponential moving average filter."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize moving average.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = less smoothing.
        """
        self.alpha = alpha
        self._value: Optional[float] = None
    
    def update(self, value: float) -> float:
        """Update with new value and return filtered result."""
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return self._value
    
    @property
    def value(self) -> Optional[float]:
        """Get current filtered value."""
        return self._value
    
    def reset(self) -> None:
        """Reset the filter."""
        self._value = None


def generate_aruco_marker(
    marker_id: int,
    dictionary: str = "DICT_6X6_250",
    size_pixels: int = 200,
    border_bits: int = 1
) -> np.ndarray:
    """
    Generate an ArUco marker image.
    
    Args:
        marker_id: Marker ID to generate
        dictionary: ArUco dictionary name
        size_pixels: Size of output image in pixels
        border_bits: Number of border bits
        
    Returns:
        Marker image as numpy array
    """
    import cv2
    import cv2.aruco as aruco
    
    from ..vision.marker_detector import ARUCO_DICT_MAP
    
    dict_id = ARUCO_DICT_MAP.get(dictionary)
    if dict_id is None:
        raise ValueError(f"Unknown dictionary: {dictionary}")
    
    aruco_dict = aruco.getPredefinedDictionary(dict_id)
    marker_image = aruco.generateImageMarker(
        aruco_dict, marker_id, size_pixels, borderBits=border_bits
    )
    
    return marker_image


def save_aruco_marker(
    marker_id: int,
    output_path: str,
    dictionary: str = "DICT_6X6_250",
    size_pixels: int = 200
) -> None:
    """
    Generate and save an ArUco marker to file.
    
    Args:
        marker_id: Marker ID to generate
        output_path: Path to save the marker image
        dictionary: ArUco dictionary name
        size_pixels: Size of output image in pixels
    """
    import cv2
    
    marker = generate_aruco_marker(marker_id, dictionary, size_pixels)
    cv2.imwrite(output_path, marker)
    logger.info(f"Saved marker {marker_id} to {output_path}")


def create_default_config_file(filepath: str) -> None:
    """
    Create a default configuration file.
    
    Args:
        filepath: Path to save the config file
    """
    from ..core.config import Config
    
    config = Config()
    config.save(filepath)
    logger.info(f"Created default config at {filepath}")
