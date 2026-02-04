"""
QuackOps Raspberry Pi Computer Vision Module

This module provides the object-oriented architecture for the QuackOps drone
delivery system's companion computer. It handles:
- ArUco marker detection for precision landing
- MAVSDK flight control commands
- Camera management for visual processing
- Coordination of vision-based landing sequences
- Communication with the web backend

Authors: QuackOps Senior Design Team (Group 8)
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "QuackOps Team"

from .core.pi_cv_module import PiCvModule
from .core.config import Config
from .vision.marker_detector import MarkerDetector
from .flight.flight_controller import FlightController
from .communication.backend_client import BackendClient

__all__ = [
    "PiCvModule",
    "Config",
    "MarkerDetector",
    "FlightController",
    "BackendClient",
]
