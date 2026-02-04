# QuackOps Pi CV Module

Raspberry Pi companion computer module for the QuackOps autonomous drone delivery system. This module handles computer vision processing, flight control, and communication with the web backend.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PiCvModule (Orchestrator)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │  FlightController │  │   CameraManager   │  │   MarkerDetector     │   │
│  │    (MAVSDK)       │  │   (PiCamera2)     │  │    (OpenCV)          │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘   │
│           │                     │                        │               │
│           v                     v                        v               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    LandingCoordinator                              │   │
│  │              (Precision Vision-Based Landing)                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      BackendClient                                 │   │
│  │           (WebSocket + HTTP Communication)                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ MAVLink (UDP)
                                    v
                          ┌─────────────────┐
                          │ Pixhawk 2.4.8   │
                          │ Flight Controller│
                          └─────────────────┘
```

## Module Structure

```
qps_pi_cv_module/
├── __init__.py              # Package exports
├── __main__.py              # Module entry point
├── run.py                   # CLI runner script
├── requirements.txt         # Python dependencies
│
├── core/                    # Core module components
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── interfaces.py        # Abstract base classes & data types
│   ├── pi_cv_module.py      # Main orchestrator
│   └── landing_coordinator.py # Precision landing logic
│
├── vision/                  # Computer vision components
│   ├── __init__.py
│   ├── marker_detector.py   # ArUco marker detection
│   └── camera_manager.py    # Camera management
│
├── flight/                  # Flight control components
│   ├── __init__.py
│   └── flight_controller.py # MAVSDK flight control
│
├── communication/           # Backend communication
│   ├── __init__.py
│   └── backend_client.py    # WebSocket & HTTP clients
│
└── utils/                   # Utility functions
    └── __init__.py          # Helpers & utilities
```

## Key Classes

### PiCvModule
The main orchestrator that coordinates all subsystems. Handles:
- Initialization and shutdown of all components
- Command handling from backend
- Telemetry streaming
- Mission execution

### FlightController
MAVSDK-based flight control interface providing:
- Connection management
- Arm/disarm commands
- Takeoff/land commands
- Position navigation (goto)
- Velocity control for precision landing
- Telemetry streaming

### CameraManager
Manages dual camera setup (front + bottom):
- PiCamera2 support for Raspberry Pi
- OpenCV fallback for development
- Threaded frame capture
- Camera calibration data management

### MarkerDetector
ArUco marker detection using OpenCV:
- Configurable dictionary (DICT_6X6_250 default)
- Pose estimation for 3D position
- Confidence scoring
- Visualization helpers

### LandingCoordinator
Orchestrates precision landing sequence:
1. **Approach** - Descend to approach altitude
2. **Search** - Locate target marker
3. **Acquire** - Lock onto marker
4. **Descend** - Visual tracking during descent
5. **Final Approach** - Slow descent
6. **Touchdown** - Complete landing

### BackendClient
WebSocket/HTTP communication with web backend:
- Real-time telemetry streaming
- Command reception and handling
- Landing status updates
- Automatic reconnection

## Installation

### On Raspberry Pi

```bash
# Clone the repository
git clone https://github.com/your-org/quackops.git
cd quackops/pi-module

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Raspberry Pi Camera support
pip install picamera2
```

### For Development (Non-Pi)

```bash
# Install without picamera2
pip install -r requirements.txt

# Use OpenCV webcam capture instead
# Set use_picamera: false in config
```

## Configuration

### Using Config File

```json
{
  "camera": {
    "front_camera_id": 0,
    "bottom_camera_id": 1,
    "resolution_width": 640,
    "resolution_height": 480,
    "framerate": 30,
    "use_picamera": true
  },
  "vision": {
    "aruco_dictionary": "DICT_6X6_250",
    "marker_size_cm": 10.0,
    "detection_confidence_threshold": 0.8
  },
  "flight": {
    "mavsdk_system_address": "udp://:14540",
    "landing_descent_rate_mps": 0.3
  },
  "communication": {
    "backend_base_url": "http://localhost:8000",
    "websocket_url": "ws://localhost:8000/ws/drone",
    "telemetry_update_interval_sec": 0.5
  },
  "landing": {
    "approach_altitude_m": 5.0,
    "final_descent_altitude_m": 1.0,
    "marker_lock_threshold_pixels": 50
  },
  "log_level": "INFO",
  "debug_mode": false,
  "simulation_mode": false
}
```

### Environment Variables

```bash
export QPS_BACKEND_URL="http://192.168.1.100:8000"
export QPS_MAVSDK_ADDRESS="udp://:14540"
export QPS_SIMULATION_MODE="true"
export QPS_DEBUG_MODE="true"
export QPS_LOG_LEVEL="DEBUG"
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python -m qps_pi_cv_module

# Run with custom config file
python run.py --config /path/to/config.json

# Run in simulation mode (no real hardware)
python run.py --simulation

# Run with debug logging
python run.py --debug
```

### Generate Default Config

```bash
python run.py --generate-config config.json
```

### Programmatic Usage

```python
import asyncio
from qps_pi_cv_module import PiCvModule, Config

async def main():
    # Load configuration
    config = Config.from_file("config.json")
    
    # Create module
    module = PiCvModule(config)
    
    # Start the module
    await module.start()
    
    # Execute a delivery mission
    success = await module.execute_delivery(
        order_id="order-123",
        destination_lat=40.7442,
        destination_lon=-74.0245,
        target_marker_id=42
    )
    
    # Stop the module
    await module.stop()

asyncio.run(main())
```

## CRC Card Mapping

| CRC Card Component | Implementation |
|-------------------|----------------|
| qpsPiCvModule | `PiCvModule` (core/pi_cv_module.py) |
| FlightController | `FlightController` (flight/flight_controller.py) |
| MarkerDetector | `MarkerDetector` (vision/marker_detector.py) |
| Camera Modules | `CameraManager` (vision/camera_manager.py) |
| qpsDroneCommunication | `BackendClient` (communication/backend_client.py) |
| OpenCV processing | Integrated in `MarkerDetector` |

## Interfaces

All major components implement interfaces defined in `core/interfaces.py`:
- `IFlightController` - Flight control operations
- `IMarkerDetector` - Marker detection
- `ICameraManager` - Camera management
- `IBackendClient` - Backend communication
- `ILandingCoordinator` - Landing coordination

This enables dependency injection and easier testing.

## Data Types

Key data types from `core/interfaces.py`:
- `DroneState` - Drone state enumeration
- `LandingPhase` - Landing sequence phases
- `MarkerDetection` - Detected marker data
- `DronePosition` - GPS position data
- `DroneVelocity` - Velocity vector
- `DroneTelemetry` - Comprehensive telemetry

## Testing

```bash
# Run tests (requires pytest)
pip install pytest pytest-asyncio
pytest tests/

# Run with coverage
pip install pytest-cov
pytest --cov=qps_pi_cv_module tests/
```

## Hardware Setup

### Required Hardware
- Raspberry Pi 5
- Raspberry Pi Camera Module 3 (front - navigation)
- Raspberry Pi Camera Module 3 Wide (bottom - landing)
- USB/UART connection to Pixhawk 2.4.8

### Camera Configuration
```bash
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Verify cameras
libcamera-hello --list-cameras
```

## Troubleshooting

### Camera Issues
```python
# Test camera directly
from picamera2 import Picamera2
picam = Picamera2()
picam.start()
frame = picam.capture_array()
print(f"Frame shape: {frame.shape}")
```

### MAVSDK Connection
```python
# Test MAVSDK connection
import asyncio
from mavsdk import System

async def test():
    drone = System()
    await drone.connect("udp://:14540")
    async for state in drone.core.connection_state():
        print(f"Connected: {state.is_connected}")
        break

asyncio.run(test())
```

## Authors

QuackOps Senior Design Team (Group 8)
- Spurthi Setty - AI/ML & Computer Vision
- Thomas Ung - Embedded Systems
- Gianna Cerbone - Systems Engineering
- Camila Valdez - Mechanical Systems
- George Redfern - Hardware Integration

## License

Stevens Institute of Technology - Senior Design Project 2025-2026
