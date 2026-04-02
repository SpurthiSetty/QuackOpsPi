# QuackOps Pi Module — Claude Code Context

## What This Project Is

QuackOps is an autonomous drone delivery system built as a senior design capstone at Stevens Institute of Technology (Sep 2025–May 2026). This repo contains the **Raspberry Pi companion computer software** that controls the drone, runs computer vision, and communicates with a Node.js web backend.

Target: Live demonstration for L3 Harris as a potential sponsor in May 2026.

## Hardware Stack

- **Drone frame:** HAWK'S WORK F450
- **Flight controller:** Pixhawk 2.4.8 running ArduPilot
- **Companion computer:** Raspberry Pi 5 (username: `setty`, hostname: `ssetty`, SSH: `setty@ssetty.local`)
- **Camera:** Dual Raspberry Pi Camera Module 3
- **GPS:** M8N GPS with HMC5883L compass
- **Networking:** Linksys Wireless-G router for local lab network; Pi connects via ethernet
- **UART connection:** Pi ↔ Pixhawk via TELEM1 port (TX→RX, RX→TX, GND→GND), 57600 baud, MAVLink2
- **Serial:** `/dev/ttyAMA0` (serial console disabled via raspi-config)

## Software Stack

- **Pi module:** Python 3 with MAVSDK 3.15.3, OpenCV, FastAPI
- **Web app:** Node.js/Express + React/TypeScript
- **Database:** MongoDB
- **Real-time comms:** Socket.IO / WebSocket
- **Mapping:** Leaflet.js
- **Protocol:** MAVLink 2.0
- **Venv:** `~/SeniorD/QuackOpsPi/venv` (needs `--system-site-packages` for picamera2)

## Architecture Principles

- **`qps` namespace prefix** on all classes and files. One class per file.
- **Dependency injection** throughout — interfaces define contracts, implementations are swappable.
- **Design-first approach** — architecture decisions are formalized before implementation.
- **Tradeoff analysis required** for major decisions — present at least 2 options, recommend one, ask for confirmation.
- **Hardware-aware** — assume limited CPU/RAM, real-time constraints, deterministic behavior.
- **Production-oriented code** — structured logging, failure mode handling (camera disconnect, MAVSDK timeout, telemetry loss), startup/shutdown/recovery states.
- **No tightly coupled god modules.**
- **Testability** — mock implementations for all interfaces enable unit testing without hardware.

## Current Architecture (CRC Card Summary)

### Interfaces

| Interface                    | Contract                                                                                                    |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `qpsFlightManagerInterface`  | Drone flight commands: arm, takeoff, land, navigate, offboard control, **pause_mission**, **goto_location** |
| `qpsCameraManagerInterface`  | Camera frame acquisition on demand                                                                          |
| `qpsMarkerDetectorInterface` | ArUco marker detection + 3D pose estimation                                                                 |
| `qpsBackendClientInterface`  | Bidirectional WebSocket comms with Node.js backend                                                          |

### Core Classes

| Class                  | Role                                                                                                                                                                    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qpsMissionController` | Orchestrates full mission lifecycle via state machine. **Owns orbit waypoint calculation and all flight decisions.**                                                    |
| `qpsLandingController` | **Search-only role.** Runs camera, detects markers during orbit, computes marker world GPS from pose + telemetry. Returns `qpsLandingResult`. Does NOT command landing. |
| `qpsTelemetryMonitor`  | Subscribes to MAVSDK telemetry streams, provides `qpsDroneState` and `qpsGPSPosition` on demand. Fires battery callbacks.                                               |

### Data Classes (immutable, frozen dataclasses)

| Class                | Contents                                                                 |
| -------------------- | ------------------------------------------------------------------------ |
| `qpsGPSPosition`     | lat, lon, altitude, heading, speed, timestamp                            |
| `qpsDroneState`      | GPS, battery, armed, flight mode, GPS quality                            |
| `qpsMarkerDetection` | marker ID, corners, center, confidence, tvec/rvec/distance               |
| `qpsMissionCommand`  | command type, order ID, destination GPS, marker ID                       |
| `qpsStatusMessage`   | status type, order ID, data payload, timestamp                           |
| `qpsLandingResult`   | outcome enum, marker_gps, fallback_gps, search_duration, frames_searched |
| `qpsConfig`          | All tunable parameters, loaded from JSON with defaults                   |

### Production Implementations

- `qpsFlightManager` → MAVSDK Action/Mission/Offboard plugins
- `qpsPiCameraManager` → PiCamera2
- `qpsCVCameraManager` → OpenCV VideoCapture
- `qpsMarkerDetector` → cv2.aruco + solvePnP
- `qpsBackendClient` → WebSocket with heartbeat, reconnection, message queuing

### Mock Implementations (Testing)

- `qpsMockFlightManager`, `qpsMockCameraManager`, `qpsMockMarkerDetector`, `qpsMockBackendClient`

## Key Design Decision: Orbit-Based Landing (DECIDED)

The landing sequence uses an **orbit search pattern** (NOT staircase descent):

### Happy Path

1. `qpsMissionController` calculates N orbit waypoints around delivery GPS center
2. Uploads and starts the orbit mission via `qpsFlightManagerInterface`
3. `qpsLandingController.execute_marker_search()` runs concurrently — captures frames, detects ArUco markers
4. Marker found → landing controller computes marker's world GPS from camera pose + drone telemetry
5. Returns `qpsLandingResult(MARKER_FOUND, marker_gps=...)`
6. Mission controller pauses orbit, flies to marker GPS via `goto_location()`, commands `land()`
7. Waits for landed state, sends `DELIVERY_COMPLETE` to backend

### Fallback Path (timeout, no marker)

1. Search times out → landing controller captures current GPS, returns `qpsLandingResult(SEARCH_TIMEOUT, fallback_gps=...)`
2. Mission controller pauses orbit, flies to delivery center GPS, lands
3. Sends `FALLBACK_LANDING` + `MARKER_NOT_FOUND` notifications to backend

### Design Decisions Applied

- **Orbit replaces staircase** — once marker is spotted during orbit, land directly at computed GPS
- **"Land at Destination GPS"** = land at the marker's detected GPS position (computed from camera pose + drone telemetry)
- **qpsMissionController owns orbit planning** — landing controller only runs the visual search
- **Camera assumed pointing straight down** — body frame mapping: camera*x → body_east, camera*-y → body_north

## Interface Changes from Original CRC Cards

### Added to qpsFlightManagerInterface

- `pause_mission()` — pause running waypoint mission (ArduPilot: switch to GUIDED mode hold)
- `goto_location(lat, lon, alt, yaw)` — fly to specific GPS and loiter

### qpsLandingController Revised Role

- **Removed:** staircase descent, offboard mode control, velocity corrections, hover setpoints
- **Added:** orbit-phase marker search loop, marker GPS computation from pose + telemetry
- **Simplified:** returns qpsLandingResult; does NOT command landing

### qpsLandingResult Updated

- `outcome: qpsLandingOutcome` enum (MARKER_FOUND, SEARCH_TIMEOUT, CAMERA_FAILURE, ABORTED)
- `marker_gps: Optional[qpsGPSPosition]`
- `fallback_gps: Optional[qpsGPSPosition]`
- `search_duration_s`, `frames_searched`, `target_marker_id`

## Config Parameters (orbit-related)

```python
orbit_radius_m: float = 15.0          # search orbit radius
orbit_num_points: int = 8             # waypoints on orbit circle
orbit_altitude_m: float = 10.0        # orbit altitude AMSL
search_timeout_s: float = 60.0        # max marker search time
goto_arrival_tolerance_m: float = 2.0  # "arrived" threshold
target_marker_id: int = 0             # default ArUco marker ID
```

## Known Open Issues

- `in_air` and `landed_state` telemetry timeout — ArduPilot/MAVSDK compatibility gap with `EXTENDED_SYS_STATE`
- No battery current sensor configured
- Docker Desktop virtualization disabled on Windows laptop, blocking PX4 SITL testing
- `picamera2` incompatible with standard venv — requires `--system-site-packages`
- Pi IP on router subnet needs confirmation (laptop is 192.168.1.2)

## ArduPilot-Specific Notes

- Stream parameters are port-specific: `SR1_*` governs TELEM1. Writing to `SR2_*` silently does nothing for TELEM1.
- Correct telemetry config: `SR1_POSITION=2`, `SR1_EXTRA1=5`, `SERIAL1_PROTOCOL=2`
- ESP32 DroneBridge was removed — Pi handles all MAVLink routing directly.
- `pause_mission()` on ArduPilot = switch to GUIDED mode hold
- `goto_location()` on ArduPilot = GUIDED mode goto

## Code Style

- Python async/await throughout (MAVSDK is async)
- Structured logging via `logging.getLogger("qps.<module>")`
- Type hints on all public methods
- Frozen dataclasses for immutable value objects
- `from __future__ import annotations` in all files

## File Structure (Domain-Based)

Each domain folder contains its interface, production implementation(s), and mock together.
Every folder has an `__init__.py` that re-exports public names for clean imports.

```
quackops_pi/
├── __init__.py
│
├── flight/                              # Drone flight control
│   ├── __init__.py
│   ├── qps_flight_manager_interface.py
│   ├── qps_flight_manager.py           # TODO — production MAVSDK impl
│   └── qps_mock_flight_manager.py      # TODO
│
├── vision/                              # Camera + ArUco marker detection
│   ├── __init__.py
│   ├── qps_camera_manager_interface.py
│   ├── qps_pi_camera_manager.py        # TODO — PiCamera2 impl
│   ├── qps_cv_camera_manager.py        # TODO — OpenCV VideoCapture impl
│   ├── qps_mock_camera_manager.py      # TODO
│   ├── qps_marker_detector_interface.py
│   ├── qps_marker_detector.py          # TODO — cv2.aruco + solvePnP
│   └── qps_mock_marker_detector.py     # TODO
│
├── comms/                               # Backend communication
│   ├── __init__.py
│   ├── qps_backend_client_interface.py
│   ├── qps_backend_client.py           # TODO — WebSocket impl
│   └── qps_mock_backend_client.py      # TODO
│
├── mission/                             # Mission orchestration + landing search
│   ├── __init__.py
│   ├── qps_mission_controller.py       # TODO — full state machine
│   ├── qps_mission_controller_landing.py  # Orbit landing orchestration (mixin)
│   └── qps_landing_controller.py       # Marker search during orbit
│
├── telemetry/                           # Drone state monitoring
│   ├── __init__.py
│   └── qps_telemetry_monitor.py        # Stub — needs MAVSDK subscription impl
│
├── models/                              # All data classes
│   ├── __init__.py
│   ├── qps_gps_position.py
│   ├── qps_drone_state.py
│   ├── qps_marker_detection.py
│   ├── qps_landing_result.py
│   ├── qps_mission_command.py          # TODO
│   └── qps_status_message.py          # TODO
│
├── config/                              # Configuration
│   ├── __init__.py
│   └── qps_config.py
│
└── main.py                              # Entry point — wires everything together
```

### Import Convention

With domain folders, imports look like:

```python
from quackops_pi.flight import qpsFlightManagerInterface
from quackops_pi.vision import qpsCameraManagerInterface, qpsMarkerDetectorInterface
from quackops_pi.comms import qpsBackendClientInterface
from quackops_pi.models import qpsGPSPosition, qpsDroneState, qpsMarkerDetection
from quackops_pi.config import qpsConfig
from quackops_pi.telemetry import qpsTelemetryMonitor
from quackops_pi.mission import qpsMissionController, qpsLandingController
```

Each domain `__init__.py` re-exports its public classes so consumers never import from individual files:

```python
# flight/__init__.py
from .qps_flight_manager_interface import qpsFlightManagerInterface
from .qps_flight_manager import qpsFlightManager
```

### Why This Structure

- **Debug by domain:** When fixing flight issues, everything is in `flight/`. Camera bugs → `vision/`.
- **Interface + impl + mock co-located:** No hopping between 4 folders to understand one subsystem.
- **Scales forward:** Adding swarm coordination → new `swarm/` folder. New sensor → goes in `vision/`. New comms channel → goes in `comms/`.
- **Clean imports:** Domain `__init__.py` files re-export, so consumers don't couple to file names.

## Team

Kantharaju (primary dev on Pi module, system architecture, full-stack web), Spurthi Setty, Thomas Ung, Gianna Cerbone, Camila Valdez, George Redfern. Faculty advisor.
