# QuackOps Outdoor Flight Tests

Test scripts for the Pi 5 + Pixhawk 2.4.8 / ArduCopter 4.6.3 drone.

---

## Prerequisites

**On the Pi (in the venv):**
```bash
cd ~/SeniorD/QuackOpsPi
source venv/bin/activate
pip install mavproxy        # provides mavproxy.py CLI
# pymavlink, opencv-python, picamera2 already installed
```

**Network (DD-WRT router):**
- SSID: `ddwrt`
- Pi static IP: `192.168.1.128`  (set in `/etc/dhcpcd.conf` via ethernet)
- Laptop joins same Wi-Fi

---

## Network / Port Map

| Service | Address | Who connects |
|---|---|---|
| Pi → Pixhawk (serial) | `/dev/ttyAMA0` @ 57600 | MAVProxy |
| Pi script ↔ MAVProxy | `udpin:127.0.0.1:14550` | `qpsFlightManager`, `preflight_check.py` |
| Mission Planner ↔ MAVProxy | `UDP Client 192.168.1.128:14551` | Laptop MP |
| MJPEG stream | `http://192.168.1.128:8080/stream` | Laptop browser |

**Mission Planner setup:**  
`Connect` → `UDP Client` → IP `192.168.1.128`, port `14551`

**Camera stream:**  
Open `http://192.168.1.128:8080` in any browser.

---

## Running a Test

All commands run from the repo root on the Pi:

```bash
# Hover test (3 m, 10 s)
./outdoor_tests/run_test.sh hover --alt 3 --hover 10

# Hover at 5 m for 20 s
./outdoor_tests/run_test.sh hover --alt 5 --hover 20

# Square pattern, 3 m legs, 3 m altitude
./outdoor_tests/run_test.sh lateral --pattern square --size 3 --alt 3

# Line pattern, 5 m due east, 3 m altitude
./outdoor_tests/run_test.sh lateral --pattern line --size 5 --bearing 90 --alt 3

# Triangle pattern, 4 m legs
./outdoor_tests/run_test.sh lateral --pattern triangle --size 4 --alt 3
```

`run_test.sh` automatically starts MAVProxy + the camera daemon before the flight script and tears everything down on exit or Ctrl+C.

---

## Pre-flight Checklist

1. **Power on drone and RC transmitter**
2. **Pi boots and joins network** — confirm: `ping 192.168.1.128`
3. **Start MAVProxy** (done automatically by `run_test.sh`, or manually):
   ```bash
   ./outdoor_tests/start_mavproxy.sh /tmp
   ```
4. **Run preflight parameter check:**
   ```bash
   python3 outdoor_tests/preflight_check.py
   ```
   All rows must show `✓ PASS` before flying.
5. **Connect Mission Planner** (UDP Client → 192.168.1.128:14551) — verify telemetry + map
6. **Confirm battery voltage** in Mission Planner HUD matches physical pack
7. **Test RC failsafe** on the bench: disarm, turn off TX, confirm STATUSTEXT "Failsafe: Radio"
8. **STAND CLEAR** — the script arms automatically 3 seconds after you confirm GPS ready

---

## Retrieving Logs

```bash
# From laptop (replace <timestamp_test> with actual dir name):
scp -r setty@192.168.1.128:~/SeniorD/QuackOpsPi/outdoor_tests/logs/<timestamp_test> .
```

Log directory contents:

| File | Contents |
|---|---|
| `flight.log` | Full timestamped Python log |
| `telemetry.csv` | 2 Hz snapshot: GPS, EKF, vibration, battery |
| `failsafes.log` | Failsafe events with timestamps |
| `video.mp4` (or `.avi`) | Annotated video with ArUco overlays |
| `detections.csv` | Per-frame marker detections |
| `frame_timestamps.csv` | Frame number ↔ Unix timestamp (for video/telemetry sync) |
| `mavproxy.tlog` | Raw MAVLink binary log (open in Mission Planner) |
| `camera.log` | Camera daemon stdout/stderr |

---

## Emergency Procedures

| Situation | Action |
|---|---|
| Flyaway / lost control | **Ch5 switch → RTL** on transmitter |
| Script error | **Ctrl+C** → script attempts `land()` + `disarm()` automatically |
| MAVProxy crash | Ctrl+C the run_test.sh; script detects GCS failsafe and RTLs |
| Battery critical | ArduCopter LAND mode activates (BATT_FS_CRT_ACT=1) |
| Fence breach | ArduCopter RTL activates (FENCE_ACTION=1) |

**The RC transmitter is always the override — switch Ch5 to RTL at any time.**

---

## Safety Limits (enforced by scripts)

| Parameter | Cap |
|---|---|
| `--alt` | 10 m max |
| `--hover` / `--hover-at-each` / `--hover-before` | 30 s max |
| `--size` (lateral leg) | 5 m max |
| Geofence radius | 15 m (set via `FENCE_RADIUS`) |
