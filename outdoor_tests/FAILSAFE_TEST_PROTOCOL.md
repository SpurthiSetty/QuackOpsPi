# Failsafe Test Protocol

Four ArduCopter failsafes verified in-flight on the QuackOps drone.
Run preflight_check.py first — all params must pass before proceeding.

---

## 1. RC (Radio) Failsafe

### Setup
| Parameter | Value | Meaning |
|---|---|---|
| `FS_THR_ENABLE` | 1 | RC failsafe enabled → RTL |
| `FS_THR_VALUE` | ≈975 | PWM threshold for throttle loss detection |

### Trigger
1. Take off to 3 m in GUIDED hover (`./run_test.sh hover --alt 3 --hover 30`).
2. During the hover, **power off the RC transmitter**.

### Expected ArduCopter Behavior
- ArduCopter detects throttle channel below `FS_THR_VALUE` within ~1 s.
- Switches to RTL mode automatically.
- Climbs to `RTL_ALT` (15 m), flies to home GPS, descends and lands.

### Expected Mission Planner Display
- Flight mode indicator changes from `GUIDED` → `RTL`.
- Red "Radio Failsafe" banner in HUD.

### Expected Pi Log Output
```
WARNING  qps.failsafe_watcher: RC failsafe detected in STATUSTEXT: Failsafe: Radio
WARNING  qps.test_hover: Hover ended early: rc_failsafe
```
- `failsafes.log` records: `<timestamp>,rc_failsafe,`
- `FailsafeWatcher.rc_failsafe` event fires → `hover_with_logging` returns early.

### Recovery
1. Drone RTLs and lands autonomously — do not interfere.
2. Once landed: power TX back on, confirm disarm in MP.
3. Reset for next test.

---

## 2. GCS (Ground Control Station) Failsafe

### Setup
| Parameter | Value | Meaning |
|---|---|---|
| `FS_GCS_ENABLE` | 1 | GCS failsafe enabled |
| `FS_GCS_TIMEOUT` | 5 | Seconds before failsafe triggers |

MAVProxy must be running and connected (heartbeats flowing to Pixhawk).

### Trigger
1. Take off to 3 m in GUIDED hover.
2. On the Pi, **kill MAVProxy** while in flight:
   ```bash
   pkill mavproxy.py
   ```

### Expected ArduCopter Behavior
- ArduCopter stops receiving GCS heartbeats.
- After `FS_GCS_TIMEOUT` (5 s), switches to RTL.
- Completes RTL and lands.

### Expected Mission Planner Display
- Connection drops — MP shows "No Heartbeat".
- (Reconnect MP after test to verify.)

### Expected Pi Log Output
```
WARNING  qps.failsafe_watcher: GCS failsafe detected in STATUSTEXT: GCS Failsafe
WARNING  qps.test_hover: Hover ended early: gcs_failsafe
```
- `FailsafeWatcher.gcs_failsafe` event fires → early exit.
- The flight script itself also loses MAVLink contact — it will raise an exception and attempt `land()` + `disarm()`, which will likely timeout since MAVProxy is gone. **This is expected** — ArduCopter is already in RTL.

### Recovery
1. Wait for drone to land autonomously.
2. Restart MAVProxy: `./outdoor_tests/start_mavproxy.sh /tmp`
3. Reconnect Mission Planner.

---

## 3. Battery Failsafe

### Setup
| Parameter | Value | Meaning |
|---|---|---|
| `BATT_FS_LOW_ACT` | 2 | Low battery → RTL |
| `BATT_FS_CRT_ACT` | 1 | Critical battery → LAND |
| `BATT_LOW_VOLT` | *(artificially high)* | See below |

**To trigger the failsafe quickly (without actually depleting the battery):**  
Before flight, temporarily set `BATT_LOW_VOLT` to a value **above** the current pack voltage.

Example: if pack reads 12.2 V, set `BATT_LOW_VOLT = 12.5` via Mission Planner or:
```bash
# Via MAVProxy console:
param set BATT_LOW_VOLT 12.5
```

### Trigger
1. Set `BATT_LOW_VOLT` above current pack voltage.
2. Take off to 3 m in GUIDED hover.
3. Failsafe triggers within seconds of arming.

### Expected ArduCopter Behavior
- ArduCopter sees voltage < `BATT_LOW_VOLT`.
- Switches to RTL (`BATT_FS_LOW_ACT=2`).
- If voltage further drops below `BATT_CRT_VOLT` during RTL: switches to LAND.

### Expected Mission Planner Display
- Yellow/red battery indicator, "Battery Failsafe" banner.
- Mode: `RTL`.

### Expected Pi Log Output
```
WARNING  qps.telemetry_monitor: BATTERY WARNING: 85%   ← (percent still high; voltage triggered)
WARNING  qps.failsafe_watcher: Battery WARNING: 85%
WARNING  qps.failsafe_watcher: RC failsafe...  ← may not appear
```
Note: `qpsTelemetryMonitor` fires the warning/critical callbacks on **percentage**, not voltage. The STATUSTEXT "Battery Failsafe" will appear in the log from `FailsafeWatcher`'s STATUSTEXT parsing. The `battery_failsafe` event fires when ArduCopter's voltage-based threshold fires.

### Recovery
1. After landing: **immediately reset `BATT_LOW_VOLT`** to the correct value for the pack:
   ```bash
   param set BATT_LOW_VOLT 10.5   # 3S pack example
   ```
2. Confirm correct value with `preflight_check.py` before next flight.

---

## 4. Geofence Breach

### Setup
| Parameter | Value | Meaning |
|---|---|---|
| `FENCE_ENABLE` | 1 | Geofence active |
| `FENCE_TYPE` | 3 | Altitude + circle fence |
| `FENCE_RADIUS` | 15 | 15 m radius from home |
| `FENCE_ACTION` | 1 | Breach → RTL |

### Trigger
Use `test_lateral.py` with a pattern that intentionally exits the 15 m fence:
```bash
./outdoor_tests/run_test.sh lateral --pattern line --size 20 --bearing 0 --alt 3
```
The script will attempt to fly 20 m north — ArduCopter enforces the 15 m fence.

### Expected ArduCopter Behavior
- As drone approaches 15 m fence boundary, ArduCopter slows.
- On breach: mode switches to RTL.
- Drone returns to home and lands.

### Expected Mission Planner Display
- Mode indicator: `RTL`.
- Red "Fence Breach" banner.
- Fence circle visible on map (if enabled in MP display).

### Expected Pi Log Output
```
WARNING  qps.failsafe_watcher: Fence breach detected in STATUSTEXT: Fence breach
WARNING  qps.test_lateral: Failsafe during transit to WP1: fence_breach
```
- `FailsafeWatcher.fence_breach` fires.
- `wait_for_arrival` raises `RuntimeError` → lateral script's pattern loop breaks.
- Script returns to home (or attempts to — drone is already in RTL).

### Recovery
1. Drone RTLs and lands autonomously.
2. Confirm landing in Mission Planner.
3. No parameter changes needed — fence settings are correct for ongoing use.
4. If testing is done: optionally set `FENCE_ENABLE=0` to disable fence for non-test flights.

---

## Notes

- **Run failsafe tests one at a time.** Reset between each.
- **Always have Ch5 RTL switch available** regardless of which failsafe is being tested.
- `preflight_check.py` verifies all four failsafes are armed and ready before each session.
- After any aborted flight, review `failsafes.log` and `telemetry.csv` in the log directory to confirm the correct failsafe fired.
