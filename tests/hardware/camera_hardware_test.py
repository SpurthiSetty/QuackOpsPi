"""Hardware smoke test for the Raspberry Pi camera.

Run this directly on the Pi to verify the camera is wired and working
before running any project-level tests.

Usage:
    python tests/hardware/camera_hardware_test.py

NOT a pytest test — run with python, not pytest.
"""

import cv2
from picamera2 import Picamera2

# --- List detected cameras ---
camera_info = Picamera2.global_camera_info()
print(f"Detected {len(camera_info)} camera(s):")
for i, info in enumerate(camera_info):
    print(f"  [{i}] {info}")

if len(camera_info) < 2:
    print("\nWARNING: Expected 2 cameras but only found "
          f"{len(camera_info)}. Check /boot/firmware/config.txt "
          "and run: libcamera-hello --list-cameras")

# --- Capture from each camera ---
for i, info in enumerate(camera_info):
    model = info.get("Model", f"camera{i}")
    filename = f"test_picamera_{i}_{model}.jpg"

    print(f"\nCapturing from camera {i} ({model})...")

    cam = Picamera2(i)
    cam.preview_configuration.main.size = (640, 480)
    cam.preview_configuration.main.format = "RGB888"
    cam.configure("preview")
    cam.start()

    frame = cam.capture_array()
    print(f"  Frame captured: {frame.shape}")

    cv2.imwrite(filename, frame)
    print(f"  Saved to {filename}")

    cam.close()

print("\nSUCCESS — all cameras tested!")
