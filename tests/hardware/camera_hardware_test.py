"""Hardware smoke test for the Raspberry Pi camera.

Run this directly on the Pi to verify the camera is wired and working
before running any project-level tests.

Usage:
    python tests/hardware/camera_hardware_test.py

NOT a pytest test — run with python, not pytest.
"""

from picamera2 import Picamera2

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")

picam2.start()
print("Camera started, capturing frame...")

frame = picam2.capture_array()
print(f"Frame captured: {frame.shape}")

# Save to file instead of displaying (no display over SSH)
import cv2
cv2.imwrite("test_picamera.jpg", frame)
print("Saved to test_picamera.jpg")

picam2.close()
print("SUCCESS — picamera2 works!")