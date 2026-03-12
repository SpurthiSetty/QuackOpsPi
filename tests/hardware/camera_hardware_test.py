"""Hardware smoke test for the Raspberry Pi camera.

Run this directly on the Pi to verify the camera is wired and working
before running any project-level tests.

Usage:
    python tests/hardware/camera_hardware_test.py

NOT a pytest test — run with python, not pytest.
"""

from picamera2 import Picamera2
import cv2

# Initialize camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")

# Start the camera
picam2.start()

# Capture and show image
frame = picam2.capture_array()

# Display using OpenCV
cv2.imshow("Camera Test", frame)
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
picam2.close()
