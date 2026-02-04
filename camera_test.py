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
