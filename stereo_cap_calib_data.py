import cv2
import numpy as np
import time

def capture_dual_camera_images():
    # Initialize both cameras (0 and 1 are typical camera indices)
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    
    # Check if cameras opened successfully
    if not cam1.isOpened():
        print("Error: Camera 1 not accessible")
        return
    if not cam2.isOpened():
        print("Error: Camera 2 not accessible")
        return

    # Optional: Set camera resolution (if needed)
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 9999999999)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 999999999)
    cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 99999999)
    cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 999999999)

    try:
        while True:
            # Read frames from both cameras
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()

            # Verify frames were captured
            if not ret1 or not ret2:
                print("Error: Failed to capture images")
                break

            # Display the frames
            cv2.imshow('Camera 1', frame1)
            cv2.imshow('Camera 2', frame2)

            # Press 's' to save images, 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save images with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'cam1_{timestamp}.png', frame1)
                cv2.imwrite(f'cam2_{timestamp}.png', frame2)
                print(f"Images saved: cam1_{timestamp}.jpg, cam2_{timestamp}.jpg")
            
            elif key == ord('q'):
                break

    finally:
        # Release camera resources and close windows
        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_dual_camera_images()