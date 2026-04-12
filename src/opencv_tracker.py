"""
OpenCV-Only Face Tracker - No MediaPipe dependencies

Uses OpenCV's built-in face detection to avoid MediaPipe issues.
"""

import cv2
import numpy as np
import yaml
import os


class OpenCVFaceTracker:
    def __init__(self, config_path="../config/settings.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize OpenCV face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        print("✅ OpenCV face detection initialized")

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} not found. Using defaults.")
            return {
                'camera': {'device_id': 0, 'width': 640, 'height': 480, 'fps': 30}
            }

    def detect_phone_looking(self, face_rect, eyes, image_shape):
        """Detect if user is looking down at phone based on face and eye positions"""
        height, width = image_shape[:2]
        x, y, w, h = face_rect

        # Calculate face center
        face_center_y = y + h // 2

        # If face is in the lower part of the screen, likely looking down
        if face_center_y > height * 0.6:
            return True, "Face positioned low (looking down)"

        # Check eye positions relative to face
        if len(eyes) >= 1:
            eye_y_positions = [eye[1] + eye[3]//2 for eye in eyes]  # Eye centers
            avg_eye_y = sum(eye_y_positions) / len(eye_y_positions)

            # Eyes in lower part of detected face = looking down
            relative_eye_position = (avg_eye_y - y) / h
            if relative_eye_position > 0.7:  # Eyes in bottom 30% of face
                return True, f"Eyes low in face ({relative_eye_position:.2f})"

        # Check face aspect ratio - looking down makes face appear taller
        face_aspect_ratio = h / w
        if face_aspect_ratio > 1.4:  # Taller than normal
            return True, f"Face aspect ratio tall ({face_aspect_ratio:.2f})"

        return False, "Looking forward"

    def run(self):
        """Main face tracking loop"""
        print("🎥 Starting OpenCV face tracker...")

        # Initialize camera
        cap = cv2.VideoCapture(self.config['camera']['device_id'])

        if not cap.isOpened():
            print("❌ Failed to open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])

        print("📱 Face Tracker Started!")
        print("   • Green box: Looking at screen")
        print("   • Red box: Looking down (phone territory!)")
        print("   • Press 'q' to quit")
        print("-" * 50)

        frame_count = 0

        while True:
            success, image = cap.read()
            if not success:
                print("Failed to read from camera")
                continue

            frame_count += 1

            # Flip image horizontally for selfie-view
            image = cv2.flip(image, 1)
            height, width, _ = image.shape

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = face

                # Detect eyes within face region
                face_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)

                # Check if looking at phone
                is_phone_looking, reason = self.detect_phone_looking(face, eyes, image.shape)

                if is_phone_looking:
                    # Red alert for phone looking
                    color = (0, 0, 255)
                    status = "📱 LOOKING AT PHONE!"
                    detail = f"Reason: {reason}"

                    # Draw thick red rectangle around face
                    cv2.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), color, 5)

                    # Big warning text
                    cv2.putText(image, status, (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(image, "GET BACK TO WORK!", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                else:
                    # Green for normal looking
                    color = (0, 255, 0)
                    status = "💻 Looking at screen"
                    detail = f"Status: {reason}"

                    # Draw green rectangle around face
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

                # Draw eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), 2)

                # Status text
                cv2.putText(image, status, (10, height-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(image, detail, (10, height-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Face info
                face_info = f"Face: {w}x{h} at ({x},{y})"
                cv2.putText(image, face_info, (10, height-90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                cv2.putText(image, "👤 No face detected", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Frame counter
            cv2.putText(image, f"Frame: {frame_count}", (width-150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            # Show frame
            cv2.imshow('Procrastination Police - OpenCV Tracker', image)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Stopping face tracker...")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Face tracker stopped!")


def main():
    """Test the OpenCV face tracker"""
    tracker = OpenCVFaceTracker()
    tracker.run()


if __name__ == "__main__":
    main()