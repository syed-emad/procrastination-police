"""
Simple Face Tracker - Step 1: Basic Face Detection

Lightweight version that avoids font loading issues on macOS.
"""

import cv2
import numpy as np
import yaml
import os

# Set environment variables to avoid font loading issues
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = '/tmp'

import mediapipe as mp


class SimpleFaceTracker:
    def __init__(self, config_path="../config/settings.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Face mesh model - use minimal settings to avoid issues
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Simplified
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
            return {
                'camera': {'device_id': 0, 'width': 640, 'height': 480, 'fps': 30},
                'face_detection': {'min_detection_confidence': 0.5, 'min_tracking_confidence': 0.5}
            }

    def get_simple_head_pose(self, landmarks, image_shape):
        """Calculate simple head pose from face landmarks"""
        height, width = image_shape[:2]

        # Get key landmark points
        nose_tip = landmarks[1]
        chin = landmarks[175]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        # Convert to pixel coordinates
        nose_y = nose_tip.y * height
        chin_y = chin.y * height

        # Simple pitch calculation - if chin is much lower than nose, looking down
        face_height = chin_y - nose_y

        # Estimate pitch (positive = looking down)
        if face_height > height * 0.12:  # Face takes up more vertical space
            pitch = 25  # Looking down (phone territory)
        elif face_height > height * 0.08:
            pitch = 10  # Slightly down
        else:
            pitch = 0   # Looking ahead

        # Simple yaw calculation based on eye positions
        left_eye_x = left_eye.x * width
        right_eye_x = right_eye.x * width
        eye_center = (left_eye_x + right_eye_x) / 2
        face_center = width / 2

        yaw = (eye_center - face_center) / width * 30  # Simple yaw estimate

        return pitch, yaw, 0  # pitch, yaw, roll

    def run(self):
        """Main face tracking loop"""
        print("🎥 Initializing camera...")

        # Initialize camera
        cap = cv2.VideoCapture(self.config['camera']['device_id'])

        if not cap.isOpened():
            print("❌ Failed to open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])

        print("📱 Face Tracker Started!")
        print("   • Look at screen: Green status")
        print("   • Look down (phone): Red alert!")
        print("   • Press 'q' to quit")
        print("-" * 40)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read from camera")
                continue

            # Flip image horizontally for selfie-view
            image = cv2.flip(image, 1)
            height, width, _ = image.shape

            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False

            # Process face detection
            results = self.face_mesh.process(rgb_image)

            # Convert back to BGR for OpenCV
            rgb_image.flags.writeable = True
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw simple face outline (not full mesh to avoid complexity)
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_FACE_OVAL,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=1, circle_radius=1))

                    # Calculate simple head pose
                    pitch, yaw, roll = self.get_simple_head_pose(face_landmarks.landmark, image.shape)

                    # Display pose information
                    pose_text = f"Head: {pitch:.1f}° down, {yaw:.1f}° side"
                    cv2.putText(image, pose_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Check if looking at phone
                    if pitch > 15:  # Looking down enough to be phone territory
                        status = "📱 LOOKING AT PHONE! GET BACK TO WORK!"
                        color = (0, 0, 255)  # Red
                        # Make it more obvious
                        cv2.rectangle(image, (5, 50), (width-5, 100), color, 3)
                    else:
                        status = "💻 Good - Looking at screen"
                        color = (0, 255, 0)  # Green

                    cv2.putText(image, status, (10, 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(image, "👤 No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show frame
            cv2.imshow('Procrastination Police - Face Tracker', image)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quitting face tracker...")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Face tracker stopped!")


def main():
    """Test the simplified face tracker"""
    tracker = SimpleFaceTracker()
    tracker.run()


if __name__ == "__main__":
    main()