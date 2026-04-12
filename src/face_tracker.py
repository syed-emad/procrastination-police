"""
Face Tracker - Step 1: Basic Face Detection

Uses MediaPipe to detect and track face landmarks in real-time.
Displays face mesh and calculates basic head pose.
"""

import cv2
import os
import numpy as np
import yaml

# Fix matplotlib font loading issue on macOS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import mediapipe as mp


class FaceTracker:
    def __init__(self, config_path="../config/settings.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Face mesh model
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config['face_detection']['min_detection_confidence'],
            min_tracking_confidence=self.config['face_detection']['min_tracking_confidence']
        )

        # 3D face model points (nose tip, chin, left eye, right eye, left mouth, right mouth)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left Mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])

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

    def get_head_pose(self, landmarks, image_shape):
        """Calculate head pose from face landmarks"""
        height, width = image_shape[:2]

        # 2D image points from landmarks
        image_points = np.array([
            (landmarks[1].x * width, landmarks[1].y * height),      # Nose tip
            (landmarks[175].x * width, landmarks[175].y * height),  # Chin
            (landmarks[33].x * width, landmarks[33].y * height),    # Left eye left corner
            (landmarks[263].x * width, landmarks[263].y * height),  # Right eye right corner
            (landmarks[61].x * width, landmarks[61].y * height),    # Left mouth corner
            (landmarks[291].x * width, landmarks[291].y * height)   # Right mouth corner
        ], dtype="double")

        # Camera matrix (assuming focal length = width)
        focal_length = width
        center = (width/2, height/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4,1))

        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs
        )

        if success:
            # Convert rotation vector to angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Calculate Euler angles
            sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])

            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                yaw = np.arctan2(-rotation_matrix[2,0], sy)
                roll = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                pitch = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                yaw = np.arctan2(-rotation_matrix[2,0], sy)
                roll = 0

            # Convert to degrees
            pitch = np.degrees(pitch)
            yaw = np.degrees(yaw)
            roll = np.degrees(roll)

            return pitch, yaw, roll

        return None, None, None

    def run(self):
        """Main face tracking loop"""
        # Initialize camera
        cap = cv2.VideoCapture(self.config['camera']['device_id'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])

        print("🎥 Face Tracker Started!")
        print("Press 'q' to quit")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read from camera")
                break

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
                    # Draw face mesh
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                    # Calculate head pose
                    pitch, yaw, roll = self.get_head_pose(face_landmarks.landmark, image.shape)

                    if pitch is not None:
                        # Display pose information
                        pose_text = f"Pitch: {pitch:.1f}° Yaw: {yaw:.1f}° Roll: {roll:.1f}°"
                        cv2.putText(image, pose_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Check if looking at phone (basic threshold)
                        if pitch > 15:  # Looking down
                            status = "📱 LOOKING AT PHONE!"
                            color = (0, 0, 255)  # Red
                        else:
                            status = "💻 Looking at screen"
                            color = (0, 255, 0)  # Green

                        cv2.putText(image, status, (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(image, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show frame
            cv2.imshow('Procrastination Police - Face Tracker', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Test the face tracker"""
    tracker = FaceTracker()
    tracker.run()


if __name__ == "__main__":
    main()