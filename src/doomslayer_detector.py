"""
Procrastination Police - Inspired by doom-slayer detector

Uses proven detection algorithms from the doom-slayer project:
- Calibration phase for personalization
- Multi-factor scoring with refined thresholds
- Sustained detection requirement
- Gaze direction analysis
"""

import cv2
import numpy as np
import yaml
import os
import time


class DoomSlayerDetector:
    def __init__(self, config_path="../config/settings.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize OpenCV cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Calibration phase (like doom-slayer)
        self.calibration_frames = 30
        self.calibration_data = []
        self.baseline = None
        self.is_calibrated = False

        # Detection state
        self.detection_count = 0
        self.detection_threshold = 3  # Like doom-slayer: require 3 consecutive frames
        self.last_detection_time = 0
        self.sensitivity = 0.5  # Adjustable sensitivity

        # Frame tracking
        self.frame_count = 0

        print("🎯 Doom-Slayer style detector initialized")
        print("📊 Calibration phase: Looking at screen normally for first 30 frames")

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

    def analyze_face_metrics(self, face_rect, eyes, image_shape):
        """Extract face metrics for analysis"""
        height, width = image_shape[:2]
        x, y, w, h = face_rect

        metrics = {}

        # Face position (normalized 0-1)
        metrics['face_center_y'] = (y + h // 2) / height
        metrics['face_size'] = (w * h) / (width * height)  # Normalized face size

        # Eye analysis
        if len(eyes) >= 1:
            # Get primary eye positions
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_center_x = (x + ex + ew // 2) / width
                eye_center_y = (y + ey + eh // 2) / height
                eye_centers.append((eye_center_x, eye_center_y))

            if len(eye_centers) >= 2:
                # Sort by x position
                eye_centers.sort(key=lambda e: e[0])
                left_eye, right_eye = eye_centers[0], eye_centers[1]

                # Gaze analysis (simplified iris tracking)
                metrics['gaze_vertical'] = (left_eye[1] + right_eye[1]) / 2
                metrics['gaze_horizontal'] = abs(left_eye[0] - right_eye[0])

                # Eye position within face (like doom-slayer's relative positioning)
                eye_in_face_y = ((left_eye[1] + right_eye[1]) / 2 - y / height) / (h / height)
                metrics['eye_relative_y'] = eye_in_face_y
            elif len(eye_centers) == 1:
                # Single eye detected
                metrics['gaze_vertical'] = eye_centers[0][1]
                metrics['gaze_horizontal'] = eye_centers[0][0]
                metrics['eye_relative_y'] = (eye_centers[0][1] - y / height) / (h / height)

        return metrics

    def calibrate(self, metrics):
        """Calibration phase - establish baseline like doom-slayer"""
        self.calibration_data.append(metrics)

        if len(self.calibration_data) >= self.calibration_frames:
            # Calculate baseline from calibration data
            baseline_gaze_vertical = sum(d.get('gaze_vertical', 0.5) for d in self.calibration_data) / len(self.calibration_data)
            baseline_face_center_y = sum(d.get('face_center_y', 0.5) for d in self.calibration_data) / len(self.calibration_data)
            baseline_face_size = sum(d.get('face_size', 0.1) for d in self.calibration_data) / len(self.calibration_data)

            self.baseline = {
                'gaze_vertical': baseline_gaze_vertical,
                'face_center_y': baseline_face_center_y,
                'face_size': baseline_face_size
            }

            self.is_calibrated = True
            print("✅ Calibration complete! Baseline established.")
            print(f"📊 Baseline: gaze={baseline_gaze_vertical:.3f}, position={baseline_face_center_y:.3f}")

    def detect_doomscrolling(self, metrics):
        """Main detection logic inspired by doom-slayer"""
        if not self.is_calibrated:
            return False, ["Still calibrating..."], 0

        detection_score = 0
        reasons = []

        # 1. Looking down detection (like doom-slayer's gazeDownOffset)
        gaze_down_offset = metrics.get('gaze_vertical', 0.5) - self.baseline['gaze_vertical']
        face_down_offset = metrics.get('face_center_y', 0.5) - self.baseline['face_center_y']

        # Doom-slayer thresholds: gazeDownOffset > 0.12 || headDownOffset > 0.08
        if gaze_down_offset > 0.08 or face_down_offset > 0.06:  # Slightly more sensitive
            detection_score += 4
            reasons.append(f"Looking down (gaze:{gaze_down_offset:.3f}, face:{face_down_offset:.3f})")

        # 2. Face proximity detection (closer to screen = phone usage)
        size_offset = metrics.get('face_size', 0.1) - self.baseline['face_size']
        if size_offset > 0.02:  # Face 2% larger than baseline
            detection_score += 2
            reasons.append(f"Moved closer ({size_offset:.3f})")

        # 3. Eye position analysis
        if 'eye_relative_y' in metrics:
            if metrics['eye_relative_y'] > 0.7:  # Eyes in lower part of face
                detection_score += 2
                reasons.append(f"Eyes low in face ({metrics['eye_relative_y']:.3f})")

        # 4. Sensitivity adjustment (like doom-slayer)
        sensitivity_bonus = (0.55 - self.sensitivity) * 4
        detection_score += sensitivity_bonus

        # 5. Detection threshold logic (like doom-slayer's sustained detection)
        is_detected = detection_score >= 5

        if is_detected:
            self.detection_count += 1
            reasons.append(f"Detection count: {self.detection_count}/{self.detection_threshold}")
        else:
            self.detection_count = max(0, self.detection_count - 1)  # Gradual decay

        # Require sustained detection (like doom-slayer)
        sustained_detection = self.detection_count >= self.detection_threshold

        return sustained_detection, reasons, detection_score

    def run(self):
        """Main detection loop"""
        print("🎯 Doom-Slayer Procrastination Detector")
        print("=" * 50)

        # Initialize camera
        cap = cv2.VideoCapture(self.config['camera']['device_id'])
        if not cap.isOpened():
            print("❌ Failed to open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])

        print("📹 Starting detection...")
        if not self.is_calibrated:
            print("📊 CALIBRATION PHASE: Look at your screen normally for a few seconds")
            print("   (This establishes your baseline for accurate detection)")

        while True:
            success, image = cap.read()
            if not success:
                continue

            self.frame_count += 1

            # Flip image horizontally for selfie-view
            image = cv2.flip(image, 1)
            height, width, _ = image.shape

            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = face

                # Detect eyes within face region
                face_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3)

                # Extract face metrics
                metrics = self.analyze_face_metrics(face, eyes, image.shape)

                # Calibration phase
                if not self.is_calibrated:
                    self.calibrate(metrics)

                    # Draw calibration progress
                    progress = len(self.calibration_data) / self.calibration_frames
                    bar_width = int(progress * 300)
                    cv2.rectangle(image, (50, 50), (50 + bar_width, 80), (0, 255, 0), -1)
                    cv2.rectangle(image, (50, 50), (350, 80), (255, 255, 255), 2)
                    cv2.putText(image, f"Calibrating... {len(self.calibration_data)}/{self.calibration_frames}",
                               (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Draw face during calibration
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

                else:
                    # Detection phase
                    is_doomscrolling, reasons, score = self.detect_doomscrolling(metrics)

                    if is_doomscrolling:
                        # RED ALERT - DOOMSCROLLING DETECTED!
                        color = (0, 0, 255)
                        status = "🚫 DOOMSCROLLING DETECTED!"

                        # Big red warning (like doom-slayer)
                        cv2.rectangle(image, (10, 10), (width-10, 100), color, 4)
                        cv2.putText(image, "STOP DOOMSCROLLING!", (20, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                        cv2.putText(image, "GET BACK TO WORK!", (20, 85),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                        # Red face border
                        cv2.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), color, 5)
                    else:
                        # Green - productive
                        color = (0, 255, 0)
                        status = "✅ Focused & Productive"
                        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

                    # Draw eyes
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), 2)

                    # Status display
                    cv2.putText(image, f"{status}", (10, height-90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(image, f"Score: {score:.1f} | Count: {self.detection_count}/{self.detection_threshold}",
                               (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Show main reasons
                    if reasons:
                        reason_text = " | ".join(reasons[:2])[:70]
                        cv2.putText(image, reason_text, (10, height-30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            else:
                cv2.putText(image, "👤 No face detected", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Frame counter
            cv2.putText(image, f"Frame: {self.frame_count}", (width-120, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            # Show frame
            cv2.imshow('Procrastination Police - Doom Slayer Edition', image)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset calibration
                print("🔄 Resetting calibration...")
                self.calibration_data = []
                self.baseline = None
                self.is_calibrated = False
                self.detection_count = 0

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Doom-slayer detector stopped!")


def main():
    """Test the doom-slayer inspired detector"""
    detector = DoomSlayerDetector()
    detector.run()


if __name__ == "__main__":
    main()