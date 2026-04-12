"""
Enhanced OpenCV Face Tracker with Phone Detection

Combines multiple detection methods for accurate phone usage detection.
"""

import cv2
import numpy as np
import yaml
import os


class EnhancedFaceTracker:
    def __init__(self, config_path="../config/settings.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize OpenCV cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Phone detection parameters
        self.phone_templates = self._create_phone_templates()

        # Tracking history for stability
        self.detection_history = []
        self.history_size = 10

        # Baseline face size (will be set during first few frames)
        self.baseline_face_size = None
        self.baseline_frames = 30
        self.frame_count = 0

        print("✅ Enhanced OpenCV face detection initialized")

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

    def _create_phone_templates(self):
        """Create simple phone-shaped templates for detection"""
        templates = []

        # Various phone aspect ratios
        for width, height in [(30, 60), (25, 50), (35, 70), (20, 40)]:
            template = np.ones((height, width), dtype=np.uint8) * 255
            cv2.rectangle(template, (2, 2), (width-3, height-3), 0, 2)
            templates.append(template)

        return templates

    def detect_phone_objects(self, gray_image):
        """Detect phone-shaped rectangular objects"""
        phone_candidates = []

        # Edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Phone-like aspect ratio (taller than wide)
            aspect_ratio = h / w if w > 0 else 0

            # Phone size constraints
            min_size = 30
            max_size = 200

            if (1.5 < aspect_ratio < 3.5 and  # Phone-like aspect ratio
                min_size < w < max_size and
                min_size < h < max_size and
                cv2.contourArea(contour) > 500):  # Minimum area

                phone_candidates.append((x, y, w, h, aspect_ratio))

        return phone_candidates

    def detect_hands(self, gray_image):
        """Simple hand/object detection using contours"""
        # Threshold to find skin-like regions
        _, thresh = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hand_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:  # Hand-like size
                x, y, w, h = cv2.boundingRect(contour)
                hand_candidates.append((x, y, w, h))

        return hand_candidates

    def analyze_face_distance(self, face_rect):
        """Detect if face is closer than normal (phone proximity effect)"""
        x, y, w, h = face_rect
        face_size = w * h

        # Establish baseline during first frames
        if self.frame_count < self.baseline_frames:
            if self.baseline_face_size is None:
                self.baseline_face_size = face_size
            else:
                self.baseline_face_size = (self.baseline_face_size * 0.9 + face_size * 0.1)

        if self.baseline_face_size is None:
            return False, "Calibrating..."

        # If face is significantly larger than baseline, user moved closer
        size_ratio = face_size / self.baseline_face_size

        if size_ratio > 1.15:  # Lowered from 1.3 - 15% larger = moved closer
            return True, f"Face {size_ratio:.2f}x closer (phone proximity)"

        return False, f"Normal distance ({size_ratio:.2f}x)"

    def detect_head_angle(self, face_rect, eyes):
        """Enhanced head angle detection"""
        x, y, w, h = face_rect

        reasons = []
        phone_score = 0

        # 1. Face position in frame
        frame_height = 480  # Default height
        face_center_y = y + h // 2
        relative_position = face_center_y / frame_height

        if relative_position > 0.55:  # Lowered from 0.65 to 0.55 - face in lower 45% of frame
            phone_score += 2
            reasons.append(f"Face low in frame ({relative_position:.2f})")

        # 2. Eye analysis
        if len(eyes) >= 2:
            # Sort eyes by x position
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye, right_eye = eyes[0], eyes[1]

            # Eye altitude within face
            left_eye_y = left_eye[1] + left_eye[3] // 2
            right_eye_y = right_eye[1] + right_eye[3] // 2
            avg_eye_y = (left_eye_y + right_eye_y) / 2

            eye_relative_pos = (avg_eye_y - y) / h  # Position within face (0=top, 1=bottom)

            if eye_relative_pos > 0.5:  # Lowered from 0.6 - Eyes in lower half of face
                phone_score += 3
                reasons.append(f"Eyes low in face ({eye_relative_pos:.2f})")

            # Eye distance (looking down can change eye appearance)
            eye_distance = abs(left_eye[0] - right_eye[0])
            if eye_distance < w * 0.3:  # Eyes appear closer (head turned down)
                phone_score += 1
                reasons.append(f"Eyes close together ({eye_distance}/{w})")

        # 3. Face aspect ratio
        aspect_ratio = h / w
        if aspect_ratio > 1.2:  # Taller face (looking down)
            phone_score += 1
            reasons.append(f"Tall face aspect ({aspect_ratio:.2f})")

        return phone_score >= 2, reasons, phone_score  # Lowered threshold from 3 to 2

    def comprehensive_phone_detection(self, image, gray, face_rect, eyes):
        """Combine all detection methods"""
        detection_score = 0
        all_reasons = []

        # Method 1: Head angle analysis
        head_phone_detected, head_reasons, head_score = self.detect_head_angle(face_rect, eyes)
        detection_score += head_score
        all_reasons.extend(head_reasons)

        # Method 2: Face distance analysis
        distance_detected, distance_reason = self.analyze_face_distance(face_rect)
        if distance_detected:
            detection_score += 2
            all_reasons.append(distance_reason)

        # Method 3: Phone object detection
        phone_objects = self.detect_phone_objects(gray)
        if phone_objects:
            detection_score += 3
            all_reasons.append(f"Phone object detected ({len(phone_objects)})")

            # Draw detected phones
            for px, py, pw, ph, aspect in phone_objects:
                cv2.rectangle(image, (px, py), (px+pw, py+ph), (255, 0, 255), 2)
                cv2.putText(image, f"Phone?", (px, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Method 4: Hand detection
        hands = self.detect_hands(gray)
        if hands:
            detection_score += 1
            all_reasons.append(f"Hand/object detected ({len(hands)})")

            # Draw detected hands
            for hx, hy, hw, hh in hands:
                cv2.rectangle(image, (hx, hy), (hx+hw, hy+hh), (0, 255, 255), 1)

        # More sensitive thresholds
        is_phone_detected = detection_score >= 2  # Lowered from 4 to 2
        self.detection_history.append(is_phone_detected)

        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)

        # Less aggressive stability filtering (2 out of 3 recent frames)
        recent_frames = self.detection_history[-3:] if len(self.detection_history) >= 3 else self.detection_history
        stable_detection = sum(recent_frames) >= 2 if len(recent_frames) >= 3 else is_phone_detected

        return stable_detection, all_reasons, detection_score

    def run(self):
        """Main face tracking loop with enhanced phone detection"""
        print("🎥 Starting Enhanced Face Tracker...")

        # Initialize camera
        cap = cv2.VideoCapture(self.config['camera']['device_id'])

        if not cap.isOpened():
            print("❌ Failed to open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])

        print("📱 Enhanced Phone Detection Started!")
        print("   • Multiple detection methods combined")
        print("   • Stability filtering applied")
        print("   • Press 'q' to quit")
        print("-" * 50)

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
                eyes = self.eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)

                # Comprehensive phone detection
                is_phone, reasons, score = self.comprehensive_phone_detection(image, gray, face, eyes)

                if is_phone:
                    # RED ALERT - PHONE DETECTED!
                    color = (0, 0, 255)
                    status = "📱 PHONE DETECTED!"

                    # Big red warning
                    cv2.rectangle(image, (x-15, y-15), (x+w+15, y+h+15), color, 6)
                    cv2.rectangle(image, (10, 10), (width-10, 120), color, 3)

                    # Warning text
                    cv2.putText(image, "STOP LOOKING AT PHONE!", (20, 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                    cv2.putText(image, "GET BACK TO WORK!", (20, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                else:
                    # Green - good behavior
                    color = (0, 255, 0)
                    status = "💻 Focused on work"
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)

                # Draw eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), 2)

                # Detection details
                cv2.putText(image, f"Score: {score} | {status}", (10, height-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Show main reasons
                main_reasons = reasons[:2] if len(reasons) > 2 else reasons
                reason_text = " | ".join(main_reasons)[:60]
                cv2.putText(image, reason_text, (10, height-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Stability indicator
                history_text = "".join(["🔴" if x else "🟢" for x in self.detection_history[-5:]])
                cv2.putText(image, f"History: {history_text}", (10, height-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                cv2.putText(image, "👤 No face detected", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Frame info
            cv2.putText(image, f"Frame: {self.frame_count}", (width-120, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            # Show frame
            cv2.imshow('Procrastination Police - Enhanced Tracker', image)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Stopping enhanced tracker...")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Enhanced tracker stopped!")


def main():
    """Test the enhanced face tracker"""
    tracker = EnhancedFaceTracker()
    tracker.run()


if __name__ == "__main__":
    main()