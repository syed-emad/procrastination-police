"""
Procrastination Police - Multi-modal detection

Combines:
- Iris tracking (from Skyrim edition)
- Phone object detection (from doom-slayer)
- Eye state detection (open/closed)
"""

import cv2
import mediapipe as mp
import time
import random
import os
import subprocess
import pygame
import numpy as np
import threading


class OfficeClipDetector:
    def __init__(self):
        # Detection thresholds (adjusted for better sensitivity)
        self.looking_down_threshold = 0.5  # Increased from 0.25 to catch your eye position
        self.debounce_threshold = 0.6      # Increased from 0.45
        self.timer = 0.0  # seconds - instant trigger

        # State tracking
        self.video_playing = False
        self.start_time = None
        self.last_clip_time = 0
        self.clip_cooldown = 15.0  # seconds before next clip can trigger

        # Calibration for personalized thresholds
        self.calibration_frames = []
        self.baseline_ratio = None
        self.is_calibrating = True
        self.calibration_count = 0
        self.max_calibration_frames = 30

        # MediaPipe setup (exact from Skyrim edition)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils

        # Phone detection setup
        self.phone_detected = False
        self.last_phone_check = 0
        self.phone_check_interval = 0.5  # Check every 500ms like doom-slayer

        # Load phone detection model (simplified COCO-like detection)
        self.setup_phone_detection()

        # Load actual Office clip files
        clips_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'office-clips')
        self.office_clips = [
            os.path.join(clips_dir, f)
            for f in os.listdir(clips_dir)
            if f.lower().endswith(('.mov', '.mp4', '.avi', '.gif'))
        ]
        if self.office_clips:
            print(f"🎬 Loaded {len(self.office_clips)} Office clips:")
            for clip in self.office_clips:
                print(f"   • {os.path.basename(clip)}")
        else:
            print("⚠️  No clips found in assets/office-clips/")

        # Initialize pygame for sound
        pygame.mixer.init()

        print("🚔 Multi-Modal Office Clip Detector")
        print("👁️ Iris tracking + 📱 Phone detection + 🔍 Eye state")
        print("📊 Will calibrate to YOUR eye position first!")
        print("📱 Then: Looking down for 2+ seconds = SHAME CLIP!")

    def setup_phone_detection(self):
        """Setup YOLO phone detection"""
        try:
            from ultralytics import YOLO
            self.yolo = YOLO('yolov8n.pt')  # Downloads once (~6MB)
            self.yolo_enabled = True
            print("📱 Phone detection ready (YOLOv8)")
        except Exception as e:
            self.yolo = None
            self.yolo_enabled = False
            print(f"⚠️  YOLO unavailable: {e}")

    def detect_phone_object_yolo(self, frame):
        """Detect phone using YOLOv8 - class 67 = cell phone in COCO"""
        if not self.yolo_enabled:
            return False, []
        try:
            results = self.yolo(frame, verbose=False, conf=0.35)
            phone_boxes = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 67:  # COCO class 67 = cell phone
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        phone_boxes.append((x1, y1, x2 - x1, y2 - y1, f"yolo:{conf:.2f}"))
            return len(phone_boxes) > 0, phone_boxes
        except Exception as e:
            return False, []

    def detect_phone_object(self, frame):
        """Enhanced phone detection - avoid wall false positives"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        phone_candidates = []

        # Method 1: Edge detection (original - more restrictive)
        edges = cv2.Canny(gray, 40, 120)  # Slightly higher thresholds
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = ch / cw if cw > 0 else 0
            area = cv2.contourArea(contour)

            # More restrictive criteria for edges
            if (1.5 < aspect_ratio < 3.5 and     # Stricter aspect ratio
                800 < area < 15000 and           # Smaller max area (no walls!)
                30 < cw < 200 and               # Phone-sized width limits
                50 < ch < 400 and               # Phone-sized height limits
                x > w//10 and x < w*9//10 and    # Not at edges (avoid walls)
                y > h//10 and y < h*9//10):      # Not at edges (avoid walls)

                # Check if it's actually a rectangular object (not just noise)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if area / hull_area > 0.7:  # Should be reasonably rectangular
                    phone_candidates.append((x, y, cw, ch, "edge"))

        # Method 2: Template matching (more restrictive)
        templates_found = 0
        for width, height in [(60, 120), (80, 160), (50, 100)]:
            if width < w//6 and height < h//6:  # Smaller max size
                template = np.ones((height, width), dtype=np.uint8) * 200  # Lighter template
                cv2.rectangle(template, (3, 3), (width-4, height-4), 255, 2)

                res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(res >= 0.4)  # Higher threshold

                for pt in zip(*locations[::-1]):
                    # Avoid edge areas (where walls might be)
                    if (w//8 < pt[0] < w*7//8 and h//8 < pt[1] < h*7//8):
                        templates_found += 1
                        if templates_found <= 2:  # Limit template matches
                            phone_candidates.append((pt[0], pt[1], width, height, "template"))

        # Method 3: Color-based detection (MUCH more restrictive for white)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Much more specific white range (avoid walls)
        lower_white = np.array([0, 0, 200])    # Higher brightness threshold
        upper_white = np.array([180, 20, 255])  # Lower saturation threshold
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Morphological operations to clean up noise
        kernel = np.ones((3,3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        white_regions = 0
        for contour in white_contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = ch / cw if cw > 0 else 0
            area = cv2.contourArea(contour)

            # Very restrictive criteria for white regions
            if (1.3 < aspect_ratio < 3.8 and      # Phone aspect ratio
                1200 < area < 25000 and           # Much smaller max (no walls!)
                40 < cw < 150 and                 # Phone-width range
                60 < ch < 300 and                 # Phone-height range
                w//6 < x < w*5//6 and             # Center area only (no walls)
                h//6 < y < h*5//6):               # Center area only (no walls)

                # Additional check: white region should be compact
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if area / hull_area > 0.75:  # Should be very rectangular
                    white_regions += 1
                    if white_regions <= 1:  # Only one white region at a time
                        phone_candidates.append((x, y, cw, ch, "color"))

        # Much more aggressive duplicate removal
        filtered_candidates = []
        for i, (x1, y1, w1, h1, method1) in enumerate(phone_candidates):
            # Check if this region is reasonable for a phone in hands
            center_x, center_y = x1 + w1//2, y1 + h1//2

            # Phone should be in lower 2/3 of frame (hand-held region)
            if center_y < h//3:
                continue  # Skip objects in top third (likely not handheld)

            # Phone should not be too close to edges (likely wall/background)
            if (x1 < w//8 or x1 + w1 > w*7//8 or
                y1 < h//8 or y1 + h1 > h*7//8):
                continue

            is_duplicate = False
            for j, (x2, y2, w2, h2, method2) in enumerate(phone_candidates[i+1:], i+1):
                # Check overlap
                overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                if overlap_area > 0.2 * min(w1*h1, w2*h2):  # 20% overlap
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_candidates.append((x1, y1, w1, h1, method1))

        # Limit total phone detections (prevent wall spam)
        if len(filtered_candidates) > 2:
            filtered_candidates = filtered_candidates[:2]

        return len(filtered_candidates) > 0, filtered_candidates

    def detect_eye_state(self, landmarks):
        """Detect if eyes are open, closed, or looking down"""
        # Eye landmarks for eye opening detection
        left_eye = [landmarks[33], landmarks[7], landmarks[163], landmarks[144], landmarks[145], landmarks[153]]
        right_eye = [landmarks[362], landmarks[382], landmarks[381], landmarks[380], landmarks[374], landmarks[373]]

        def eye_aspect_ratio(eye_points):
            # Vertical distances
            A = ((eye_points[1].x - eye_points[5].x)**2 + (eye_points[1].y - eye_points[5].y)**2)**0.5
            B = ((eye_points[2].x - eye_points[4].x)**2 + (eye_points[2].y - eye_points[4].y)**2)**0.5
            # Horizontal distance
            C = ((eye_points[0].x - eye_points[3].x)**2 + (eye_points[0].y - eye_points[3].y)**2)**0.5

            return (A + B) / (2.0 * C + 1e-6)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # EAR thresholds
        eyes_closed = avg_ear < 0.2    # Eyes closed
        eyes_open = avg_ear > 0.25     # Eyes fully open

        return {
            'eyes_closed': eyes_closed,
            'eyes_open': eyes_open,
            'eye_aspect_ratio': avg_ear
        }

    def calculate_gaze_ratio(self, landmarks):
        """Head tilt detection using nose/eye/chin landmarks.
        Returns ratio: higher = more tilted down (looking at phone)"""
        nose   = landmarks[1]
        chin   = landmarks[152]
        l_eye  = landmarks[33]
        r_eye  = landmarks[263]

        # Vertical midpoint of eyes
        eye_mid_y = (l_eye.y + r_eye.y) / 2.0

        # Face height (eye to chin)
        face_h = chin.y - eye_mid_y + 1e-6

        # Nose distance below eyes, normalised by face height
        # When looking DOWN: nose appears higher in face → smaller ratio
        # When looking at SCREEN normally: nose is lower → larger ratio
        ratio = (nose.y - eye_mid_y) / face_h

        # Fake iris/left/right returns so rest of code doesn't break
        l_iris = landmarks[468] if len(landmarks) > 468 else nose
        r_iris = landmarks[473] if len(landmarks) > 473 else nose
        left  = [l_eye, l_eye]
        right = [r_eye, r_eye]

        return ratio, l_iris, r_iris, left, right

    def detect_looking_down(self, avg_ratio, eye_state, phone_detected):
        """Detection: phone in view OR iris deviated from baseline"""
        if self.is_calibrating:
            return False, "Calibrating..."

        reasons = []

        # Signal 1: Phone is visible in frame — strongest signal
        if phone_detected:
            reasons.append("Phone in view")

        # Signal 2: Iris ratio deviated DOWN from baseline (looking down = ratio drops)
        if self.baseline_ratio is not None:
            deviation = self.baseline_ratio - avg_ratio  # positive = looking down
            if deviation > 0.04:
                reasons.append(f"Iris down ({deviation:.3f} below baseline)")
        else:
            if avg_ratio < self.looking_down_threshold:
                reasons.append(f"Iris low ({avg_ratio:.3f})")

        # Trigger on ANY signal (phone alone is enough)
        is_detected = len(reasons) > 0

        return is_detected, " | ".join(reasons) if reasons else "No signal"

    def play_office_clip(self):
        """Play random Office clip in a popup via separate process (macOS thread-safe)"""
        if self.video_playing or not self.office_clips:
            return

        self.video_playing = True
        clip_path = random.choice(self.office_clips)
        print(f"🎬 Playing: {os.path.basename(clip_path)}")

        # Popup player script — runs in its own process so cv2 GUI works on macOS
        popup_script = f"""
import cv2, subprocess, sys

clip_path = {repr(clip_path)}
cap = cv2.VideoCapture(clip_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = int(1000 / fps)

audio = subprocess.Popen(['afplay', clip_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

cv2.namedWindow('GET BACK TO WORK', cv2.WINDOW_NORMAL)
cv2.resizeWindow('GET BACK TO WORK', 480, 360)
cv2.moveWindow('GET BACK TO WORK', 200, 150)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (480, 360))
    cv2.rectangle(frame, (0, 0), (480, 55), (0, 0, 180), -1)
    cv2.putText(frame, 'PUT THE PHONE DOWN!', (15, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2)
    cv2.imshow('GET BACK TO WORK', frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
audio.terminate()
"""

        import sys

        def launch():
            proc = subprocess.Popen(
                [sys.executable, '-c', popup_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            proc.wait()  # Wait for popup to close
            self.video_playing = False

        threading.Thread(target=launch, daemon=True).start()

    def stop_office_clip(self):
        """Stop the Office clip"""
        if not self.video_playing:
            return

        self.video_playing = False
        print("✅ Office clip stopped - back to work!")

        # Kill audio
        try:
            subprocess.run(['pkill', 'afplay'], check=False)
        except:
            pass

    def run(self):
        """Main detection loop - exact structure from Skyrim edition"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("📹 Starting webcam detection...")
        print("Press 'q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    # Calculate gaze ratio (exact from Skyrim edition)
                    avg_ratio, l_iris, r_iris, left, right = self.calculate_gaze_ratio(landmarks)

                    # Detect eye state (open/closed/looking down)
                    eye_state = self.detect_eye_state(landmarks)

                    # Phone detection via YOLO (every 500ms)
                    current_time = time.time()
                    if current_time - self.last_phone_check > self.phone_check_interval:
                        self.phone_detected, phone_candidates = self.detect_phone_object_yolo(frame)
                        self.last_phone_check = current_time

                        # Debug: Print detection results
                        if phone_candidates:
                            methods = [candidate[4] for candidate in phone_candidates]
                            print(f"📱 Phone detected via: {', '.join(set(methods))}")

                        # Draw phone detection results with different colors per method
                        if phone_candidates:
                            for (px, py, pw, ph, method) in phone_candidates:
                                # Different colors for different detection methods
                                if method == "edge":
                                    color = (255, 0, 255)  # Magenta for edge detection
                                    label = "Edge"
                                elif method == "template":
                                    color = (0, 255, 255)  # Cyan for template matching
                                    label = "Template"
                                elif method == "color":
                                    color = (255, 255, 0)  # Yellow for color detection
                                    label = "Color"
                                else:
                                    color = (128, 128, 128)  # Gray for unknown
                                    label = "Unknown"

                                cv2.rectangle(frame, (px, py), (px+pw, py+ph), color, 3)
                                cv2.putText(frame, f"📱{label}", (px, py-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Calibration phase - learn normal screen-looking position
                    if self.is_calibrating:
                        self.calibration_frames.append(avg_ratio)
                        self.calibration_count += 1

                        if self.calibration_count >= self.max_calibration_frames:
                            self.baseline_ratio = sum(self.calibration_frames) / len(self.calibration_frames)
                            self.is_calibrating = False
                            print(f"✅ Calibration complete! Your baseline ratio: {self.baseline_ratio:.3f}")
                            print(f"📊 Looking down threshold: {self.baseline_ratio + 0.15:.3f}")

                    # Multi-modal detection
                    is_looking_down, detection_reason = self.detect_looking_down(avg_ratio, eye_state, self.phone_detected)

                    # Trigger logic
                    now = time.time()
                    on_cooldown = (now - self.last_clip_time) < self.clip_cooldown

                    if is_looking_down:
                        if self.start_time is None:
                            self.start_time = now
                        elif now - self.start_time >= self.timer and not self.video_playing and not on_cooldown:
                            self.play_office_clip()
                            self.last_clip_time = now
                    else:
                        self.start_time = None
                        self.video_playing = False  # Reset so next detection can trigger

                    # Visual feedback (like original)
                    h, w = frame.shape[:2]

                    # Draw eye regions (green rectangles like original)
                    for eye_points in [left, right]:
                        for point in eye_points:
                            x = int(point.x * w)
                            y = int(point.y * h)
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                    # Draw iris points
                    for iris in [l_iris, r_iris]:
                        x = int(iris.x * w)
                        y = int(iris.y * h)
                        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                    # Status display
                    if self.is_calibrating:
                        status_color = (255, 255, 0)  # Yellow during calibration
                        status_text = f"CALIBRATING: {self.calibration_count}/{self.max_calibration_frames} | Ratio: {avg_ratio:.3f}"
                        cv2.putText(frame, "Look at your SCREEN normally", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    else:
                        status_color = (0, 0, 255) if is_looking_down else (0, 255, 0)
                        baseline_text = f" | Base: {self.baseline_ratio:.3f}" if self.baseline_ratio else ""
                        status_text = f"Down: {is_looking_down} | Ratio: {avg_ratio:.3f}{baseline_text}"

                        # Multi-modal status
                        eye_status = "👀" if eye_state['eyes_open'] else "👁️" if not eye_state['eyes_closed'] else "😴"
                        phone_status = "📱" if self.phone_detected else "❌"
                        indicators_text = f"{eye_status} EAR:{eye_state['eye_aspect_ratio']:.2f} | {phone_status} Phone | {detection_reason}"

                        cv2.putText(frame, indicators_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                    # Timer display
                    if self.start_time:
                        elapsed = time.time() - self.start_time
                        timer_text = f"Timer: {elapsed:.1f}/{self.timer}s"
                        cv2.putText(frame, timer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Video status
                    if self.video_playing:
                        cv2.rectangle(frame, (10, 10), (w-10, 100), (0, 0, 255), 3)
                        cv2.putText(frame, "OFFICE CLIP PLAYING!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.putText(frame, "Look up to stop!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Procrastination Police - Office Edition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                # Manual test: force play a clip
                print("🧪 Manual test triggered!")
                self.video_playing = False  # Reset so it can play
                self.play_office_clip()

        cap.release()
        cv2.destroyAllWindows()


def main():
    detector = OfficeClipDetector()
    detector.run()


if __name__ == "__main__":
    main()