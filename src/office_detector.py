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
        self.video_proc = None  # subprocess handle so we can kill it
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
        self.debug_phone = True  # Set False to silence verbose YOLO logs
        try:
            from ultralytics import YOLO
            self.yolo = YOLO('yolov8n.pt')  # Downloads once (~6MB)
            self.yolo_enabled = True
            print("📱 Phone detection ready (YOLOv8)")
            print(f"   conf threshold: 0.35 | class 67 = cell phone")
        except Exception as e:
            self.yolo = None
            self.yolo_enabled = False
            print(f"⚠️  YOLO unavailable: {e}")

    def detect_phone_object_yolo(self, frame):
        """Detect phone using YOLOv8 - class 67 = cell phone in COCO"""
        if not self.yolo_enabled:
            return False, []
        try:
            # Run with a low conf floor so we can see near-misses in the logs
            results = self.yolo(frame, verbose=False, conf=0.10)
            phone_boxes = []
            all_detections = []  # every detection this frame for debug

            for r in results:
                for box in r.boxes:
                    cls_id  = int(box.cls[0])
                    conf    = float(box.conf[0])
                    label   = self.yolo.names.get(cls_id, str(cls_id))
                    all_detections.append((cls_id, label, conf))

                    if cls_id == 67:  # cell phone
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if conf >= 0.20:
                            phone_boxes.append((x1, y1, x2 - x1, y2 - y1, f"yolo:{conf:.2f}"))
                            if self.debug_phone:
                                print(f"[YOLO] ✅ PHONE ACCEPTED  cls=67  conf={conf:.3f}")
                        else:
                            if self.debug_phone:
                                print(f"[YOLO] ⚠️  PHONE REJECTED  cls=67  conf={conf:.3f}  (below 0.20 threshold)")
                    elif cls_id == 65 and conf >= 0.35:
                        # YOLOv8n frequently misclassifies phones as "remote" — treat as phone
                        # (no actual remote in this setup, so all remote detections = phone)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        phone_boxes.append((x1, y1, x2 - x1, y2 - y1, f"yolo-remote:{conf:.2f}"))
                        if self.debug_phone:
                            print(f"[YOLO] ✅ REMOTE→PHONE  cls=65  conf={conf:.3f}  (phone misclassified as remote)")

            if self.debug_phone:
                if all_detections:
                    # Show top-5 by confidence so logs don't spam on busy frames
                    top = sorted(all_detections, key=lambda d: d[2], reverse=True)[:5]
                    top_str = "  |  ".join(f"{label}({conf:.2f})" for _, label, conf in top)
                    print(f"[YOLO] top detections: {top_str}")
                else:
                    print("[YOLO] no detections above 0.10 this frame")

            return len(phone_boxes) > 0, phone_boxes
        except Exception as e:
            print(f"[YOLO] error: {e}")
            return False, []

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
            self.video_proc = proc
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
            self.video_proc = None
            self.video_playing = False

        threading.Thread(target=launch, daemon=True).start()

    def stop_office_clip(self):
        """Stop the Office clip and kill the popup process"""
        if not self.video_playing:
            return

        self.video_playing = False
        self.start_time = None  # force timer to restart from scratch on next detection
        self.last_clip_time = time.time() - self.clip_cooldown + 3  # 3s buffer before re-trigger
        print("📵 Phone gone — clip stopped!")

        # Kill the popup video window
        if self.video_proc and self.video_proc.poll() is None:
            self.video_proc.kill()
            self.video_proc = None

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
                        if self.debug_phone:
                            print(f"[YOLO] --- check at t={current_time:.2f} ---")
                        self.phone_detected, phone_candidates = self.detect_phone_object_yolo(frame)
                        self.last_phone_check = current_time

                        if self.debug_phone:
                            result_str = f"{len(phone_candidates)} box(es)" if phone_candidates else "not detected"
                            print(f"[YOLO] phone_detected={self.phone_detected}  ({result_str})")

                        # Stop clip immediately if phone left the frame
                        if self.video_playing and not self.phone_detected:
                            self.stop_office_clip()

                        # Draw phone detection results with different colors per method
                        if phone_candidates:
                            for (px, py, pw, ph, method) in phone_candidates:
                                # Different colors for different detection methods
                                if method.startswith("yolo"):
                                    color = (0, 165, 255)  # Orange for YOLO
                                    label = f"Phone {method.split(':')[1]}"
                                else:
                                    color = (128, 128, 128)
                                    label = method

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
                            print(f"✅ Calibration complete! Baseline: {self.baseline_ratio:.3f}")
                            print(f"📊 Triggers when ratio < {self.baseline_ratio - 0.04:.3f} (head tilt) or phone detected")

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

                    # Draw head-tilt landmarks: eye centers (green) and nose (blue)
                    for point in [left[0], right[0]]:
                        cv2.circle(frame, (int(point.x * w), int(point.y * h)), 4, (0, 255, 0), -1)
                    cv2.circle(frame, (int(l_iris.x * w), int(l_iris.y * h)), 4, (255, 100, 0), -1)

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