"""
Procrastination Police - Multi-modal detection

Combines:
- Iris tracking (from Skyrim edition)
- Phone object detection (from doom-slayer)
- Eye state detection (open/closed)
"""

import sys
import cv2
import mediapipe as mp
import time
import random
import os
import subprocess
import pygame
import numpy as np
import threading
from collections import deque


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
        self.phone_check_interval = 0.2  # Check every 200ms

        # Temporal smoothing: track last N raw YOLO results
        # Phone "confirmed present"  = ≥2 of last 4 checks detected it
        # Phone "confirmed gone"     = 0 of last 3 checks detected it  (requires 3 consecutive misses)
        self.phone_history = deque(maxlen=4)
        self.phone_confirmed = False  # smoothed state used for triggering

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

        # Shuffle-based playlist so the same clip never repeats back-to-back
        self._clip_queue: list = []

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
                        h_frame = frame.shape[0]
                        center_y = (y1 + y2) // 2
                        if center_y > h_frame * 0.75:  # ignore bottom 25% (recording phone)
                            continue
                        if conf >= 0.12:
                            phone_boxes.append((x1, y1, x2 - x1, y2 - y1, f"yolo:{conf:.2f}"))
                            if self.debug_phone:
                                print(f"[YOLO] ✅ PHONE ACCEPTED  cls=67  conf={conf:.3f}")
                        else:
                            if self.debug_phone:
                                print(f"[YOLO] ⚠️  PHONE REJECTED  cls=67  conf={conf:.3f}  (below 0.12 threshold)")
                    elif cls_id == 65 and conf >= 0.35:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if (y1 + y2) // 2 > frame.shape[0] * 0.90:
                            continue
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
            if deviation > 0.025:
                reasons.append(f"Iris down ({deviation:.3f} below baseline)")
        else:
            if avg_ratio < self.looking_down_threshold:
                reasons.append(f"Iris low ({avg_ratio:.3f})")

        # Trigger on ANY signal (phone alone is enough)
        is_detected = len(reasons) > 0

        return is_detected, " | ".join(reasons) if reasons else "No signal"

    def _build_ordered_playlist(self):
        """Fixed order: Oh God → Stay Calm short → Where are turtles → Stay Calm long → repeat"""
        order_keywords = [
            "oh god please",       # "Oh God Please no edit.mov"
            "stay fucking calm short",  # short version second
            "where are the turtles",    # turtles third
            "stay fucking calm",        # full version fourth
        ]
        ordered = []
        for kw in order_keywords:
            match = next((c for c in self.office_clips if kw in os.path.basename(c).lower()), None)
            if match:
                ordered.append(match)
        # Append any clips not matched by keywords
        for c in self.office_clips:
            if c not in ordered:
                ordered.append(c)
        return ordered

    def _next_clip(self) -> str:
        """Return next clip in fixed rotation, looping forever."""
        if not self._clip_queue:
            self._clip_queue = self._build_ordered_playlist()
            print(f"[Clips] Playlist order: {[os.path.basename(c) for c in self._clip_queue]}")
        return self._clip_queue.pop(0)  # pop from front to preserve order

    def play_office_clip(self):
        """Play next Office clip (shuffle rotation) in a popup via separate process."""
        if self.video_playing or not self.office_clips:
            return

        self.video_playing = True
        clip_path = self._next_clip()
        print(f"🎬 Playing: {os.path.basename(clip_path)}")

        # Call the standalone popup_player.py — much faster startup than inline script
        popup_player = os.path.join(os.path.dirname(__file__), 'popup_player.py')

        def launch():
            proc = subprocess.Popen(
                [sys.executable, popup_player, clip_path],
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

    def draw_hud(self, frame, landmarks, phone_candidates, is_looking_down):
        """Aesthetic HUD: corner brackets on eyes + phone box"""
        h, w = frame.shape[:2]
        active = is_looking_down
        eye_color   = (0, 80, 255)   if active else (0, 220, 180)   # red-ish : teal
        phone_color = (0, 80, 255)   if active else (0, 200, 255)   # red-ish : cyan

        def corner_brackets(frame, x, y, bw, bh, color, arm=10, thickness=2):
            """Draw 4 L-shaped corner brackets instead of a full rectangle"""
            pts = [
                # top-left
                ((x, y + arm),        (x, y),        (x + arm, y)),
                # top-right
                ((x+bw-arm, y),       (x+bw, y),     (x+bw, y + arm)),
                # bottom-left
                ((x, y+bh-arm),       (x, y+bh),     (x + arm, y+bh)),
                # bottom-right
                ((x+bw-arm, y+bh),    (x+bw, y+bh),  (x+bw, y+bh-arm)),
            ]
            for p1, vertex, p2 in pts:
                cv2.line(frame, p1, vertex, color, thickness, cv2.LINE_AA)
                cv2.line(frame, vertex, p2, color, thickness, cv2.LINE_AA)

        # ── Eye brackets ──────────────────────────────────────────────
        # Use outer/inner eye corners to bound each eye
        eye_pairs = [
            (landmarks[33],  landmarks[133]),   # left eye:  outer → inner
            (landmarks[362], landmarks[263]),   # right eye: inner → outer
        ]
        for lm_a, lm_b in eye_pairs:
            px1 = int(min(lm_a.x, lm_b.x) * w)
            py1 = int(min(lm_a.y, lm_b.y) * h)
            px2 = int(max(lm_a.x, lm_b.x) * w)
            py2 = int(max(lm_a.y, lm_b.y) * h)
            pad_x, pad_y = max(10, (px2 - px1) // 2), max(8, (py2 - py1))
            corner_brackets(frame, px1 - pad_x, py1 - pad_y,
                            (px2 - px1) + pad_x * 2, (py2 - py1) + pad_y * 2,
                            eye_color, arm=8, thickness=2)

        # ── Phone box ─────────────────────────────────────────────────
        for (px, py, pw, ph, _) in phone_candidates:
            corner_brackets(frame, px, py, pw, ph, phone_color, arm=16, thickness=2)
            cv2.putText(frame, 'PHONE', (px, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, phone_color, 1, cv2.LINE_AA)

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

                    # Phone detection via YOLO (every 200ms)
                    phone_candidates = []  # reset each frame; updated inside gate
                    current_time = time.time()
                    if current_time - self.last_phone_check > self.phone_check_interval:
                        if self.debug_phone:
                            print(f"[YOLO] --- check at t={current_time:.2f} ---")
                        raw_detected, phone_candidates = self.detect_phone_object_yolo(frame)
                        self.last_phone_check = current_time

                        # Update smoothing history
                        self.phone_history.append(raw_detected)
                        hits = sum(self.phone_history)
                        history_len = len(self.phone_history)

                        # Confirm PRESENT: any single detection is enough
                        if hits >= 1:
                            self.phone_confirmed = True
                        # Confirm GONE: 0 of last 3 checks saw phone (requires 3 consecutive misses)
                        elif history_len >= 3 and hits == 0:
                            self.phone_confirmed = False

                        self.phone_detected = self.phone_confirmed

                        if self.debug_phone:
                            raw_str = f"{len(phone_candidates)} box(es)" if phone_candidates else "not detected"
                            print(f"[YOLO] raw={raw_detected} ({raw_str})  |  history={list(self.phone_history)}  hits={hits}  confirmed={self.phone_confirmed}")

                        # Stop clip only when phone is confirmed gone (not on a single miss)
                        if self.video_playing and not self.phone_confirmed:
                            self.stop_office_clip()


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

                    # HUD overlay — eyes + phone brackets
                    self.draw_hud(frame, landmarks, phone_candidates if self.phone_detected else [], is_looking_down)

            cv2.imshow('Procrastination Police - Office Edition', frame)
            cv2.moveWindow('Procrastination Police - Office Edition', 760, 100)

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