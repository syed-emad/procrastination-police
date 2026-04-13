"""Standalone popup video player — called as subprocess by office_detector.py"""
import sys
import subprocess
import cv2

clip_path = sys.argv[1]

cap = cv2.VideoCapture(clip_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = int(1000 / fps)

audio = subprocess.Popen(
    ['afplay', clip_path],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

cv2.namedWindow('GET BACK TO WORK', cv2.WINDOW_NORMAL)
cv2.resizeWindow('GET BACK TO WORK', 480, 360)
cv2.moveWindow('GET BACK TO WORK', 290, 160)

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
