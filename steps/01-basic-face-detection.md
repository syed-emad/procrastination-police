# Step 1: Basic Face Detection

## Goal
Set up MediaPipe face detection to identify and track the user's face in real-time using the webcam.

## What We're Building
A simple face tracker that:
- Opens webcam feed
- Detects face landmarks using MediaPipe
- Draws face mesh overlay for visualization
- Calculates basic face orientation

## Key Components
- MediaPipe Face Mesh for landmark detection
- OpenCV for camera handling and display
- Real-time face pose estimation

## Testing Criteria
✅ Webcam feed opens successfully
✅ Face is detected and landmarks are visible
✅ Face mesh overlay is drawn accurately
✅ Basic head pose angles are calculated and displayed

## Output
A working face tracker that shows:
- Live webcam feed
- Face landmark points
- Head rotation angles (pitch, yaw, roll)

## Next Step
Once face detection is solid, we'll use the landmarks to estimate gaze direction in Step 2.