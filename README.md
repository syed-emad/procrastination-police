# Procrastination Police 👮‍♂️📱

A computer vision app that catches you looking at your phone while working and shame-reminds you with Office clips to get back to work.

## How It Works
1. Uses your laptop's webcam + MediaPipe to track face/gaze direction
2. Detects when you're looking down/away from screen (toward phone)
3. Triggers a random Office clip popup telling you to focus
4. Tracks your "procrastination stats" for shame accountability

## Project Structure
```
procrastination-police/
├── src/
│   ├── face_tracker.py      # MediaPipe face detection
│   ├── gaze_estimator.py    # Gaze direction calculation
│   ├── shame_popup.py       # Office clip popup system
│   └── main.py             # Main app runner
├── steps/                   # Implementation steps
├── assets/office-clips/     # Place .mp4/.gif files here
├── config/
│   └── settings.yaml       # App configuration
└── requirements.txt
```

## Quick Start
1. Install dependencies: `uv sync`
2. Add Office clips to `assets/office-clips/`
3. Run: `python src/main.py`

## Implementation Progress
- [ ] Step 1: Basic face detection
- [ ] Step 2: Gaze estimation
- [ ] Step 3: Phone detection logic
- [ ] Step 4: Shame popup system
- [ ] Step 5: Office clips integration