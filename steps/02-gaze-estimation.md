# Step 2: Gaze Estimation

## Goal
Improve the basic face detection to accurately estimate gaze direction and define "phone zones" for more precise detection.

## What We're Building
Enhanced detection that:
- Uses facial landmarks to estimate eye gaze direction
- Defines configurable "zones" (phone area, screen area)  
- Tracks gaze duration before triggering
- Reduces false positives from head movement

## Key Improvements
- Eye landmark analysis for gaze estimation
- Temporal filtering (must look at phone for X seconds)
- Configurable detection zones based on user setup
- Calibration mode to set personal "phone zone"

## Testing Criteria
✅ More accurate phone detection (fewer false positives)
✅ Gaze direction vector calculation
✅ Zone-based detection with visual feedback
✅ Temporal filtering (trigger delay) working

## Output
A refined gaze tracker that:
- Shows gaze direction vector overlay
- Highlights active detection zones
- Only triggers after sustained phone-looking
- Has user-configurable sensitivity

## Next Step
Step 3 will implement the Office clip popup system when phone-looking is detected.