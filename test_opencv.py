#!/usr/bin/env python3
"""
Test OpenCV-only face detection - no MediaPipe dependencies

This should work reliably on macOS without any font/import issues.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from opencv_tracker import OpenCVFaceTracker


def test_opencv_face_detection():
    """Test OpenCV-only face detection"""
    print("🧪 OpenCV Face Detection Test")
    print("=" * 40)
    print("Using OpenCV built-in face detection")
    print("No MediaPipe = No font loading issues!")
    print()
    print("You'll see:")
    print("✅ Green box around face when looking forward")
    print("✅ Red box + alert when looking down (phone!)")
    print("✅ Eye detection (yellow boxes)")
    print("✅ Real-time detection reasoning")
    print()
    print("Try looking down like you're checking your phone!")
    print("Press 'q' in video window to quit")
    print("=" * 40)

    try:
        tracker = OpenCVFaceTracker()
        tracker.run()
        print("✅ OpenCV face detection test completed!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_opencv_face_detection()