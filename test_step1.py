#!/usr/bin/env python3
"""
Test script for Step 1: Basic Face Detection

Run this to verify face tracking works before proceeding to next steps.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from face_tracker import FaceTracker


def test_face_detection():
    """Test basic face detection functionality"""
    print("🧪 Testing Face Detection - Step 1")
    print("=" * 50)
    print("This will open your webcam and show face detection.")
    print("You should see:")
    print("✅ Face mesh overlay on your face")
    print("✅ Head pose angles (pitch, yaw, roll)")
    print("✅ Status indicator when looking down (phone detection)")
    print()
    print("Press 'q' to quit the test")
    print("=" * 50)

    try:
        tracker = FaceTracker()
        tracker.run()
        print("✅ Face detection test completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    test_face_detection()