#!/usr/bin/env python3
"""
Simple test for Face Detection - avoiding macOS font issues

Just run: python test_simple.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simple_face_tracker import SimpleFaceTracker


def test_simple_face_detection():
    """Test simplified face detection"""
    print("🧪 Simple Face Detection Test")
    print("=" * 40)
    print("This should work without hanging!")
    print("You'll see:")
    print("✅ Face outline (not full mesh)")
    print("✅ Head angle estimation")
    print("✅ Phone detection when looking down")
    print()
    print("Press 'q' in the video window to quit")
    print("=" * 40)

    try:
        tracker = SimpleFaceTracker()
        tracker.run()
        print("✅ Simple face detection test completed!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_simple_face_detection()