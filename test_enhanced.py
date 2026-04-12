#!/usr/bin/env python3
"""
Test Enhanced Phone Detection

Multiple detection methods for accurate phone usage tracking.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_tracker import EnhancedFaceTracker


def test_enhanced_detection():
    """Test enhanced phone detection"""
    print("🚔 Enhanced Phone Detection Test")
    print("=" * 50)
    print("NEW DETECTION METHODS:")
    print("📱 Phone object detection (purple boxes)")
    print("✋ Hand/object detection (cyan boxes)")
    print("📏 Face proximity analysis (closer = phone)")
    print("👁️ Advanced eye position tracking")
    print("📊 Stability filtering (reduces false positives)")
    print("📈 Multi-factor scoring system")
    print()
    print("WHAT TO TRY:")
    print("1. Hold your phone and look at it")
    print("2. Move phone closer to your face")
    print("3. Look down at phone in different positions")
    print("4. Normal screen looking (should stay green)")
    print()
    print("INDICATORS:")
    print("🔴 Red = Phone detected!")
    print("🟢 Green = Focused on work")
    print("📈 Score = Detection confidence")
    print("🟢🔴🟢🔴 History = Recent detection pattern")
    print()
    print("Press 'q' in video window to quit")
    print("=" * 50)

    try:
        tracker = EnhancedFaceTracker()
        tracker.run()
        print("✅ Enhanced phone detection test completed!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_detection()