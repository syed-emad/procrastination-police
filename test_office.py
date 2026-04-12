#!/usr/bin/env python3
"""
Test Office Detector - Using exact Skyrim edition logic

This uses the proven iris tracking algorithm from the Skyrim edition.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from office_detector import OfficeClipDetector


def test_office_detector():
    """Test the Office clip detector with exact Skyrim logic"""
    print("🎬 Office Clip Detector - Skyrim Edition Logic")
    print("=" * 60)
    print("✨ PROVEN ALGORITHM:")
    print("👁️ Iris tracking (landmarks 468, 473)")
    print("📏 Eye corners (145, 159, 374, 386)")
    print("🎯 Threshold: 0.25 (looking down)")
    print("⏰ Timer: 2.0 seconds sustained")
    print("🔄 Debounce: 0.45 (to stop clip)")
    print()
    print("🎭 OFFICE CLIPS INCLUDED:")
    print("• 'No... god no... NOOOO!'")
    print("• 'That's what she said!'")
    print("• 'Bears. Beets. Battlestar Galactica.'")
    print("• 'I DECLARE BANKRUPTCY!'")
    print("• And 6 more...")
    print()
    print("🚀 HOW TO TEST:")
    print("1. Look at screen normally (green status)")
    print("2. Look down like checking phone for 2+ seconds")
    print("3. Random Office clip plays with speech!")
    print("4. Look up to stop the clip")
    print()
    print("🎮 CONTROLS:")
    print("'q' = Quit")
    print()
    print("🔊 Note: Includes text-to-speech on macOS/Windows/Linux")
    print("=" * 60)

    try:
        detector = OfficeClipDetector()
        detector.run()
        print("✅ Office detector test completed!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_office_detector()