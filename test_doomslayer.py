#!/usr/bin/env python3
"""
Test Doom-Slayer Inspired Detection

Uses proven algorithms from the doom-slayer project for accurate detection.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from doomslayer_detector import DoomSlayerDetector


def test_doomslayer_detection():
    """Test doom-slayer inspired detection"""
    print("🎯 Doom-Slayer Procrastination Detection")
    print("=" * 50)
    print("Based on proven algorithms from doom-slayer project!")
    print()
    print("🔬 HOW IT WORKS:")
    print("📊 Calibration Phase - Learns YOUR normal behavior")
    print("📈 Multi-factor scoring - Combines multiple signals")
    print("🎯 Sustained detection - Requires 3 consecutive frames")
    print("⚙️ Adaptive thresholds - Personalized to your baseline")
    print()
    print("🚀 GETTING STARTED:")
    print("1. Look at your screen normally during calibration")
    print("2. After calibration, try looking at your phone")
    print("3. System will detect sustained doomscrolling behavior")
    print()
    print("🎮 CONTROLS:")
    print("'q' = Quit")
    print("'r' = Reset calibration and start over")
    print()
    print("📊 The system learns YOUR specific patterns!")
    print("=" * 50)

    try:
        detector = DoomSlayerDetector()
        detector.run()
        print("✅ Doom-slayer detection test completed!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_doomslayer_detection()