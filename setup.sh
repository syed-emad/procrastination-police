#!/bin/bash
# Setup script for Procrastination Police

echo "🚔 Setting up Procrastination Police..."

# Create virtual environment and install dependencies
echo "📦 Installing dependencies with uv..."
cd "$(dirname "$0")"

# Create virtual environment
uv venv

# Install dependencies directly
uv pip install mediapipe opencv-python numpy PyYAML pillow python-dotenv

echo "🎥 Testing webcam access..."
echo "About to test face detection. Press 'q' to quit the test."
echo "Press Enter to continue..."
read

# Activate virtual environment and run the test
source .venv/bin/activate
python test_step1.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "📁 Project structure:"
echo "├── src/face_tracker.py       # Face detection (Step 1 ✅)"
echo "├── assets/office-clips/      # Put your Office clips here!"
echo "├── config/settings.yaml     # Configuration"
echo "└── steps/                   # Next implementations"
echo ""
echo "🎬 Next: Add Office clips to assets/office-clips/ folder"
echo "🔨 Then: Run Step 2 implementation for gaze estimation"
echo ""
echo "💡 To run manually: source .venv/bin/activate && python test_step1.py"