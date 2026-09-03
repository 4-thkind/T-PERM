#!/bin/bash
set -e

echo "=== AR Rubik's Cube Setup ==="

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt -q

# Download MediaPipe hand landmarker model
if [ ! -f "hand_landmarker.task" ]; then
    echo "Downloading MediaPipe hand landmarker model..."
    wget -q -O hand_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    echo "Model downloaded."
fi

echo ""
echo "Setup complete."
echo "Run with: source venv/bin/activate && python main.py"
