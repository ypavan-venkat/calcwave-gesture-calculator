CalcWave is a real-time touchless calculator built using computer vision. It allows users to perform arithmetic operations using hand pinch gestures captured via webcam.

The project demonstrates real-time human-computer interaction using AI-powered hand tracking.

🛠 Tech Stack

Python

OpenCV

MediaPipe

AST (Safe Expression Evaluation)

🎮 How It Works

Index finger → Cursor movement

Thumb + Index pinch → Button click

Secure arithmetic evaluation (No unsafe eval)

ESC → Exit application

If MediaPipe is unavailable, the calculator automatically switches to mouse-click mode.

▶ Installation
pip install opencv-python mediapipe

▶ Run
python calcwave.py

🚀 Features

Real-time hand tracking

Pinch-based click detection

Hover highlighting

Cooldown logic to prevent double clicks

Secure expression parse
