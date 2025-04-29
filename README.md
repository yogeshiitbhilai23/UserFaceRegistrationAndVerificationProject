Face Recognition System
This repository contains two Python scripts for a complete face recognition system with registration and verification capabilities.

Scripts
1. face_registration.py
A GUI application for registering new users with facial recognition.

Features:
Captures live video from webcam

Detects faces in real-time

Stores user information (ID, name) along with facial encodings

Saves face images for reference

Validates user input before registration

Prevents duplicate user IDs

Usage:
Enter a unique User ID and Full Name

Click "Submit Details"

Position your face in the camera frame

Click "Capture Face" when ready

System will save the facial data and confirm registration

Data Storage:
Facial encodings are stored in data/face_encodings.pkl

Face images are saved in the face_data/ directory

2. face_verification.py
A GUI application for verifying registered users through facial recognition.

Features:
Real-time face detection and verification

Displays verification status and confidence level

Shows user information when matched

Toggle between verification modes (on/off)

Visual feedback with face bounding boxes

Usage:
Click "Start Verification" to begin

System will attempt to match detected faces with registered users

Verification results are displayed in real-time

Click "Stop Verification" to pause the process

Requirements
Python 3.7+

OpenCV (pip install opencv-python)

face_recognition (pip install face_recognition)

customtkinter (pip install customtkinter)

Pillow (pip install pillow)

NumPy (pip install numpy)

Installation
Clone this repository:

bash
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
Install the required packages:

bash
pip install -r requirements.txt
Run the applications:

For registration: python face_registration.py

For verification: python face_verification.py

File Structure
face-recognition-system/
│
├── face_registration.py       # Registration application
├── face_verification.py       # Verification application
├── data/                      # Directory for storing facial encodings
│   └── face_encodings.pkl     # Database of facial encodings
├── face_data/                 # Directory for storing face images
├── README.md                  # This file
└── requirements.txt           # Python dependencies
Notes
The system uses dlib's face recognition model which is quite accurate but may be computationally intensive

For better performance, ensure good lighting conditions during registration and verification

The system is designed for single-user verification at a time (not optimized for multiple simultaneous faces)
