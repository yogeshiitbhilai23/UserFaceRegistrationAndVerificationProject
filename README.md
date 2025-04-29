# 👤 Face Recognition System

This repository contains two Python scripts for a complete face recognition system with registration and verification capabilities.

## 📜 Scripts

### `face_registration.py`  
A GUI application for registering new users with facial recognition.

#### 🔧 Features
- ✔️ Captures live video from webcam  
- ✔️ Detects faces in real-time  
- ✔️ Stores user information (ID, name) along with facial encodings  
- ✔️ Saves face images for reference  
- ✔️ Validates user input before registration  
- ✔️ Prevents duplicate user IDs  

#### 🚀 Usage
1. Enter a unique **User ID** and **Full Name**  
2. Click **"Submit Details"**  
3. Position your face in the camera frame  
4. Click **"Capture Face"** when ready  
5. System will save the facial data and confirm registration  

#### 📁 Data Storage
- **Facial encodings:** `data/face_encodings.pkl`  
- **Face images:** `face_data/` directory  

---

### `face_verification.py`  
A GUI application for verifying registered users through facial recognition.

#### 🔧 Features
- ✔️ Real-time face detection and verification  
- ✔️ Displays verification status and confidence level  
- ✔️ Shows user information when matched  
- ✔️ Toggle between verification modes (on/off)  
- ✔️ Visual feedback with face bounding boxes  

#### 🚀 Usage
1. Click **"Start Verification"** to begin  
2. System will attempt to match detected faces with registered users  
3. Verification results are displayed in real-time  
4. Click **"Stop Verification"** to pause the process  

---

## 📦 Requirements
- Python 3.7+  
- `OpenCV` → `pip install opencv-python`  
- `face_recognition` → `pip install face_recognition`  
- `customtkinter` → `pip install customtkinter`  
- `Pillow` → `pip install pillow`  
- `NumPy` → `pip install numpy`  

## ⚙️ Installation

```bash
# Clone the Repository
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system

# Install Dependencies
pip install -r requirements.txt

# Run the Applications
# For registration:
python face_registration.py

# For verification:
python face_verification.py

###
face-recognition-system/
├── face_registration.py       # Registration application
├── face_verification.py       # Verification application
├── data/                     
│   └── face_encodings.pkl     # Database of facial encodings
├── face_data/                 # Directory for storing face images
├── README.md                  # This file
└── requirements.txt           # Python dependencies




### Key Formatting Elements Used:
1. **Headers:** `#`, `##`, `###` for hierarchy
2. **Lists:** `-` for unordered lists, `1.` for ordered lists
3. **Code Blocks:** ``` ``` for multi-line code, `` ` `` for inline code
4. **Emphasis:** `**bold**`, `*italic*`
5. **Horizontal Rule:** `---`
6. **Emojis:** For visual categorization
7. **File Tree:** Using plain text with alignment
8. **Special Notices:** ⚠️, 💡, 📌 with bold text
