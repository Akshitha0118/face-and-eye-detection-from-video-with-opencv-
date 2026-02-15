## 📌 Project Overview

This project implements a real-time face and eye detection system using OpenCV Haar Cascade Classifiers integrated with a Gradio web interface for live webcam streaming.

The system detects faces and eyes instantly from webcam input and displays bounding boxes around them. It demonstrates practical implementation of classical computer vision techniques combined with modern interactive UI tools.

## 🚀 Features

🎥 Real-time webcam face detection

👁️ Eye detection inside detected faces

⚡ Fast and efficient using Haar Cascade classifiers

🌐 Interactive Gradio web interface

🧠 Uses classical Computer Vision (no deep learning required)

💻 Runs locally in browser

## 🛠️ Technologies Used

Python

OpenCV

NumPy

Gradio

Haar Cascade Classifiers


## ⚙️ Installation
### Step 1: Clone the repository
git clone https://github.com/your-username/face-eye-detection.git
cd face-eye-detection

### Step 2: Install dependencies
pip install opencv-python gradio numpy

### ▶️ Usage

Run the application:

python app.py


Then open the local Gradio URL in your browser.

Allow webcam access to start detection.

## 🧠 How It Works
### 1. Capture Webcam Frame

Gradio captures live webcam frames and sends them to the function.

### 2. Convert to Grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


Grayscale improves detection speed and accuracy.

### 3. Detect Faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

### 4. Detect Eyes inside Face Region
eyes = eye_cascade.detectMultiScale(roi_gray)

### 5. Draw Bounding Boxes

Blue → Face

Green → Eyes

## 📸 Output Example

Detects face in real-time

Detects eyes within face

Draws bounding rectangles

Displays output in browser

## 🎯 Applications

Face recognition preprocessing

Attendance systems

Driver monitoring systems

Security and surveillance

Computer vision learning projects

## 🔮 Future Improvements

Add face recognition

Add emotion detection

Add multiple face tracking

Deploy on cloud
