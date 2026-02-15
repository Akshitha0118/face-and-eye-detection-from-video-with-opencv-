import cv2
import gradio as gr
import numpy as np

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_faces_and_eyes(frame):
    frame = np.array(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex+ew, ey+eh),
                (0, 255, 0),
                2
            )

    return frame

gr.Interface(
    fn=detect_faces_and_eyes,
    inputs=gr.Image(
        sources=["webcam"],
        streaming=True,
        type="numpy"
    ),
    outputs=gr.Image(type="numpy"),
    title="👁️ Live Face & Eye Detection",
    description="Real-time webcam detection using OpenCV + Gradio",
    live=True
).launch()
