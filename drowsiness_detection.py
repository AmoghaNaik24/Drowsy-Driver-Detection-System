import cv2
import os
import numpy as np
import time
from pygame import mixer
from tensorflow.keras.models import load_model

# Initialize the mixer for sound alerts
mixer.init()

# Define file paths
ALARM_PATH = 'alarm.wav'
MODEL_PATH = 'models/cnnCat2.h5'
CASCADE_PATH = 'haar cascade files/'

# Check if alarm file exists
if os.path.exists(ALARM_PATH):
    sound = mixer.Sound(ALARM_PATH)
else:
    print("⚠️ Warning: alarm.wav file not found!")

# Load Haar cascade classifiers
face_cascade = os.path.join(CASCADE_PATH, 'haarcascade_frontalface_alt.xml')
left_eye_cascade = os.path.join(CASCADE_PATH, 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = os.path.join(CASCADE_PATH, 'haarcascade_righteye_2splits.xml')

# Verify if Haar cascade files exist
if not all(map(os.path.exists, [face_cascade, left_eye_cascade, right_eye_cascade])):
    raise FileNotFoundError("⚠️ One or more Haar cascade files are missing!")

face = cv2.CascadeClassifier(face_cascade)
leye = cv2.CascadeClassifier(left_eye_cascade)
reye = cv2.CascadeClassifier(right_eye_cascade)

# Verify if the trained model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"⚠️ Model file not found: {MODEL_PATH}")

# Load the trained deep learning model
model = load_model(MODEL_PATH, compile=False)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("⚠️ Could not access the webcam!")

# Initialize variables
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
rpred, lpred = [99], [99]

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame from webcam. Exiting...")
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Process right eye
    for (x, y, w, h) in right_eye:
        r_eye = gray[y:y+h, x:x+w]
        r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        break

    # Process left eye
    for (x, y, w, h) in left_eye:
        l_eye = gray[y:y+h, x:x+w]
        l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        break

    # Determine if eyes are closed
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = max(0, score - 1)  # Prevent score from going negative
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f'Score: {score}', (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Trigger alarm if drowsiness detected
    if score > 15:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        if os.path.exists(ALARM_PATH):
            try:
                sound.play()
            except:
                print("⚠️ Error: Unable to play alarm sound!")

        thicc = min(thicc + 2, 16) if thicc < 16 else max(thicc - 2, 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
