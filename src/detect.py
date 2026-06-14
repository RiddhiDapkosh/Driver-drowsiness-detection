import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import winsound  # 🔔 Works on Windows

# ✅ Fix base path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Correct paths
model_path = os.path.join(base_dir, 'models', 'drowsiness_model.keras')
cascade_path = os.path.join(base_dir, 'haarcascade', 'haarcascade_frontalface_default.xml')

# ✅ Debug check (optional but useful)
print("Model Path:", model_path)
print("Model Exists:", os.path.exists(model_path))

# Load model
model = load_model(model_path)

# Load face detector
face_cascade = cv2.CascadeClassifier(cascade_path)

# ✅ Class labels (must match training order)
class_labels = ['close', 'no_yawn', 'open', 'yawn']

# Start webcam
cap = cv2.VideoCapture(0)

counter = 0  # for alarm delay

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 3))

        pred = model.predict(face, verbose=0)
        label_index = np.argmax(pred)
        label = class_labels[label_index]
        confidence = np.max(pred)

        # 🚨 Drowsy logic
        if label in ['close', 'yawn']:
            status = "DROWSY 😴"
            color = (0, 0, 255)
            counter += 1

            if counter > 10:
                winsound.Beep(1000, 500)

        else:
            status = "AWAKE 🙂"
            color = (0, 255, 0)
            counter = 0

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Show text
        text = f"{status} ({label}) {confidence*100:.1f}%"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Cascade Path:", cascade_path)
print("Exists:", os.path.exists(cascade_path))