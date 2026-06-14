import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import time
import base64

# ---------------- LOAD MODEL ----------------
model = load_model("models/drowsiness_model.keras", compile=False)

# ---------------- CLASSES ----------------
classes = ['close', 'open', 'yawn', 'no_yawn']

# ---------------- STREAMLIT UI ----------------
st.title("Driver Drowsiness Detection System")
st.markdown("Real-time detection of eye state + yawning")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
alert_placeholder = st.empty()

# ---------------- 🔊 BEEP SOUND (NO FILE NEEDED) ----------------
def play_beep():
    beep_base64 = """
    UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQgAAAAA/////wAAAP///wAA
    AAAAAAA=
    """
    audio_bytes = base64.b64decode(beep_base64)
    st.audio(audio_bytes, format="audio/wav", autoplay=True)

# ---------------- CAMERA ----------------
cap = None

if run:
    cap = cv2.VideoCapture(0)

    cooldown = 2
    last_alert_time = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found")
            break

        # ---------------- PREPROCESS ----------------
        img = cv2.resize(frame, (64, 64))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # ---------------- PREDICTION ----------------
        pred = model.predict(img, verbose=0)[0]
        class_index = np.argmax(pred)
        confidence = np.max(pred)

        label = classes[class_index]

        # ---------------- SHOW TEXT ----------------
        cv2.putText(frame,
                    f"{label} ({confidence:.2f})",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # ---------------- ALERT LOGIC ----------------
        current_time = time.time()
        alert_message = ""

        if confidence > 0.6:
            if label == "close":
                alert_message = "⚠️ DROWSINESS ALERT: Eyes Closed!"
            elif label == "yawn":
                alert_message = "⚠️ FATIGUE ALERT: Driver Yawning!"

        # ---------------- ALERT + SOUND ----------------
        if alert_message and (current_time - last_alert_time > cooldown):
            alert_placeholder.error(alert_message)

            # 🔊 PLAY BEEP SOUND
            play_beep()

            last_alert_time = current_time
        else:
            alert_placeholder.success("Driver is alert")

        # ---------------- DISPLAY FRAME ----------------
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()