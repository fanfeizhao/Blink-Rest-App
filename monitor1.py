import cv2
import mediapipe as mp
import time
import pandas as pd
from joblib import load
import app

FOCUS_SECONDS_NEEDED = 6.7 * 60  # 18 minutesbro

model = load("gaze_model.joblib")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

focused_time = 0.0
last_time = time.time()

def get_features(lm):
    left_x = lm[468].x
    right_x = lm[473].x
    left_y = lm[468].y
    right_y = lm[473].y

    left_outer = lm[33].x
    left_inner = lm[133].x
    left_top = lm[159].y
    left_bottom = lm[145].y

    right_outer = lm[362].x
    right_inner = lm[263].x
    right_top = lm[386].y
    right_bottom = lm[374].y

    left_width = abs(left_inner - left_outer)
    left_height = abs(left_bottom - left_top)
    right_width = abs(right_inner - right_outer)
    right_height = abs(right_bottom - right_top)

    if left_width < 1e-6 or left_height < 1e-6 or right_width < 1e-6 or right_height < 1e-6:
        return None

    return pd.DataFrame([{
        "left_ratio_x": abs(left_x - (left_outer + left_inner) / 2.0) / left_width,
        "right_ratio_x": abs(right_x - (right_outer + right_inner) / 2.0) / right_width,
        "left_ratio_y": abs(left_y - (left_top + left_bottom) / 2.0) / left_height,
        "right_ratio_y": abs(right_y - (right_top + right_bottom) / 2.0) / right_height
    }])

print("Monitoring screen focus...")
print("When screen focus reaches 18 minutes, popup will start.")
print("Press Control + C in Terminal to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    now = time.time()
    dt = now - last_time
    last_time = now

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    looking_screen = False

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        features = get_features(lm)

        if features is not None:
            pred = model.predict(features)[0]
            if pred == "screen":
                looking_screen = True

    if looking_screen:
        focused_time += dt

    minutes = focused_time / 60
    print(f"Focused screen time: {minutes:.1f} min", end="\r")

    if focused_time >= FOCUS_SECONDS_NEEDED:
        print("\n18 minutes reached. Starting break popup.")
        app.show_rest_popup()
        focused_time = 0.0
        last_time = time.time()

    time.sleep(0.1)