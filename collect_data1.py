import os
import cv2
import mediapipe as mp
import csv

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
current_label = "none"

file_exists = os.path.exists("data.csv")
file_empty = (not file_exists) or os.path.getsize("data.csv") == 0

file = open("data.csv", "a", newline="")
writer = csv.writer(file)

if file_empty:
    writer.writerow([
        "left_ratio_x",
        "right_ratio_x",
        "left_ratio_y",
        "right_ratio_y",
        "label"
    ])

print("Press S = screen, A = away, C = closed, Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # Iris positions
        left_x = lm[468].x
        right_x = lm[473].x
        left_y = lm[468].y
        right_y = lm[473].y

        # Eye boundaries
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

        if (
            current_label != "none"
            and left_width > 1e-6
            and left_height > 1e-6
            and right_width > 1e-6
            and right_height > 1e-6
        ):
            left_ratio_x = abs(left_x - (left_outer + left_inner) / 2.0) / left_width
            right_ratio_x = abs(right_x - (right_outer + right_inner) / 2.0) / right_width
            left_ratio_y = abs(left_y - (left_top + left_bottom) / 2.0) / left_height
            right_ratio_y = abs(right_y - (right_top + right_bottom) / 2.0) / right_height

            writer.writerow([
                left_ratio_x,
                right_ratio_x,
                left_ratio_y,
                right_ratio_y,
                current_label
            ])

    cv2.imshow("Collecting Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        current_label = "screen"
        print("SCREEN")
    elif key == ord("a"):
        current_label = "away"
        print("AWAY")
    elif key == ord("c"):
        current_label = "closed"
        print("CLOSED")
    elif key == ord("q"):
        break

cap.release()
file.close()
cv2.destroyAllWindows()