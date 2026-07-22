import cv2
import mediapipe as mp
import time
import math
import tkinter as tk
import pandas as pd
from joblib import load

EYE_CLOSED_THRESHOLD = 0.20
REQUIRED_REST_SECONDS = 20

model = load("gaze_model.joblib")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye_points, landmarks, w, h):
    pts = []
    for idx in eye_points:
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))

    v1 = distance(pts[1], pts[5])
    v2 = distance(pts[2], pts[4])
    hor = distance(pts[0], pts[3])

    if hor == 0:
        return 0.0
    return (v1 + v2) / (2.0 * hor)


def rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    points = [
        x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
        x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
        x1, y2, x1, y2-r, x1, y1+r, x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


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

    left_ratio_x = abs(left_x - (left_outer + left_inner) / 2.0) / left_width
    right_ratio_x = abs(right_x - (right_outer + right_inner) / 2.0) / right_width
    left_ratio_y = abs(left_y - (left_top + left_bottom) / 2.0) / left_height
    right_ratio_y = abs(right_y - (right_top + right_bottom) / 2.0) / right_height

    return pd.DataFrame([{
        "left_ratio_x": left_ratio_x,
        "right_ratio_x": right_ratio_x,
        "left_ratio_y": left_ratio_y,
        "right_ratio_y": right_ratio_y
    }])


def show_rest_popup():
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 1.0)
    root.configure(bg="#15151f")
    root.protocol("WM_DELETE_WINDOW", lambda: None)

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    canvas = tk.Canvas(root, width=sw, height=sh, bg="#15151f", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    # background
    canvas.create_rectangle(0, 0, sw, sh, fill="#15151f", outline="")
    canvas.create_oval(-250, -200, 500, 500, fill="#2b2033", outline="")
    canvas.create_oval(sw - 500, sh - 400, sw + 200, sh + 200, fill="#311f2a", outline="")

    # main card
    card_w = int(sw * 0.72)
    card_h = int(sh * 0.74)
    x1 = (sw - card_w) // 2
    y1 = (sh - card_h) // 2
    x2 = x1 + card_w
    y2 = y1 + card_h

    rounded_rect(canvas, x1, y1, x2, y2, 45, fill="#1d1d2b", outline="#ff6b6b", width=2)

    canvas.create_text(sw // 2, y1 + 65, text="🌱 BlinkRest", fill="#ffffff",
                       font=("Arial", 34, "bold"))
    canvas.create_text(sw // 2, y1 + 108, text="Rest your eyes. Recharge your focus.",
                       fill="#c8c8d8", font=("Arial", 17))

    cx = sw // 2
    cy = y1 + 285
    radius = 150

    # circle parts
    canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius,
                       outline="#3a3045", width=18)

    progress_arc = canvas.create_arc(cx-radius, cy-radius, cx+radius, cy+radius,
                                     start=90, extent=0, style="arc",
                                     outline="#ff6b6b", width=18)

    spinner_arc = canvas.create_arc(cx-radius-20, cy-radius-20, cx+radius+20, cy+radius+20,
                                    start=0, extent=55, style="arc",
                                    outline="#ffb3b3", width=7)

    timer_text = canvas.create_text(cx, cy - 15, text="20", fill="#ffffff",
                                    font=("Arial", 92, "bold"))
    seconds_text = canvas.create_text(cx, cy + 70, text="seconds", fill="#ffb3b3",
                                      font=("Arial", 22, "bold"))

    status_box = rounded_rect(canvas, cx - 360, cy + 190, cx + 360, cy + 300,
                              25, fill="#2a2029", outline="#ff6b6b", width=1)

    status_title = canvas.create_text(cx, cy + 225, text="Don’t stare at the screen",
                                      fill="#ff6b6b", font=("Arial", 26, "bold"))
    status_sub = canvas.create_text(cx, cy + 265, text="Look away or close your eyes to continue.",
                                    fill="#dddddd", font=("Arial", 17))

    # instruction cards
    bottom_y = y2 - 115
    card_gap = 25
    small_w = 260
    small_h = 115
    start_x = cx - small_w * 1.5 - card_gap

    labels = [
        ("👁", "Looking at screen?", "Timer pauses"),
        ("👀", "Looking away?", "Keep looking away"),
        ("😌", "Eyes closed?", "Keep them closed"),
    ]

    for i, (icon, title, desc) in enumerate(labels):
        sx = start_x + i * (small_w + card_gap)
        rounded_rect(canvas, sx, bottom_y, sx + small_w, bottom_y + small_h,
                     22, fill="#242434", outline="#34344a", width=1)
        canvas.create_text(sx + small_w//2, bottom_y + 28, text=icon,
                           fill="#ffffff", font=("Arial", 25))
        canvas.create_text(sx + small_w//2, bottom_y + 63, text=title,
                           fill="#ff8585" if i == 0 else "#78e0c2" if i == 1 else "#a6a6ff",
                           font=("Arial", 16, "bold"))
        canvas.create_text(sx + small_w//2, bottom_y + 92, text=desc,
                           fill="#d0d0d0", font=("Arial", 13))

    cap = cv2.VideoCapture(0)

    accumulated = 0.0
    last_good_time = None
    should_close = False
    spin_angle = 0

    def exit_animation(alpha=1.0):
        if alpha <= 0:
            try:
                root.quit()
            except Exception:
                pass
            return

        root.attributes("-alpha", alpha)
        root.after(30, lambda: exit_animation(alpha - 0.08))

    def close_popup(event=None):
        nonlocal should_close
        should_close = True
        exit_animation()

    root.bind("<Escape>", close_popup)

    def update():
        nonlocal accumulated, last_good_time, should_close, spin_angle

        if should_close:
            return

        ret, frame = cap.read()
        if not ret:
            root.after(50, update)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        eyes_closed = False
        looking_away = False
        looking_screen = False

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            left_ear = eye_aspect_ratio(LEFT_EYE, lm, frame.shape[1], frame.shape[0])
            right_ear = eye_aspect_ratio(RIGHT_EYE, lm, frame.shape[1], frame.shape[0])
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EYE_CLOSED_THRESHOLD:
                eyes_closed = True
            else:
                features = get_features(lm)
                if features is not None:
                    pred = model.predict(features)[0]
                    if pred == "away":
                        looking_away = True
                    elif pred == "screen":
                        looking_screen = True

        good_rest = eyes_closed or looking_away
        now = time.time()

        if good_rest:
            if last_good_time is None:
                last_good_time = now
            else:
                accumulated += now - last_good_time
                last_good_time = now
        else:
            last_good_time = None

        remaining = max(0, math.ceil(REQUIRED_REST_SECONDS - accumulated))
        progress = min(1.0, accumulated / REQUIRED_REST_SECONDS)

        canvas.itemconfig(timer_text, text=str(remaining))
        canvas.itemconfig(progress_arc, extent=-360 * progress)

        spin_angle = (spin_angle + 7) % 360
        canvas.itemconfig(spinner_arc, start=spin_angle)

        if eyes_closed:
            canvas.itemconfig(status_title, text="Keep closing your eyes!", fill="#a6a6ff")
            canvas.itemconfig(status_sub, text="Great. Keep your eyes closed and breathe.")
        elif looking_away:
            canvas.itemconfig(status_title, text="Keep looking away!", fill="#78e0c2")
            canvas.itemconfig(status_sub, text="Good. Keep your gaze away from the screen.")
        elif looking_screen:
            canvas.itemconfig(status_title, text="Don’t stare at the screen", fill="#ff6b6b")
            canvas.itemconfig(status_sub, text="Look away or close your eyes to continue.")
        else:
            canvas.itemconfig(status_title, text="Rest your eyes", fill="#ffb3b3")
            canvas.itemconfig(status_sub, text="Move your gaze away or close your eyes.")

        if accumulated >= REQUIRED_REST_SECONDS:
            canvas.itemconfig(timer_text, text="0")
            canvas.itemconfig(status_title, text="Break complete!", fill="#78e0c2")
            canvas.itemconfig(status_sub, text="Nice work. Returning now...")
            should_close = True
            root.after(600, close_popup)
            return

        root.after(50, update)

    update()
    root.mainloop()

    try:
        cap.release()
    except Exception:
        pass

    try:
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    show_rest_popup()