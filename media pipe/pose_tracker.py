import cv2
import mediapipe as mp
import socket
import json
import time
import psutil

# --- [Color and Style Definitions] ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5066
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

COLOR_TEXT = (0, 155, 255)      # #FF9B00
COLOR_BG = (39, 20, 0)         # #001427
COLOR_POINT = (0, 192, 255)    # #FFC000
COLOR_LINE = (255, 255, 255)   # #FFFFFF

# MediaPipe Pose Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0

# --- [CPU delay] ---
cpu_usage = 0
cpu_last_update = 0
cpu_delay = 0.5  # delay Time

print(f"Sending data to {UDP_IP}:{UDP_PORT}...")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # FPS 계산
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # --- [CPU amount used] ---
    if cTime - cpu_last_update > cpu_delay:
        cpu_usage = psutil.cpu_percent()
        cpu_last_update = cTime

    # MediaPipe Pose
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        # Line
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,0), thickness=0, circle_radius=0),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_LINE, thickness=2)
        )
        # Point
        h, w, c = image.shape
        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, COLOR_LINE, -1, cv2.LINE_AA)
            cv2.circle(image, (cx, cy), 3, COLOR_POINT, -1, cv2.LINE_AA)

    # --- [UI Panel] ---
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (230, 110), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # FPS Display
    cv2.putText(image, "FPS", (25, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(image, "|", (90, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(image, f"{int(fps)}", (110, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)
    
    # CPU Display
    cv2.putText(image, "CPU", (25, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(image, "|", (90, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(image, f"{int(cpu_usage)}%", (110, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)

    cv2.imshow('Custom MediaPipe Pose Tracker', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# --- it's reemo! ---