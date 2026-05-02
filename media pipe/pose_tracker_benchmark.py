import cv2
import mediapipe as mp
import socket
import json
import time
import numpy as np
import psutil
import platform
import subprocess

# --- [Color and Style Definitions] ---
COLOR_TEXT = (0, 155, 255)      # #FF9B00
COLOR_BG = (39, 20, 0)         # #001427
COLOR_POINT = (0, 192, 255)    # #FFC000
COLOR_LINE = (255, 255, 255)   # #FFFFFF
FONT = cv2.FONT_HERSHEY_DUPLEX

# --- [CPU Model Information] ---
def get_cpu_model():
    try:
        if platform.system() == "Windows":
            return subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split('\n')[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        else:
            return platform.processor()
    except:
        return platform.processor()

cpu_model = get_cpu_model()

# --- [User Setup] ---
MODEL_COMPLEXITY = 1 # 기본값
print("="*40)
print(" MediaPipe Pose Benchmark Setup")
print("="*40)
setup_yn = input("Do you want to set up model complexity? (default: 1) [Y/N]: ")

if setup_yn.lower() in ['y', 'yes']:
    while True:
        try:
            m_input = int(input("Select model (0: Fast, 1: Balanced, 2: Accurate): "))
            if m_input in [0, 1, 2]:
                MODEL_COMPLEXITY = m_input
                break
            else: print("Input 0, 1, or 2.")
        except ValueError: print("Invalid input.")

model_names = ["Fast", "Balanced", "Accurate"]
print(f"Selected model : {model_names[MODEL_COMPLEXITY]}!")

# --- [reset] ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5066
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=MODEL_COMPLEXITY, 
    smooth_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# --- [Tracking Variables] ---
frame_count = 0
start_session_time = time.time()
latencies = []
fps_list = []
last_time = time.time()
cpu_usage = 0
cpu_last_update = 0

print("\n" + "="*40)
print(f" Running Benchmark...")
print(f" CPU: {cpu_model}")
print(f" Model: {model_names[MODEL_COMPLEXITY]}")
print(" Press 'ESC' to stop.")
print("="*40)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame_start = time.time()

        # 1. MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inference_start = time.time()
        results = pose.process(img_rgb)
        inference_end = time.time()
        
        latency_ms = (inference_end - inference_start) * 1000
        latencies.append(latency_ms)

        # 2. UDP
        if results.pose_world_landmarks:
            landmarks = []
            for lm in results.pose_world_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            sock.sendto(json.dumps(landmarks).encode(), (UDP_IP, UDP_PORT))

        # 3. FPS & CPU
        curr_time = time.time()
        fps = 1 / (curr_time - last_time)
        last_time = curr_time
        fps_list.append(fps)
        frame_count += 1

        # CPU amount used
        if curr_time - cpu_last_update > 0.5:
            cpu_usage = psutil.cpu_percent()
            cpu_last_update = curr_time

        # 4. UI Render
        # (1) RendMark Styleing
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,0), thickness=0, circle_radius=0),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_LINE, thickness=2)
            )
            h, w, _ = frame.shape
            for lm in results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, COLOR_LINE, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 3, COLOR_POINT, -1, cv2.LINE_AA)

        # (2) dashboard
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 240), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # (3) Text Info
        avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
        info_text = [
            f"FPS: {int(fps)} | AVG: {int(avg_fps)}",
            f"LATENCY: {latency_ms:.1f} ms",
            f"CPU USAGE: {int(cpu_usage)}%",
            f"MODEL: {model_names[MODEL_COMPLEXITY]}",
            f"TIME: {int(time.time() - start_session_time)}s"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (25, 50 + (i * 40)), FONT, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Pose Benchmark', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

except KeyboardInterrupt: pass

# --- [Final Report] ---
duration = time.time() - start_session_time
if frame_count > 10:
    print("\n" + "X"*40)
    print(" BENCHMARK FINAL REPORT ")
    print("X" * 40)
    print(f" CPU Model:      {cpu_model}")
    print(f" Total Duration: {duration:.2f}s")
    print("-" * 40)
    print(f" Average FPS:    {np.mean(fps_list):.2f}")
    print(f" 1% Low FPS:     {np.percentile(fps_list, 1):.2f}")
    print(f" Avg Latency:    {np.mean(latencies):.2f} ms")
    print(f" 95% Latency:    {np.percentile(latencies, 95):.2f} ms")
    print("=" * 40)

input("\nPress ENTER to close...")
cap.release()
cv2.destroyAllWindows()
sock.close()

# --- it's reemo! ---