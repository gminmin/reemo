import cv2
import mediapipe as mp
import socket
import json
import time
import numpy as np
import psutil
import platform
import subprocess

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5066
MODEL_COMPLEXITY = 1  # 0: Fast, 1: Balanced, 2: Accurate
SHOW_PREVIEW = True
SEND_UDP = True

# --- CPU Info ---
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

# --- Initialize ---
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

# --- Stats Tracking ---
frame_count = 0
start_session_time = time.time()
latencies = []
fps_list = []
last_time = time.time()

print("="*40)
print(" MediaPipe Pose Benchmark Started ")
print(f" CPU:         {cpu_model}")
print(f" Resolution:  {int(width)}x{int(height)}")
print(f" Complexity:  {MODEL_COMPLEXITY}")
print(" Press 'ESC' to stop and see report.")
print("="*40)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame_start = time.time()

        # 1. Pre-processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Inference
        inference_start = time.time()
        results = pose.process(img_rgb)
        inference_end = time.time()
        
        latency_ms = (inference_end - inference_start) * 1000
        latencies.append(latency_ms)

        # 3. UDP
        if results.pose_world_landmarks and SEND_UDP:
            landmarks = []
            for lm in results.pose_world_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            sock.sendto(json.dumps(landmarks).encode(), (UDP_IP, UDP_PORT))

        # 4. FPS & CPU Usage
        curr_time = time.time()
        fps = 1 / (curr_time - last_time)
        last_time = curr_time
        fps_list.append(fps)
        cpu_usage = psutil.cpu_percent()
        frame_count += 1

        # 5. UI Rendering
        if SHOW_PREVIEW:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Metrics
            avg_latency = np.mean(latencies[-30:]) if latencies else 0
            avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
            one_percent_low = np.percentile(fps_list, 1) if len(fps_list) > 100 else 0
            max_fps = np.max(fps_list) if fps_list else 0
            elapsed_time = time.time() - start_session_time
            
            info_text = [
                f"FPS: {int(fps)} (Avg: {int(avg_fps)} | Max: {int(max_fps)})",
                f"1% Low FPS: {one_percent_low:.1f}",
                f"Latency: {latency_ms:.1f}ms (Avg: {avg_latency:.1f}ms)",
                f"CPU Usage: {cpu_usage}%",
                f"Session Time: {int(elapsed_time)}s",
                f"Model: {MODEL_COMPLEXITY} | {int(width)}x{int(height)}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (20, 50 + (i * 35)), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Pose Benchmark', frame)
            if cv2.waitKey(1) & 0xFF == 27: break

except KeyboardInterrupt:
    pass

# --- Final Report ---
duration = time.time() - start_session_time
if frame_count > 10:
    print("\n" + "="*40)
    print(" Benchmark Final Report ")
    print("-" * 40)
    print(f" CPU Model:        {cpu_model}")
    print(f" Total Duration:   {duration:.2f}s")
    print("-" * 40)
    print(f" Average FPS:      {np.mean(fps_list):.2f}")
    print(f" Maximum FPS:      {np.max(fps_list):.2f}")
    print(f" 1% Low FPS:       {np.percentile(fps_list, 1):.2f}")
    print("-" * 40)
    print(f" Avg Processing:   {np.mean(latencies):.2f} ms")
    print(f" 95% Processing:   {np.percentile(latencies, 95):.2f} ms")
    print("=" * 40)
else:
    print("\nNot enough data for benchmark report.")

input("\nBenchmark Finished. Press ENTER to close this window...")

cap.release()
cv2.destroyAllWindows()
sock.close()
