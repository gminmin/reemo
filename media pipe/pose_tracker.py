import cv2
import mediapipe as mp
import socket
import json
import time

# UDP Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5066
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe Pose Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0

print(f"Sending data to {UDP_IP}:{UDP_PORT}...")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # To improve performance, optionally mark the image as not writeable to
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_world_landmarks:
        # Landmarks list
        landmarks_list = []
        for lm in results.pose_world_landmarks.landmark:
            # MediaPipe World Landmarks are in meters. 
            # Origin is roughly at the hips.
            # x: right is positive, y: down is positive, z: forward is positive
            # We invert y later in Unity to match coordinate systems
            landmarks_list.extend([lm.x, lm.y, lm.z])
        
        # Send data via UDP
        data = json.dumps(landmarks_list)
        sock.sendto(data.encode(), (UDP_IP, UDP_PORT))

        # We can still draw the normalized landmarks on screen for preview
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display FPS on image
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow('MediaPipe Pose Tracker', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
