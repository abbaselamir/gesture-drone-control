import cv2
import json
import time
import socket
import numpy as np
from ultralytics import YOLO

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

model = YOLO("model/best.onnx", task="detect")

# --- Control & Safety Parameters ---
DEAD_ZONE = 0.15
ALPHA = 0.25
MAX_SPEED = 1.5
neutral_area = 0.08
DEBOUNCE_FRAMES = 10      # Hold gesture ~0.35s to switch flight mode
FAILSAFE_TIMEOUT = 1.5    # Auto-hover if hand disappears for 1.5s

smooth_vx, smooth_vy, smooth_vz = 0.0, 0.0, 0.0
flight_mode = "LANDED"
gesture_streak = 0
last_candidate = ""
last_seen_time = time.time()
current_box_area = neutral_area

def apply_deadzone(val, threshold=DEAD_ZONE):
    if abs(val) < threshold:
        return 0.0
    sign = 1 if val > 0 else -1
    return sign * (abs(val) - threshold) / (1.0 - threshold)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = model(frame, conf=0.55, verbose=False)

    raw_vx, raw_vy, raw_vz = 0.0, 0.0, 0.0
    gesture = "NONE"
    status_text = "READY"

    if len(results.boxes) > 0:
        last_seen_time = time.time()
        box = results.boxes
        gesture = model.names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()

        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        norm_x = (cx - 0.5) * 2.0
        norm_y = (0.5 - cy) * 2.0

        current_box_area = ((x2 - x1) * (y2 - y1)) / (w * h)
        norm_z = np.clip((current_box_area - neutral_area) * 8.0, -1.0, 1.0)

        # Debounced Gesture State Machine
        g = gesture.lower()
        if g == last_candidate:
            gesture_streak += 1
        else:
            last_candidate = g
            gesture_streak = 1

        if gesture_streak >= DEBOUNCE_FRAMES:
            if g in ["stop", "palm", "five"]:
                flight_mode = "FLYING"
            elif g in ["ok", "peace", "fist"]:
                flight_mode = "LANDED"

        if flight_mode == "FLYING":
            raw_vy = -apply_deadzone(norm_x) * MAX_SPEED
            raw_vz = apply_deadzone(norm_y) * 1.0
            raw_vx = apply_deadzone(norm_z) * MAX_SPEED

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx * w), int(cy * h)), 6, (0, 0, 255), -1)
    else:
        gesture_streak = 0
        if time.time() - last_seen_time > FAILSAFE_TIMEOUT and flight_mode == "FLYING":
            status_text = "AUTO-HOVER (HAND LOST)"

    smooth_vx = ALPHA * raw_vx + (1 - ALPHA) * smooth_vx
    smooth_vy = ALPHA * raw_vy + (1 - ALPHA) * smooth_vy
    smooth_vz = ALPHA * raw_vz + (1 - ALPHA) * smooth_vz

    packet = json.dumps({
        "mode": flight_mode,
        "vx": round(smooth_vx, 3),
        "vy": round(smooth_vy, 3),
        "vz": round(smooth_vz, 3)
    })
    sock.sendto(packet.encode(), (UDP_IP, UDP_PORT))

    # Draw Virtual Joystick Dead-Zone & Telemetry HUD
    dz_x1, dz_y1 = int(w * (0.5 - DEAD_ZONE / 2)), int(h * (0.5 - DEAD_ZONE / 2))
    dz_x2, dz_y2 = int(w * (0.5 + DEAD_ZONE / 2)), int(h * (0.5 + DEAD_ZONE / 2))
    cv2.rectangle(frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (255, 255, 0), 1)

    cv2.putText(frame, f"Mode: {flight_mode} | Gesture: {gesture} ({status_text})", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Vx:{smooth_vx:.2f} Vy:{smooth_vy:.2f} Vz:{smooth_vz:.2f} | [C] Calibrate [Q] Quit", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("YOLO26 Gesture Drone Control", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        neutral_area = current_box_area
        print(f"Calibrated neutral hand area to: {neutral_area:.4f}")

cap.release()
cv2.destroyAllWindows()
