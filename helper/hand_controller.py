import cv2
import json
import time
import socket
import numpy as np
from pathlib import Path
from ultralytics import YOLO

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "best.onnx"
model = YOLO(str(MODEL_PATH), task="detect")

# --- Control & Safety Parameters ---
DEAD_ZONE = 0.10
ALPHA = 0.35
MAX_SPEED = 1.8
AREA_SENSITIVITY = 10.0
VERTICAL_SPEED = 1.2
neutral_area = 0.08
DEBOUNCE_FRAMES = 10      # Hold gesture ~0.35s to switch flight mode
FAILSAFE_TIMEOUT = 1.5    # Auto-hover if hand disappears for 1.5s

smooth_vx, smooth_vy, smooth_vz = 0.0, 0.0, 0.0
flight_mode = "LANDED"
gesture_streak = 0
last_candidate = ""
stop_latched = False
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
    hand_detected = False

    result = results[0] if results else None
    boxes = result.boxes if result is not None else None
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
        last_seen_time = time.time()
        hand_detected = True
        gesture = model.names[class_id]

        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        norm_x = (cx - 0.5) * 2.0

        current_box_area = ((x2 - x1) * (y2 - y1)) / (w * h)
        norm_z = np.clip((current_box_area - neutral_area) * AREA_SENSITIVITY, -1.0, 1.0)

        # Debounced Gesture State Machine
        g = gesture.lower()
        if g == last_candidate:
            gesture_streak += 1
        else:
            last_candidate = g
            gesture_streak = 1
            if g != "stop":
                stop_latched = False

        if gesture_streak >= DEBOUNCE_FRAMES and g == "like":
            flight_mode = "FLYING"
        elif gesture_streak >= DEBOUNCE_FRAMES and g == "stop" and not stop_latched:
            flight_mode = "LANDED"
            stop_latched = True

        if flight_mode == "FLYING":
            if g == "peace":
                if gesture_streak >= DEBOUNCE_FRAMES:
                    raw_vy = MAX_SPEED
                    status_text = "BANK LEFT"
            elif g == "ok":
                if gesture_streak >= DEBOUNCE_FRAMES:
                    raw_vy = -MAX_SPEED
                    status_text = "BANK RIGHT"
            elif g != "stop":
                raw_vy = -apply_deadzone(norm_x) * MAX_SPEED
            if g == "like" and gesture_streak >= DEBOUNCE_FRAMES:
                raw_vz = VERTICAL_SPEED
            raw_vx = apply_deadzone(norm_z) * MAX_SPEED

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx * w), int(cy * h)), 6, (0, 0, 255), -1)
    else:
        gesture_streak = 0
        last_candidate = ""
        stop_latched = False
        if time.time() - last_seen_time > FAILSAFE_TIMEOUT and flight_mode == "FLYING":
            status_text = "AUTO-HOVER (HAND LOST)"
            smooth_vx = smooth_vy = smooth_vz = 0.0

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
        if hand_detected:
            neutral_area = current_box_area
            print(f"Calibrated neutral hand area to: {neutral_area:.4f}")
        else:
            print("Cannot calibrate: no hand detected.")

cap.release()
cv2.destroyAllWindows()
