import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------------------
# LOAD CUSTOM MODEL
# -------------------------------
model = YOLO("runs/detect/train12/weights/best.pt")

# -------------------------------
# TRACKER
# -------------------------------
tracker = DeepSort(max_age=30)

# -------------------------------
# VIDEO
# -------------------------------
cap = cv2.VideoCapture("Problem Statement Scenario1.mp4")

if not cap.isOpened():
    print("❌ Video not opened")
    exit()
else:
    print("✅ Video started")

# -------------------------------
# VARIABLES
# -------------------------------
count = 0
counted_ids = set()
previous_positions = {}

frame_count = 0

# -------------------------------
# MAIN LOOP
# -------------------------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # frame skipping for speed
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # counting line in middle of frame
    line_x = frame.shape[1] // 2

    # ---------------------------
    # YOLO DETECTION
    # ---------------------------
    results = model(frame)[0]

    detections = []

    for box in results.boxes:

        cls_id = int(box.cls[0])

        # detect only jute bag
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])

        if conf > 0.4:

            w = x2 - x1
            h = y2 - y1

            detections.append(([x1, y1, w, h], conf, 'bag'))

    # ---------------------------
    # TRACKING
    # ---------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    # ---------------------------
    # DRAW COUNTING LINE
    # ---------------------------
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 4)

    # ---------------------------
    # PROCESS TRACKS
    # ---------------------------
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l, t, r, b = track.to_ltrb()
        l, t, r, b = int(l), int(t), int(r), int(b)

        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        # draw bounding box
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

        cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # draw centroid
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # -----------------------
        # COUNTING LOGIC
        # RIGHT → LEFT
        # -----------------------
        if track_id in previous_positions:

            prev_x = previous_positions[track_id]

            # detect crossing
            if prev_x > line_x and cx <= line_x:

                if track_id not in counted_ids:
                    count += 1
                    counted_ids.add(track_id)

                    print(f"✅ Counted ID: {track_id}")

        # update previous position
        previous_positions[track_id] = cx

    # ---------------------------
    # DISPLAY COUNT
    # ---------------------------
    cv2.putText(frame, f"Bag Count: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # ---------------------------
    # SHOW FRAME
    # ---------------------------
    cv2.imshow("Bag Counting System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()