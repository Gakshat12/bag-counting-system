import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# -------------------------------
# LOAD CUSTOM MODEL
# -------------------------------
model = YOLO("runs/detect/train3/weights/best.pt")

# -------------------------------
# TRACKER
# -------------------------------
tracker = Sort()

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

line_x = 300   # 🔥 adjust based on your video
frame_count = 0

# -------------------------------
# MAIN LOOP
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------------------
    # SPEED OPTIMIZATION
    # ---------------------------
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (640, 360))

    # ---------------------------
    # YOLO DETECTION (ONLY BAG)
    # ---------------------------
    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])

        # Only jute bag class
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])

        if conf > 0.3:
            detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)

    # ---------------------------
    # TRACKING
    # ---------------------------
    if len(detections) > 0:
        tracks = tracker.update(detections)
    else:
        tracks = []

    # ---------------------------
    # DRAW VERTICAL LINE
    # ---------------------------
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 4)

    # ---------------------------
    # PROCESS TRACKS
    # ---------------------------
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw centroid
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Debug
        cv2.putText(frame, f"cx:{cx}", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # -----------------------
        # X-AXIS COUNTING LOGIC
        # -----------------------
        if track_id not in counted_ids:
            if cx > line_x:
                count += 1
                counted_ids.add(track_id)
                print(f"✅ Counted ID: {track_id}")

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