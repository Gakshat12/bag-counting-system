Bag Counting System

A real-time computer vision system that detects, tracks, and counts jute bags crossing a defined line in video footage — built using YOLO for object detection and DeepSORT for multi-object tracking.

Overview

This project solves a common industrial/logistics problem: automatically counting items (bags) moving through a fixed point in a video feed, without manual tallying. It combines a custom-trained YOLO object detection model with DeepSORT tracking to maintain consistent object identities across frames, then applies line-crossing logic to count each bag exactly once.

Features


Custom object detection using a YOLO model fine-tuned specifically for jute bag detection
Multi-object tracking with DeepSORT to maintain unique IDs across frames and avoid duplicate counting
Line-crossing counting logic — counts an object only once as it crosses a defined vertical line (right → left)
Real-time visualization — bounding boxes, tracking IDs, centroids, and live count overlay
Frame skipping for performance optimization on longer videos


Tech Stack


Python
OpenCV — video I/O and visualization
Ultralytics YOLO — object detection
DeepSORT (deep-sort-realtime) — multi-object tracking
NumPy — array operations
Roboflow — dataset annotation and management for custom YOLO training


Dataset

The YOLO model was trained on a custom-annotated dataset built and managed using Roboflow, containing labeled images of jute bags. The dataset includes bounding box annotations for the bag class, with standard augmentations applied during preprocessing (e.g. flips, brightness/contrast adjustments) to improve model generalization.


Annotation tool: Roboflow
Classes: bag (jute bag)
Export format: YOLO (Ultralytics)


How It Works


Detection — Each video frame is passed through a custom-trained YOLO model, filtering detections to the target class (jute bag) above a confidence threshold of 0.4.
Tracking — Detected bounding boxes are passed to DeepSORT, which assigns and maintains a unique track ID for each bag across frames.
Counting Logic — The centroid of each tracked bag is monitored relative to a vertical counting line at the horizontal midpoint of the frame. A bag is counted once when its centroid transitions from right of the line to left of it, and each track ID is only counted once (tracked via a counted_ids set).
Visualization — Bounding boxes, track IDs, centroids, the counting line, and a running count are drawn on the frame in real time.


Project Structure

bag-counting-system/
├── bag_counter.py              # Main script
├── runs/detect/train12/weights/best.pt   # Custom-trained YOLO weights
├── Problem Statement Scenario1.mp4       # Sample input video
└── README.md

Setup & Usage

Prerequisites

bashpip install opencv-python numpy ultralytics deep-sort-realtime

Run

bashpython bag_counter.py

Press q or Esc to exit the video window.

Configuration


Model path: Update the path in YOLO("runs/detect/train12/weights/best.pt") to point to your trained weights.
Video source: Update cv2.VideoCapture("Problem Statement Scenario1.mp4") to your input video path, or pass 0 for a live webcam feed.
Confidence threshold: Adjust conf > 0.4 to tune detection sensitivity.
Counting direction: The current logic counts right-to-left crossings; flip the comparison in the counting block to count left-to-right instead.


Future Improvements


Support multi-class counting (different object types simultaneously)
Configurable counting line orientation (horizontal/diagonal)
Export count logs with timestamps to CSV
Deploy as a lightweight API (FastAPI) for integration into larger systems
Optimize inference for edge devices
