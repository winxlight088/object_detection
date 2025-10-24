import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import os
from sort import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Counter")
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, required=False, help='Path to save output video (optional)')
    args = parser.parse_args()

    input_video_path = args.input
    output_video_path = args.output

    cap = cv2.VideoCapture(input_video_path)

    # If output path is provided, set up video writer
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        out = None

    # Load YOLO model
    model = YOLO("../Yolo-Weights/yolov8l.pt")

    # Class names for object detection
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    # Load and check mask
    # Get the directory where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    mask_path = os.path.join(BASE_DIR, 'mask.png')

    # Then use mask_path in your imread
    mask = cv2.imread(mask_path)

    # Initialize tracker for object tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Define counting line coordinates [x1, y1, x2, y2]
    limits = [0, 450, 640, 450]
    totalCount = []

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame from video")
            break

        # Resize mask to match frame size if needed
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Apply mask to the frame
        try:
            imgRegion = cv2.bitwise_and(img, mask)
        except cv2.error as e:
            print(f"Error applying mask: {e}")
            break

        # Overlay graphics on the image
        try:
            graphics_path = os.path.join(BASE_DIR, 'graphics.png')
            imgGraphics = cv2.imread(graphics_path, cv2.IMREAD_UNCHANGED)
            if imgGraphics is not None:
                img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
            else:
                print(f"Warning: graphics.png not found at {graphics_path}")
        except Exception as e:
            print(f"Error overlaying graphics: {e}")

        # Process frame with YOLO model
        results = model(imgRegion, stream=True)
        detections = np.empty((0, 5))

        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Get confidence and class
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                # Filter relevant classes with confidence threshold
                if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        # Update object tracker
        resultsTracker = tracker.update(detections)

        # Draw counting line
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        # Process tracking results
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Draw bounding box and ID
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                              scale=2, thickness=3, offset=10)

            # Calculate center point
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Check if object crosses the counting line
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    # Change line color to green when vehicle crosses
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # Display total count
        cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

        # Show images
        cv2.imshow("Image", img)
        if out is not None:
            out.write(img)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # Print the final count
    print(f"Total vehicles counted: {len(totalCount)}")
