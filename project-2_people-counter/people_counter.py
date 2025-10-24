import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="People Counter")
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, required=False, help='Path to save output video (optional)')
    args = parser.parse_args()

    input_video_path = args.input
    output_video_path = args.output

    cap = cv2.VideoCapture(input_video_path)

    # Load YOLO model for object detection
    model = YOLO("../Yolo-Weights/yolov8l.pt")

    # List of class names for YOLO
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

    # Load mask image
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    mask_path = os.path.join(BASE_DIR, 'mask.png')
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Error: mask.png not found at {mask_path}")
        exit(1)

    # Initialize tracker for tracking detected objects
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Define line coordinates for counting people moving up and down
    limitsUp = [103, 161, 296, 161]
    limitsDown = [527, 489, 735, 489]

    totalCountUp = []  # List to store IDs of people moving up
    totalCountDown = []  # List to store IDs of people moving down

    # If output path is provided, set up video writer
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        out = None

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame from video.")
            break

        # Apply mask to the frame to focus on region of interest
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        imgRegion = cv2.bitwise_and(img, mask)

        # Overlay graphics on the image (optional)
        graphics_path = os.path.join(BASE_DIR, 'graphics.png')
        imgGraphics = cv2.imread(graphics_path, cv2.IMREAD_UNCHANGED)
        if imgGraphics is not None:
            img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
        else:
            print(f"Warning: graphics.png not found at {graphics_path}")

        # Run YOLO model on the masked region
        results = model(imgRegion, stream=True)

        detections = np.empty((0, 5))  # Array to store detection results

        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence score
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class index
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                # Only consider detections of 'person' with confidence > 0.3
                if currentClass == "person" and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        # Update tracker with current detections
        resultsTracker = tracker.update(detections)

        # Draw counting lines on the image
        cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
        cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

        # Process each tracked object
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)  # Print tracking result for debugging
            w, h = x2 - x1, y2 - y1
            # Draw bounding box and ID
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            # Calculate center point of the bounding box
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Check if the person crosses the 'up' line
            if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
                if totalCountUp.count(id) == 0:
                    totalCountUp.append(id)
                    # Change line color to green when person crosses
                    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

            # Check if the person crosses the 'down' line
            if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
                if totalCountDown.count(id) == 0:
                    totalCountDown.append(id)
                    # Change line color to green when person crosses
                    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

        # Display the count of people moving up and down
        cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
        cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

        # Show the final image
        cv2.imshow("Image", img)
        if out is not None:
            out.write(img)

        # Wait for 1 ms and check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # Print the final count
    print(f"People Count: {len(totalCountUp) + len(totalCountDown)}")
