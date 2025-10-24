import argparse
from ultralytics import YOLO
import cv2
import cvzone
import math
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPE Detection")
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, required=False, help='Path to save output video (optional)')
    args = parser.parse_args()

    input_video_path = args.input
    output_video_path = args.output

    cap = cv2.VideoCapture(input_video_path)

    # Load the YOLO model trained for PPE detection
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ppe_weights_path = os.path.join(BASE_DIR, 'ppe.pt')
    if not os.path.exists(ppe_weights_path):
        print(f"Error: ppe.pt not found at {ppe_weights_path}")
        exit(1)
    model = YOLO(ppe_weights_path)

    # List of class names as per the model's training
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']

    # Default color for bounding boxes (red)
    myColor = (0, 0, 255)

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
        # Read a frame from the video
        success, img = cap.read()
        if not success:
            break  # Exit loop if video ends or frame not read

        # Run YOLO model on the frame (stream=True for efficiency)
        results = model(img, stream=True)

        # Iterate through detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # Optionally draw a rectangle or corner rectangle
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                # cvzone.cornerRect(img, (x1, y1, w, h))

                # Get confidence score for the detection
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Get class index and class name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                print(currentClass)  # Print detected class for debugging

                # Only process detections with confidence > 0.5
                if conf > 0.5:
                    # Set bounding box color based on class (red for missing PPE, green for present, blue for others)
                    if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                        myColor = (0, 0, 255)  # Red for missing PPE
                    elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                        myColor = (0, 255, 0)  # Green for present PPE
                    else:
                        myColor = (255, 0, 0)  # Blue for other classes

                    # Draw a smaller corner rectangle for the bounding box
                    cvzone.cornerRect(img, (x1, y1, w, h), l=7, rt=2, colorR=myColor)
                    # Draw the class label and confidence using cv2.putText for clarity
                    label = f'{classNames[cls]} {conf}'
                    cv2.putText(img, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2, cv2.LINE_AA)

        # Display the frame with detections
        cv2.imshow("Image", img)
        if out is not None:
            out.write(img)
        # Wait for 1 ms and check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # Print the final count (for now, just a placeholder since PPE detection doesn't count objects)
    print("PPE Detection Complete")
