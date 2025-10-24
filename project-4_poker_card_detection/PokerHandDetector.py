import argparse
from ultralytics import YOLO
import cv2
import cvzone
import math
import PokerHandFunction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poker Card Detection")
    parser.add_argument('--input', type=str, help='Input video file path (leave empty for webcam)')
    parser.add_argument('--output', type=str, help='Output video file path')
    args = parser.parse_args()

    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(0)  # For Webcam
        cap.set(3, 1280)
        cap.set(4, 720)

    model = YOLO("project-4_poker_card_detection/playingCards.pt")
    classNames = [
        '10C', '10D', '10H', '10S',
        '2C', '2D', '2H', '2S',
        '3C', '3D', '3H', '3S',
        '4C', '4D', '4H', '4S',
        '5C', '5D', '5H', '5S',
        '6C', '6D', '6H', '6S',
        '7C', '7D', '7H', '7S',
        '8C', '8D', '8H', '8S',
        '9C', '9D', '9H', '9S',
        'AC', 'AD', 'AH', 'AS',
        'JC', 'JD', 'JH', 'JS',
        'KC', 'KD', 'KH', 'KS',
        'QC', 'QD', 'QH', 'QS']

    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    else:
        out = None

    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        hand = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                if conf > 0.5:
                    hand.append(classNames[cls])

        print(hand)
        hand = list(set(hand))
        print(hand)
        if len(hand) == 5:
            results = PokerHandFunction.findPokerHand(hand)
            print(results)
            # Display the detected poker hand result on the top-left for better visibility
            cvzone.putTextRect(img, f'Your Hand: {results}', (30, 75), scale=2, thickness=4)

        if out:
            out.write(img)
        else:
            # Show the webcam image with all detections and results
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Poker Hand Detection Complete")
