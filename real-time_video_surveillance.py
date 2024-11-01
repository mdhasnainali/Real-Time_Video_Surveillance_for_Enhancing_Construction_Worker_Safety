import argparse
import cv2
from ultralytics import YOLO

def live_detection(model_name, video_source=0):
    model = YOLO(f'./Gun_Detection/{model_name}/weights/best')

    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Make predictions
        results = model.predict(frame)
        
        # Process predictions
        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            class_ids = result.boxes.cls
            
            for box, conf, cls in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                label = f'Class {int(cls)}: {conf:.2f}'
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('YOLO Live Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-Time Object Detection with YOLO")
    parser.add_argument('--model', type=str, default='yolov8n', help="YOLO model name for real-time detection")
    parser.add_argument('--video_source', type=int, default=0, help="Video source (0 for webcam or path to video file)")

    args = parser.parse_args()
    live_detection(args.model, args.video_source)
