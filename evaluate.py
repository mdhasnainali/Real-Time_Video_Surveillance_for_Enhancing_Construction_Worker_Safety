import argparse
from ultralytics import YOLO

def evaluate(model_name):
    model = YOLO(f'./Gun_Detection/{model_name}/weights/best')
    results = model.val(data='data.yaml')

    # Retrieve mAP metrics and other details
    map50 = results.box.map50  # mAP@0.5
    map5095 = results.box.map  # mAP@0.5:0.95
    precision = results.box.precision.mean()
    recall = results.box.recall.mean()
    f1_score = results.box.f1.mean()

    print(f'Model: {model_name}, mAP50: {map50:.4f}, mAP50-95: {map5095:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate YOLO Models for Object Detection")
    parser.add_argument('--models', nargs='+', default=['yolov8n', 'yolov8x', 'yolov9t', 'yolov9e', 'yolov10n', 'yolov10x', 'yolo11n', 'yolo11x'], help="List of YOLO model names to evaluate")

    args = parser.parse_args()

    for model_name in args.models:
        evaluate(model_name)
