import argparse
from ultralytics import YOLO

def evaluate(model_name):
    model = YOLO(f'./Gun_Detection/{model_name}/weights/best.pt')
    results = model.val(data='data.yaml', split='test', imgsz=640, batch=16, device=0, project='Gun_Detection', name=f'{model_name}_Validation', patience=30, plots=True) 

    print(f'Results of {model_name}: {results.results_dict}')
    print(f'Speed of {model_name}:{results.speed}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate YOLO Models for Object Detection")
    parser.add_argument('--models', nargs='+', default=['yolov8n', 'yolov8x', 'yolov9t', 'yolov9e', 'yolov10n', 'yolov10x', 'yolo11n', 'yolo11x'], help="List of YOLO model names to evaluate")

    args = parser.parse_args()

    for model_name in args.models:
        evaluate(model_name)
