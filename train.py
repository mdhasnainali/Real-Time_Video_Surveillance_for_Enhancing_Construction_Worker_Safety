import argparse
from ultralytics import YOLO

def train_model_and_store(model_name, epochs, img_size, batch_size, device):
    model = YOLO(f'./base_models/{model_name}')
    model.train(data='data.yaml', epochs=epochs, imgsz=img_size, batch=batch_size, device=device, project='Gun_Detection', name=model_name, patience=30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO Models for Object Detection")
    parser.add_argument('--models', nargs='+', default=['yolov8n', 'yolov8x', 'yolov9t', 'yolov9e', 'yolov10n', 'yolov10x', 'yolo11n', 'yolo11x'], help="List of YOLO model names to train")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--img_size', type=int, default=640, help="Image size for training")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--device', type=int, default=0, help="GPU device to use")

    args = parser.parse_args()

    for model_name in args.models:
        train_model_and_store(model_name, args.epochs, args.img_size, args.batch_size, args.device)


# nohup python train.py > training_log.txt 2>&1
