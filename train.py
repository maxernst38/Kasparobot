from ultralytics import YOLO
import argparse

def train_yolov8(data_yaml, model='yolov8n.pt', epochs=100, batch_size=16, img_size=640):
    # Load the YOLOv8 model
    model = YOLO(model)

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project='runs/train',  # Output directory for training results
        name='custom_model',    # Name for the training run
        save=True,               # Save the model checkpoints
        device='cuda'
    )

    print("Training complete. Results saved in:", results.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model on a custom dataset.")
    parser.add_argument('--data_yaml', type=str, required=True, help="Path to the dataset's YAML file.")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help="YOLOv8 model variant (e.g., yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--img_size', type=int, default=480, help="Image size for training (e.g., 640, 1280, etc.)")

    args = parser.parse_args()

    train_yolov8(
        data_yaml=args.data_yaml,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
