from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = YOLO("runs/train/custom_model/weights/best.pt")

results = model("dataset/valid/images/d114edc5cb4cae0ceb2f152afd15f57d_jpg.rf.b8c3458d9e3de66bb363d1559a8154c7.jpg")

detections = results[0].boxes

for det in detections:
    print(f"Class: {det.cls.item()}, Confidence: {det.conf.item()}, Coordinates: {det.xyxy.tolist()}")

results[0].save(filename="output.jpg")

