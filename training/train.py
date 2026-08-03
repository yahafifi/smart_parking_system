from ultralytics import YOLO
import os

# Go to project root
project_root = r"C:\Users\Yahia\Desktop\smart_parking_system"

data_yaml = os.path.join(project_root, "dataset", "data.yaml")

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=4,
    name="parking_yolo_demo",
    project=os.path.join(project_root, "runs")
)

print("Training finished successfully!")