from ultralytics import YOLO

# Step 1: Dataset download (already done, uncomment if needed again)


# Step 2: Train model on downloaded dataset
model = YOLO("yolov8n.pt")
model.train(data="../drone-obstacles-detection-1/data.yaml", epochs=5, device="cpu")

print("Training complete. Weights saved to runs/detect/train/weights/best.pt")