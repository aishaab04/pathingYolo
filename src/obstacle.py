from ultralytics import YOLO
import cv2

# obstacle classes the model was trained on
OBSTACLE_CLASSES = {"tree", "powerline", "building"}

# load trained model
model = YOLO("../runs/detect/train-3/weights/best.pt")

def detect_obstacle(path):
    results = model(path)

    for box in results[0].boxes:
        label = results[0].names[int(box.cls)]
        confidence = float(box.conf)
        print ("..... got sonmthing here" , label)
        if confidence > 0.4 and label in OBSTACLE_CLASSES:
            return True
        
    print ("the model here is false " , path)
    return False