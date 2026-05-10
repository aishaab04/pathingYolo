from ultralytics import YOLO
import cv2

# obstacle classes the model was trained on
OBSTACLE_CLASSES = {"tree", "power_line", "building","bird", "kite", "person", "traffic light"}

# load trained model
model = YOLO("../runs/detect/train-3/weights/best.pt")
model2 = YOLO("yolov8n.pt")

def detect_obstacle(path):
    results = model(path)
    results2 = model2(path)

    for box in results[0].boxes:
        label = results[0].names[int(box.cls)]
        confidence = float(box.conf)
        print ("..... got something here" , label)
        if confidence > 0.4 and label in OBSTACLE_CLASSES:
            return True
    
    for box in results2[0].boxes: 
        label = results2[0].names[int(box.cls)]
        confidence = float(box.conf)
        print ("..... got something here" , label)
        if confidence > 0.4 and label in OBSTACLE_CLASSES:
            return True
        
    print ("the model here is false " , path)
    return False