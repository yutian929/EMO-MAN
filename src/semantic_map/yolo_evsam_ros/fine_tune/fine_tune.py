from ultralytics import YOLOWorld
import os

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld(
    "/home/yutian/YanBot/src/Cerebellum/yolo_evsam_ros/weights/yolov8l-worldv2.pt"
)

# first try, without fine tuning
model.set_classes(["person", "horse"])
results = model.predict("test.jpg")
results[0].show()

# Train the model on the COCO8 example dataset for 100 epochs
yaml_path = os.path.join(os.path.dirname(__file__), "fine_tune.yaml")
results = model.train(data=yaml_path, epochs=10, imgsz=640)

# second try, with fine tuning
model.set_classes(["person", "horse"])
results = model.predict("test.jpg")
results[0].show()
