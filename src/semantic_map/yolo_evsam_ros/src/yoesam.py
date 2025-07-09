import rospy
from cv_bridge import CvBridge
from yolo_evsam_ros.srv import VitDetection, VitDetectionResponse
from std_msgs.msg import MultiArrayDimension

import cv2
import numpy as np

import torch
import supervision as sv

from ultralytics import YOLO

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor


class VitDetectionServer(object):
    def __init__(
        self,
        model_path,
        sam_checkpoint,
        sam_model,
        box_threshold=0.1,
        text_threshold=0.3,
    ):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("No GPU available")
            exit()

        rospy.loginfo("Loading models...")

        # Building YOLO-World inference model
        self.yolo_world_model = YOLO(model_path, task="detect")
        self.box_threshold = box_threshold

        # Building EfficientViT-SAM inference model
        efficientvit_sam = (
            create_efficientvit_sam_model(sam_model, True, sam_checkpoint).cuda().eval()
        )
        self.sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

        rospy.loginfo("Models are loaded")

        # ros service
        self.cv_bridge = CvBridge()
        rospy.Service("vit_detection", VitDetection, self.callback)
        rospy.loginfo("vit_detection service has started")

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []

        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])

        return np.array(result_masks)

    def detect(self, image, class_list):
        # detect objects
        self.yolo_world_model.set_classes(class_list)
        results = self.yolo_world_model.predict(source=image, conf=self.box_threshold)
        detections = sv.Detections.from_ultralytics(results[0])

        # classes = results[0].boxes.cls
        # class_id = classes.cpu().int().numpy()

        labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]

        # Segment

        start_time = rospy.Time.now()

        detections.mask = self.segment(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
        )

        end_time = rospy.Time.now()

        seg_time = (end_time - start_time).to_sec()
        rospy.loginfo(f"seg time: {seg_time:.3f} seconds")

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(
            scene=image.copy(), detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        return detections, labels, annotated_image

    def callback(self, request):
        img, prompt = request.color_image, request.prompt
        img = self.cv_bridge.imgmsg_to_cv2(img)

        # class_list = prompt.split(". ")
        class_list = prompt.strip().replace(" ", "").split(".")
        # class_list = [item for item in class_list if item]

        detections, labels, annotated_frame = self.detect(img, class_list)
        boxes = detections.xyxy
        scores = detections.confidence
        masks = detections.mask

        # rospy.loginfo("Detected objects: {}".format(labels))
        # rospy.loginfo("Detection scores: {}".format(scores))

        response = VitDetectionResponse()
        response.labels = labels
        response.class_id = detections.class_id
        response.scores = scores.tolist()
        response.boxes.layout.dim = [
            MultiArrayDimension(label="boxes", size=boxes.shape[0], stride=4)
        ]
        response.boxes.data = boxes.flatten().tolist()
        response.annotated_frame = self.cv_bridge.cv2_to_imgmsg(annotated_frame)

        try:
            stride = masks.shape[1] * masks.shape[2]
            response.segmasks.layout.dim = [
                MultiArrayDimension(label="masks", size=masks.shape[0], stride=stride)
            ]
            response.segmasks.data = masks.flatten().tolist()
        except:  # no masks, masks.shape=(0,)
            if masks.size == 0:
                response.segmasks.layout.dim = [
                    MultiArrayDimension(label="masks", size=0, stride=0)
                ]
                response.segmasks.data = []
            else:
                raise ValueError(f"masks.shape is unexpected {masks.shape}")

        # We release the gpu memory
        torch.cuda.empty_cache()

        return response


if __name__ == "__main__":
    rospy.init_node("yolo_evsam_ros")

    # get arguments from the ros parameter server
    model_path = rospy.get_param("~model_path")
    # config = rospy.get_param('~config')
    sam_checkpoint = rospy.get_param("~sam_checkpoint")
    sam_model = rospy.get_param("~sam_model")
    box_threshold = rospy.get_param("~box_threshold")
    text_threshold = rospy.get_param("~text_threshold")

    # start the server
    VitDetectionServer(
        model_path, sam_checkpoint, sam_model, box_threshold, text_threshold
    )
    rospy.spin()
