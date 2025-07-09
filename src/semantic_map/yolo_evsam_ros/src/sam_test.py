import torch
import numpy as np
from PIL import Image
import os
import sys

# print(os.getcwd())

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
# sys.path.append(ROOT_DIR)

from efficientvit.apps.utils import parse_unknown_args
from efficientvit.models.efficientvit.sam import (
    EfficientViTSamAutomaticMaskGenerator,
    EfficientViTSamPredictor,
)
from efficientvit.models.utils import build_kwargs_from_config
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


# def __init__(self, model_path, config, sam_checkpoint, sam_model, box_threshold=0.4, text_threshold=0.3):
def init():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available")
        exit()

    # rospy.loginfo("Loading models...")

    # # Building YOLO-World inference model
    # self.yolo_world_model = YOLO('yolov8l-worldv2.pt',task='detect')
    # self.box_threshold = box_threshold

    # Building EfficientViT-SAM inference model
    efficientvit_sam = (
        create_efficientvit_sam_model(
            "efficientvit-sam-l1", True, "../weights/efficientvit_sam_l1.pt"
        )
        .cuda()
        .eval()
    )
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

    image = np.array(Image.open("../test.png").convert("RGB"))
    efficientvit_sam_predictor.set_image(image)

    box = np.array([108.4318, 52.5968, 616.5026, 477.0328])

    masks, scores, logits = efficientvit_sam_predictor.predict(
        box=box, multimask_output=True
    )

    for i, mask in enumerate(masks):
        mask_uint8 = (mask * 255).astype(np.uint8)
        img = Image.fromarray(mask_uint8)
        filename = f"mask_{i}.png"
        img.save(filename)

    # result_masks = []
    # for box in xyxy:
    #     masks, scores, logits = self.sam_predictor.predict(
    #         box=box,
    #         multimask_output=True
    #     )
    #     index = np.argmax(scores)
    #     result_masks.append(masks[index])
    # return np.array(result_masks)

    # rospy.loginfo("Models are loaded")

    # # ros service
    # self.cv_bridge = CvBridge()
    # rospy.Service("vit_detection", VitDetection, self.callback)
    # rospy.loginfo("vit_detection service has started")


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


if __name__ == "__main__":
    init()
