import rospy
from cv_bridge import CvBridge
from std_msgs.msg import MultiArrayDimension

import cv2
import numpy as np
import torch

from ultralytics import YOLO

from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

from sensor_msgs.msg import Image
from yolo_evsam_ros.msg import AnnotationInfo, MaskInfo


class DetectSegmentation:
    def __init__(
        self, yolo_model_path, sam_model_path, sam_model_type, box_threshold=0.1
    ):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("No GPU available")
            exit()

        # rgb图像订阅
        self.image_sub = rospy.Subscriber(
            "color_to_process", Image, self.det_seg, queue_size=1
        )

        # 图像标注信息(labels,boxes) 和 掩码信息(labels,scores,segmasks) 发布者
        self.annotation_info_pub = rospy.Publisher(
            "annotation_info", AnnotationInfo, queue_size=1
        )
        self.mask_info_pub = rospy.Publisher("mask_info", MaskInfo, queue_size=1)

        self.cv_bridge = CvBridge()

        rospy.loginfo("Loading yolo evsam models...")

        # Building YOLO-World inference model
        self.yolo_world_model = YOLO(yolo_model_path, task="detect")
        self.box_threshold = box_threshold

        # Building EfficientViT-SAM inference model
        efficientvit_sam = (
            create_efficientvit_sam_model(sam_model_type, True, sam_model_path)
            .cuda()
            .eval()
        )
        self.sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

        # init_pub.publish(True)
        rospy.set_param("/yolo_evsam_ros_init", True)
        rospy.loginfo("yolo evsam models are loaded")

    def det_seg(self, image_msg):
        start_time = rospy.Time.now()

        # 将参数服务器里的det_seg_processing设置成True，防止定时器在前一张图像的检测分割还没完成时发布新图像
        rospy.set_param("det_seg_processing", True)

        # 提取图像时间戳，将图像消息转化为np数组形式
        time_stamp = image_msg.header.stamp
        image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # 从参数服务器获取检测类型（检测提示词）
        class_list = rospy.get_param("detection_prompt")
        # print(class_list)

        # 检测目标
        self.yolo_world_model.set_classes(class_list)
        results = self.yolo_world_model.predict(
            source=image, conf=self.box_threshold, verbose=False
        )

        # 如果未检测到目标，直接退出程序
        if results[0].boxes is None or len(results[0].boxes) == 0:
            rospy.loginfo("no detection")
            rospy.set_param("det_seg_processing", False)
            return

        # 生成class_id
        class_id = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        # 生成标签列表
        labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        # 获取边界框列表
        boxes = results[0].boxes.xyxy.cpu().numpy()
        # 获取置信度列表
        scores = results[0].boxes.conf.cpu().numpy()

        # 语义分割
        masks = self.segment(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), xyxy=boxes)
        masks = masks.astype(np.uint8)  # True -> 1, False -> 0
        # masks = (masks.astype(np.uint8)) * 255  # True -> 255(白色), False -> 0(黑色)

        # 将 N 张 H×W 的掩码图转换为 单个 H×W×N 的多通道 Image 消息
        masks_stacked = np.stack(masks, axis=-1)  # 变成 (H, W, N)

        # 测量检测分割的时间
        end_time = rospy.Time.now()
        seg_time = (end_time - start_time).to_sec() * 1000
        rospy.loginfo(f"detect+segment time: {seg_time:.1f} ms")

        # 发布图像标注信息
        annotation_info = AnnotationInfo()
        annotation_info.header.stamp = time_stamp
        annotation_info.class_id = class_id
        annotation_info.labels = labels
        annotation_info.boxes.layout.dim = [
            MultiArrayDimension(label="boxes", size=boxes.shape[0], stride=4)
        ]
        annotation_info.boxes.data = boxes.flatten().tolist()
        self.annotation_info_pub.publish(annotation_info)

        # 发布掩码信息
        mask_info = MaskInfo()
        mask_info.header.stamp = time_stamp
        mask_info.labels = labels
        mask_info.scores = scores.tolist()
        mask_info.segmasks = self.cv_bridge.cv2_to_imgmsg(
            masks_stacked, encoding="passthrough"
        )  # 8 位无符号整数，N 通道
        self.mask_info_pub.publish(mask_info)

        # We release the gpu memory
        torch.cuda.empty_cache()

        # 将参数服务器里的det_seg_processing设置成False，允许定时器发布新图像
        rospy.set_param("det_seg_processing", False)

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
    rospy.init_node("yolo_evsam_ros")

    model_path = rospy.get_param("~yolo_model_path")
    sam_checkpoint = rospy.get_param("~sam_checkpoint")
    sam_model = rospy.get_param("~sam_model")
    box_threshold = rospy.get_param("~box_threshold")

    DetectSegmentation(model_path, sam_checkpoint, sam_model, box_threshold)
    rospy.spin()
