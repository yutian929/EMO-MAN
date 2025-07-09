#!/usr/bin/env python3
import rospy
import clip
import torch
from PIL import Image
import numpy as np
import os
from cv_bridge import CvBridge
from semantic_map_pkg.srv import Clip, ClipResponse
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension


class CLIPNode:
    def __init__(self):
        rospy.init_node("clip_node")
        model_name = rospy.get_param("~model_name", "ViT-B/16")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            rospy.loginfo(f"CLIP model {model_name} loaded on {self.device}")
        except Exception as e:
            rospy.logerr(f"Failed to load CLIP model: {str(e)}")
            rospy.logerr(
                f"Please make sure your model name is correct. Model list: {clip.available_models()}"
            )
            return

        self.bridge = CvBridge()

        # Create ROS services
        self.clip_server = rospy.Service(
            "/clip",
            Clip,
            self.clip_callback,
        )

        rospy.loginfo("CLIP node initialized complete.")

    def encode_text(self, text):
        """Encode a single text"""
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        return text_features.cpu().numpy()

    def encode_image_from_ros_msg(self, ros_image):
        """Encode a single image from ROS Image message"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
            # Convert to PIL Image
            pil_image = Image.fromarray(cv_image)
            # Preprocess and encode
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
            return image_features.cpu().numpy()
        except Exception as e:
            rospy.logerr(f"ROS image encoding failed: {str(e)}")
            return None

    def encode_images_from_paths(self, image_paths):
        """Batch encode multiple images (efficient GPU batch processing)"""
        images = []
        valid_paths = []

        # Load and preprocess images
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                image = self.preprocess(image)
                images.append(image)
                valid_paths.append(path)
            except Exception as e:
                rospy.logwarn(f"Skipping invalid image {path}, error: {str(e)}")

        if not images:
            return None, []

        # Combine into a batch tensor
        batch = torch.stack(images).to(self.device)

        # Extract features
        with torch.no_grad():
            batch_features = self.model.encode_image(batch)

        return batch_features.cpu().numpy(), valid_paths

    def encode_images_from_ros_msgs(self, ros_images):
        """Batch encode multiple images from ROS Image messages"""
        images = []

        # Load and preprocess images
        for ros_image in ros_images:
            try:
                # Convert ROS Image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
                # Convert to PIL Image
                pil_image = Image.fromarray(cv_image)
                image = self.preprocess(pil_image)
                images.append(image)
            except Exception as e:
                rospy.logwarn(f"Invalid ROS image, error: {str(e)}")
                return None

        if not images:
            return None

        # Combine into a batch tensor
        batch = torch.stack(images).to(self.device)

        # Extract features
        with torch.no_grad():
            batch_features = self.model.encode_image(batch)

        array = Float32MultiArray()
        array.layout = MultiArrayLayout()
        array.layout.dim.append(MultiArrayDimension("rows", len(images), 512))
        array.layout.dim.append(MultiArrayDimension("cols", 512, 1))
        array.data = batch_features.cpu().numpy().flatten().tolist()
        return array

    def text_match_images(self, text, image_paths):
        """Calculate similarity between text and multiple images, return similarities"""
        # Encode text
        text_features = self.encode_text(text)
        if text_features is None:
            return []

        # Batch encode images
        batch_features, valid_paths = self.encode_images_from_paths(image_paths)
        if batch_features is None:
            return []

        # Calculate similarity matrix [num_images x 1]
        similarities = batch_features @ text_features.T
        # 展平成1维列表并转成list
        return similarities.flatten().tolist(), valid_paths

    def clip_callback(self, req):
        """
        Service callback to handle incoming requests
        # request params
        string task  # 要让CLIP做的任务
        string text  # 文本，用于文本特征提取任务，和单文本-多图像匹配任务
        sensor_msgs/Image image  # 图像，用于单个图像特征提取任务
        string[] image_paths  # 图像路径，用于单文本-多图像匹配任务
        ---
        # response params
        bool success
        string message
        float32[] data  # 单个文本特征，单个图像特征，单文本-多图像匹配任务的相似度
        Float32MultiArray array  # 多个图像特征
        """
        task = req.task
        if task == "encode_text":
            text = req.text
            text_features = self.encode_text(text)
            if text_features is not None:
                return ClipResponse(
                    success=True,
                    message="Text encoding successful",
                    data=text_features[0],
                    array=Float32MultiArray(),
                )
            else:
                return ClipResponse(
                    False, "Text encoding failed", [], Float32MultiArray()
                )
        elif task == "encode_image_from_ros_msg":
            ros_image = req.image
            image_features = self.encode_image_from_ros_msg(ros_image)
            if image_features is not None:
                return ClipResponse(
                    success=True,
                    message="Image encoding successful",
                    data=image_features[0],
                    array=Float32MultiArray(),
                )
            else:
                return ClipResponse(
                    False, "Image encoding failed", [], Float32MultiArray()
                )
        elif task == "text_match_images":
            text = req.text
            image_paths = req.image_paths
            similarities, valid_paths = self.text_match_images(text, image_paths)
            return ClipResponse(
                success=True,
                message="Text-image matching successful",
                data=similarities,
                array=Float32MultiArray(),
            )
        elif task == "encode_images_from_ros_msg":
            ros_images = req.images
            array = self.encode_images_from_ros_msgs(ros_images)
            if array is not None:
                return ClipResponse(
                    success=True,
                    message="Multiple images encoding successful",
                    data=[],
                    array=array,
                )
            else:
                return ClipResponse(
                    False, "Images encoding failed", [], Float32MultiArray()
                )
        else:
            return ClipResponse(False, "Invalid task", [], Float32MultiArray())


if __name__ == "__main__":
    try:
        node = CLIPNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
