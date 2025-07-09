#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge
from lib.semantic_map_database import SemanticMapDatabase
from lib.semantic_map_transfer import MapCoordinateTransformer
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    Point,
    Pose,
    Quaternion,
)
from tf.transformations import quaternion_from_euler
from llm_pkg.srv import LLMChat, LLMChatRequest, LLMChatResponse
from llm_analyzer.llm_analyzer import LLMAnalyzer
from vlm_pkg.srv import VLMChat, VLMChatRequest, VLMChatResponse
from vlm_analyzer.vlm_analyzer import VLMAnalyzer
from semantic_map_pkg.srv import Guide, GuideResponse, Clip, ClipResponse
import numpy as np
import pickle
import cv2
import json
import os
import re
import ast


class SemanticMapGuide:
    def __init__(self):
        rospy.init_node("semantic_map_guide_node")

        # 数据库配置
        self.db_path = rospy.get_param("~db_path", "semantic_map.db")
        self.last_seen_imgs_dir = rospy.get_param(
            "~last_seen_imgs_dir", "last_seen_imgs"
        )
        self.database = SemanticMapDatabase(
            self.db_path, self.last_seen_imgs_dir, renew_db=False
        )
        self.semantic_categories_json_path = rospy.get_param(
            "~semantic_categories_json_path",
            os.path.join(os.path.dirname(__file__), "semantic_categories.json"),
        )
        self.reload_semantic_categories()
        # ROS配置
        ## opencv
        self.bridge = CvBridge()
        ## 订阅机器人位姿
        self.robot_pose = None
        rospy.Subscriber(
            "/robot_pose_ekf/odom_combined",
            PoseWithCovarianceStamped,
            self.pose_callback,
        )
        ## 订阅地图
        self.map_cache_path = rospy.get_param(
            "~map_cache_path",
            os.path.join(os.path.dirname(__file__), "map_cache.pkl"),
        )
        self.map_transformer = self.load_map_transformer(self.map_cache_path)
        self.map_sub = rospy.Subscriber(
            "/map", OccupancyGrid, self.map_callback, queue_size=1
        )
        ## CLIP服务
        rospy.loginfo("Waiting for CLIP service...")
        rospy.wait_for_service("clip")
        self.clip_client = rospy.ServiceProxy("clip", Clip)
        rospy.loginfo("CLIP service initialized complete.")
        ## LLM服务
        rospy.loginfo("Waiting for LLM service...")
        rospy.wait_for_service("llm_chat")
        self.llm_chat_client = rospy.ServiceProxy("llm_chat", LLMChat)
        rospy.wait_for_service("llm_reason")
        self.llm_reason_client = rospy.ServiceProxy("llm_reason", LLMChat)
        self.llm_analyzer = LLMAnalyzer()
        rospy.loginfo("LLM service initialized complete.")
        ## VLM服务
        rospy.loginfo("Waiting for VLM service...")
        rospy.wait_for_service("vlm_chat")
        self.vlm_chat_client = rospy.ServiceProxy("vlm_chat", VLMChat)
        self.vlm_analyzer = VLMAnalyzer()
        rospy.loginfo("VLM service initialized complete.")
        ## Guide服务
        service_guide_server = rospy.get_param(
            "~service_guide_server", "semantic_map_guide"
        )
        self.guide_server = rospy.Service(
            service_guide_server, Guide, self.guide_callback
        )

        rospy.loginfo("semantic_map_guide_node initialized complete.")
        # self.debug()

    def debug(self):
        rospy.loginfo("Debugging information...")
        # --------------------------------------------------
        # self.get_ideal_operation_direction("washing-machine@1", "打开洗衣机门, 取出衣服，关闭洗衣机门")
        # --------------------------------------------------
        # def test(origin_cmd):
        #     self.reload_semantic_categories()
        #     content = f"以下是语义对象列表：{self.semantic_categories}。用户的指令是：{origin_cmd}"
        #     print(f"{'-'*20}reason{'-'*20}")
        #     print(f"ask: {content}")
        #     llm_reason_res = self.ask_llm_reason("task_plan_reason", content)
        #     print(f"res: {llm_reason_res.response}")
        #     task_plans = self.analyze_llm_response(
        #         "task_plan_reason", llm_reason_res.response
        #     )
        #     print(f"task_plans: {task_plans}")
        #     # task_plan: [{'navi': 'table', 'op': '找到并夹取桌子上的饮料'}, {'navi': 'refrigerator', 'op': '打开冰箱门，放入饮料，关闭冰箱门'}]
        #     for task_plan in task_plans:
        #         print(f"{'-'*20}chat{'-'*20}")
        #         content = f"要查找的对象是{task_plan['navi']}，用户的指令是{task_plan['op']}"
        #         llm_chat_res = self.ask_llm_chat("category_or_language_chat", content)
        #         print(f"ask: {content}")
        #         print(f"res: {llm_chat_res.response}")
        #         navi_object_properties = self.analyze_llm_response(
        #             "category_or_language_chat", llm_chat_res.response
        #         )
        #         print(f"navi_object_properties: {navi_object_properties}")
        #         # navi_object_properties: {'type': 'language', 'object': 'refrigerator', 'description': '用于放入饮料瓶的冰箱'}
        # test("把桌子上的饮料放到冰箱里")
        # --------------------------------------------------

    def load_map_transformer(self, cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                map_transformer = pickle.load(f)
            rospy.loginfo("Map transformer loaded from cache.")
            return map_transformer
        else:
            rospy.logwarn("Map transformer Cache path does not exist.")
            return None

    def save_map_transformer_to_cache(self):
        """将地图转换器保存到缓存"""
        if self.map_transformer:
            try:
                rospy.loginfo(f"Saving map transformer to cache: {self.map_cache_path}")
                with open(self.map_cache_path, "wb") as f:
                    pickle.dump(self.map_transformer, f)
                return True
            except Exception as e:
                rospy.logwarn(f"Failed to save map transformer to cache: {str(e)}")
        return False

    def reload_semantic_categories(self):
        self.semantic_categories = json.load(
            open(self.semantic_categories_json_path, "r")
        )["categories"]

    def map_callback(self, msg):
        """地图数据回调函数"""
        self.map_transformer = MapCoordinateTransformer(
            msg.info.resolution,
            msg.info.width,
            msg.info.height,
            msg.info.origin.position,
            msg.data,
        )
        # self.save_map_transformer_to_cache()

    def pose_callback(self, msg):
        """机器人位姿回调"""
        self.robot_pose = msg.pose.pose

    def get_distance(self, base_pose, semantic_obj):
        # base_pose: [x, y, z, qx, qy, qz, qw]
        base_x, base_y, base_z, _, _, _, _ = base_pose
        # [x_min, y_min, z_min, x_max, y_max, z_max]
        semantic_obj_bbox = semantic_obj["bbox"]
        obj_x = (semantic_obj_bbox[0] + semantic_obj_bbox[3]) / 2
        obj_y = (semantic_obj_bbox[1] + semantic_obj_bbox[4]) / 2
        obj_z = (semantic_obj_bbox[2] + semantic_obj_bbox[5]) / 2
        distance = (
            (base_x - obj_x) ** 2 + (base_y - obj_y) ** 2 + (base_z - obj_z) ** 2
        ) ** 0.5
        return distance

    def lang_imgs_match(self, lang, img_paths):
        """
        语言描述和图片匹配
        调用CLIP服务中的text_match_images方法
        返回匹配度
        """
        if type(img_paths) is not list:
            img_paths = [img_paths]
        response = self.clip_client(
            task="text_match_images",
            text=lang,
            image_paths=img_paths,
        )
        if response.success:
            # 匹配match对
            match_pairs = []
            for i, score in enumerate(response.data):
                match_pairs.append((img_paths[i], score))
            return match_pairs
        else:
            rospy.logerr(f"CLIP service call failed: {response.message}")
            return []

    def find_semantic_object_by_category(self, category) -> str:
        """
        通过类别查找语义对象
        返回距离最近的那个语义对象
        """
        # # semantic_object = {"label":str, "bbox": list(float), "time_stamp": str, "x": list(float), "y": list(float), "z": list(float), "rgb": list(int)}
        semantic_objects = self.database._get_entries_by_category(category)
        if not semantic_objects:
            rospy.logwarn(f"No semantic object found by category: {category}")
            return None
        # base_pose = [x, y, z, qx, qy, qz, qw]
        if not self.robot_pose:
            rospy.logwarn("No self.robot_pose found.")
            return None
        base_pose = [
            self.robot_pose.position.x,
            self.robot_pose.position.y,
            self.robot_pose.position.z,
            self.robot_pose.orientation.x,
            self.robot_pose.orientation.y,
            self.robot_pose.orientation.z,
            self.robot_pose.orientation.w,
        ]
        min_distance = float("inf")
        nearest_object = None
        for obj in semantic_objects:
            distance = self.get_distance(base_pose, obj)
            if distance < min_distance:
                min_distance = distance
                nearest_object = obj
        return nearest_object["label"]

    def find_semantic_object_by_language(self, category, language) -> str:
        """
        通过语言描述查找语义对象
        返回匹配度最高的的那个语义对象
        """
        self.reload_semantic_categories()
        if category in self.semantic_categories:
            label_img_paths = self.database.get_img_paths_by_category(category)
            if label_img_paths:
                best_label = None
                best_score = 0
                for label, img_paths in label_img_paths.items():
                    # match_pairs: [(img_path, match_score), ...]
                    match_pairs = self.lang_imgs_match(language, img_paths)
                    if not match_pairs:
                        continue
                    label_score = sum(
                        [match_pair[1] for match_pair in match_pairs]
                    ) / len(match_pairs)
                    if label_score > best_score:
                        best_score = label_score
                        best_label = label
                return best_label
        return None

    def connected_components_analysis(self, binary_search_area):
        """
        连通域分析
        返回连通域的中心点坐标
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_search_area, connectivity=8, ltype=cv2.CV_32S
        )
        # 寻找最大连通域（跳过背景0）
        max_area = 0
        max_label = -1
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] > max_area:
                max_area = stats[label, cv2.CC_STAT_AREA]
                max_label = label
        if max_label == -1:
            return None
        # 获取最大连通域的中心点
        cx, cy = centroids[max_label]
        return (cx, cy)

    def process_subimgs(self, subimgs_paths):
        """
        处理子图
        返回合并后的总图路径列表
        """
        # 读取子图
        subimgs = []
        for idx, img_path in enumerate(subimgs_paths):
            img = cv2.imread(img_path)
            if img is None:
                rospy.logwarn(f"Failed to read image: {img_path}")
                continue
            # 在左上角标注图片顺序
            cv2.putText(
                img,
                f"sub_img{idx + 1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
            )
            # 图片左右加入黑线以做区分
            h, w = img.shape[:2]
            black_line = np.zeros((h, 5, 3), dtype=np.uint8)
            img = np.concatenate((black_line, img), axis=1)
            img = np.concatenate((img, black_line), axis=1)
            subimgs.append(img)
        # 合并子图
        if not subimgs:
            return []
        total_img = np.concatenate(subimgs, axis=1)
        # 保存总图, 并返回绝对路径
        total_img_path = os.path.join(self.last_seen_imgs_dir, "total_img.png")
        cv2.imwrite(total_img_path, total_img)
        # 释放内存
        for img in subimgs:
            del img
        del total_img
        return os.path.abspath(total_img_path)

    def ask_llm_chat(self, type, content):
        """Handle LLM chat requests."""
        llm_chat_req = LLMChatRequest(type=type, content=content)
        llm_chat_res = self.llm_chat_client(llm_chat_req)
        return llm_chat_res

    def ask_llm_reason(self, type, content):
        """Handle LLM reason requests."""
        llm_reason_req = LLMChatRequest(type=type, content=content)
        llm_reason_res = self.llm_reason_client(llm_reason_req)
        return llm_reason_res

    def ask_vlm_chat(self, type, content, image_path):
        """Handle VLM chat requests."""
        vlm_chat_req = VLMChatRequest(type=type, content=content, image_path=image_path)
        vlm_chat_res = self.vlm_chat_client(vlm_chat_req)
        return vlm_chat_res

    def get_ideal_operate_direction(self, label, op_cmd):
        """
        获取理想的操作方向
        :param label: 语义对象的标签
        :param op_cmd: 操作指令
        :return: 理想的操作方向 eg: 'xp', 'yp', 'xn', 'yn'
        """
        # {'xp': ['xxx/bed@1@20250415192519.png', 'xxx/bed@1/xp/bed@1@20250415192520.png', 'xxx/bed@1/xp/bed@1@20250415192701.png'], 'yp': ['xxx/bed@1/yp/bed@1@20250415192556.png'], 'xn': [], 'yn': []}
        act_imgs_paths = self.database.get_img_paths_by_label(label)
        record_act_dirs = []
        send_subimgs_paths = []
        for act_dir, img_paths in act_imgs_paths.items():
            if not img_paths:
                continue
            # 取出最后一张图，也是最新拍摄的图片
            send_subimgs_paths.append(img_paths[-1])
            record_act_dirs.append(act_dir)
        # 处理子图
        total_img_path = self.process_subimgs(send_subimgs_paths)
        # 调用VLM服务
        content = f"这里一共有{len(send_subimgs_paths)}张子图，机器人要执行的操作是{op_cmd}"
        vlm_chat_res = self.ask_vlm_chat(
            type="manipulation_oriented_vision",
            content=content,
            image_path=total_img_path,
        )
        chosen_sub_img_number = self.vlm_analyzer.analyze(
            "manipulation_oriented_vision", vlm_chat_res.response
        )  # int
        # >>> DEBUG >>>
        rospy.loginfo(f"{'-'*20}manipulation_oriented_vision{'-'*20}")
        rospy.loginfo(f"ask: {content} img_path: {total_img_path}")
        rospy.loginfo(f"res: {vlm_chat_res.response}")
        rospy.loginfo(f"chosen_sub_img_number: {chosen_sub_img_number}")
        # <<< DEBUG <<<
        if chosen_sub_img_number is None:
            rospy.logwarn("Failed to analyze vlm response.")
            chosen_sub_img_number = 1
        if chosen_sub_img_number > len(send_subimgs_paths):
            rospy.logwarn(
                f"Chosen sub image number {chosen_sub_img_number} exceeds available images."
            )
            chosen_sub_img_number = 1
        # 解析VLM服务返回的结果
        if vlm_chat_res.success:
            ideal_idx = int(chosen_sub_img_number) - 1
            ideal_act_dir = record_act_dirs[ideal_idx]
            rospy.loginfo(f"chosen_action_direction: {ideal_act_dir}")
            return ideal_act_dir  # str
        else:
            rospy.logerr(f"Call VLM service error: {vlm_chat_res.response}")
            return None

    def get_possible_navi_goals(self, label, epd_dis=1.1):
        """
        获取可能的导航点和方向
        :param label: 语义对象的标签
        :param epd_dis: 外扩距离（单位：米）
        :return: 可能的导航点和方向. eg. {"xp": [w_nav_x, w_nav_y, 0, q[0], q[1], q[2], q[3]]}
        """
        # 获取语义对象信息
        navi_obj = self.database._get_entry_by_label(label)
        w_bbox = navi_obj["bbox"]  # [x_min, y_min, z_min, x_max, y_max, z_max]
        w_act = navi_obj.get("act", [1, 1, 1, 1])  # 默认可从所有方向接近
        epd_dis = epd_dis  # 外扩距离（单位：米）

        # 计算目标物体中心世界坐标（投影到地面）
        w_x_min = w_bbox[0]
        w_y_min = w_bbox[1]
        w_x_max = w_bbox[3]
        w_y_max = w_bbox[4]

        # 转换到地图像素坐标系
        m_x_min, m_y_min = self.map_transformer.world2map_p(w_x_min, w_y_min)
        m_x_max, m_y_max = self.map_transformer.world2map_p(w_x_max, w_y_max)
        m_epd_dis = self.map_transformer.world2map_l(epd_dis)

        # 转换到图片像素坐标系
        p_x_min, p_y_min = self.map_transformer.map2img_p(m_x_min, m_y_min)
        p_x_max, p_y_max = self.map_transformer.map2img_p(m_x_max, m_y_max)
        if p_y_min > p_y_max:  # 图片坐标系y轴翻转
            p_y_min, p_y_max = p_y_max, p_y_min
        p_epd_dis = m_epd_dis

        # 定义四个外扩矩形,是针对图片像素坐标系下的
        rectangles = [
            {
                "name": "xp",
                "act": w_act[0],
                "p_rx_min": p_x_min - p_epd_dis,
                "p_ry_min": p_y_min,
                "p_rx_max": p_x_min,
                "p_ry_max": p_y_max,
            },
            {
                "name": "yp",
                "act": w_act[1],
                "p_rx_min": p_x_min,
                "p_ry_min": p_y_max,
                "p_rx_max": p_x_max,
                "p_ry_max": p_y_max + p_epd_dis,
            },
            {
                "name": "xn",
                "act": w_act[2],
                "p_rx_min": p_x_max,
                "p_ry_min": p_y_min,
                "p_rx_max": p_x_max + p_epd_dis,
                "p_ry_max": p_y_max,
            },
            {
                "name": "yn",
                "act": w_act[3],
                "p_rx_min": p_x_min,
                "p_ry_min": p_y_min - p_epd_dis,
                "p_rx_max": p_x_max,
                "p_ry_max": p_y_min,
            },
        ]

        # 存储导航点的字典
        possible_navi_goals = {}

        # 遍历所有有效矩形区
        for rect in rectangles:
            if not rect["act"]:
                continue
            # 取出该语义对象存储的在act方向上的last_seen_img路径

            # 搜索范围, 确保矩形范围在地图内
            p_rx_min = max(0, rect["p_rx_min"])
            p_rx_max = min(self.map_transformer.width, rect["p_rx_max"])
            p_ry_min = max(0, rect["p_ry_min"])
            p_ry_max = min(self.map_transformer.height, rect["p_ry_max"])
            if p_rx_min >= p_rx_max or p_ry_min >= p_ry_max:
                continue
            # 提取矩形区域的二值地图, 其余全部是0
            search_area = self.map_transformer.image_data_binary[
                p_ry_min:p_ry_max, p_rx_min:p_rx_max
            ]

            # 连通域分析
            cx, cy = self.connected_components_analysis(search_area)
            if cx is None or cy is None:
                continue
            p_nav_x = int(cx + p_rx_min)
            p_nav_y = int(cy + p_ry_min)

            # 转换到世界坐标
            m_nav_x, m_nav_y = self.map_transformer.img2map_p(p_nav_x, p_nav_y)
            w_nav_x, w_nav_y = self.map_transformer.map2world_p(m_nav_x, m_nav_y)

            # 计算导航方向, 导航点指向目标物体中心
            w_obj_x = (w_x_min + w_x_max) / 2
            w_obj_y = (w_y_min + w_y_max) / 2
            dx = w_obj_x - w_nav_x
            dy = w_obj_y - w_nav_y
            yaw = np.arctan2(dy, dx)
            q = quaternion_from_euler(0, 0, yaw)

            # 加入possible_navi_goals字典
            possible_navi_goals[rect["name"]] = [
                w_nav_x,
                w_nav_y,
                0,
                q[0],
                q[1],
                q[2],
                q[3],
            ]

        return possible_navi_goals  # dict

    def get_ideal_navi_goals(self, label, op_cmd):
        # 先获取针对操作任务的理想的导航方向
        ideal_act_dir = self.get_ideal_operate_direction(label, op_cmd)  # eg: 'xp'
        # 再获取所有可能的导航点
        possible_navi_goals = self.get_possible_navi_goals(
            label, epd_dis=1.1
        )  # eg: {'xp': [w_nav_x, w_nav_y, 0, q[0], q[1], q[2], q[3]]}
        if not possible_navi_goals:
            rospy.logwarn(f"No possible navigation goals found for label: {label}")
            return []
        # 对所有可能的导航点进行排序，优先选择理想的导航方向,也就是key=ideal_act_dir的
        sorted_navi_goals = sorted(
            possible_navi_goals.items(),
            key=lambda item: item[0] == ideal_act_dir,
            reverse=True,
        )
        # 取出理想的导航点
        ideal_nav_goals_list = []
        for name, (w_nav_x, w_nav_y, w_nav_z, q0, q1, q2, q3) in sorted_navi_goals:

            # 创建 Pose 消息
            nav_goal = Pose(
                position=Point(w_nav_x, w_nav_y, w_nav_z),
                orientation=Quaternion(q0, q1, q2, q3),
            )
            ideal_nav_goals_list.append(nav_goal)

        if not ideal_nav_goals_list:
            rospy.logwarn(f"No valid navigation goals found for label: {label}")
            return []

        return ideal_nav_goals_list

    def guide_callback(self, req):
        """
        语义地图引导服务回调函数
        :param req: 请求消息, 包含有以下字段:
                    category: 导航目标对象类, eg: "refrigerator"
                    op_cmd: 操作指令, eg: "打开冰箱门, 放置苹果, 关闭冰箱门"
                    origin_cmd: 最初的人为指令, eg: "把苹果放到冰箱里"
        :return: 响应消息, 包含语义对象的标签和导航点列表
        """
        try:
            # 检查是否有指定的label和op_cmd
            if not (req.category and req.op_cmd and req.origin_cmd):
                return GuideResponse(
                    success=False,
                    message="No category or op_cmd specified.",
                    label="",
                )
            # 先找到具体的语义对象
            content = f"要找的对象是{req.category}，用户的指令是{req.origin_cmd}"
            llm_chat_res = self.ask_llm_chat("category_or_language_chat", content)
            navi_obj_properties = self.llm_analyzer.analyze(
                "category_or_language_chat", llm_chat_res.response
            )  # dict
            if type(navi_obj_properties) is str:
                return GuideResponse(
                    success=False,
                    message=f"Failed to parse navi object properties: {navi_obj_properties}",
                    label="",
                )
            # >>> DEBUG >>>
            rospy.loginfo(f"{'-'*20}category_or_language_chat{'-'*20}")
            rospy.loginfo(f"ask: {content}")
            rospy.loginfo(f"res: {llm_chat_res.response}")
            rospy.loginfo(f"navi_object_properties: {navi_obj_properties}")
            # <<< DEBUG <<<
            if not llm_chat_res.success:
                return GuideResponse(
                    success=False,
                    message=f"Call LLM chat error: {llm_chat_res.response}",
                    label="",
                )

            navi_type = navi_obj_properties.get("type", "")
            navi_obj_category = navi_obj_properties.get("object", "").strip("'\"")
            navi_obj_description = navi_obj_properties.get("description", "").strip(
                "'\""
            )

            # 通过navi_type和navi_obj_category，先找出语义对象的label
            if navi_type == "language":
                # 通过语言描述查找语义对象
                label = self.find_semantic_object_by_language(
                    navi_obj_category, navi_obj_description
                )
            elif navi_type == "category":
                # 通过类别查找语义对象
                label = self.find_semantic_object_by_category(navi_obj_category)
            else:
                return GuideResponse(
                    success=False,
                    message=f"Unsupported navi type: {navi_type}",
                    label="",
                )
            if not label:
                return GuideResponse(
                    success=False,
                    message=f"No semantic object found for {req.category}",
                    label="",
                )

            ideal_navi_goals = self.get_ideal_navi_goals(label, req.op_cmd)
            if ideal_navi_goals:
                return GuideResponse(
                    success=True,
                    message=f"Semantic object found: {label}",
                    label=label,
                    nav_goals=ideal_navi_goals,
                )
            else:
                return GuideResponse(
                    success=False,
                    message="Semantic object found, but no ideal navigation goals found.",
                    label=label,
                )
        except Exception as e:
            rospy.logerr(f"Guide callback error: {str(e)}")
            return GuideResponse(
                success=False,
                message=f"Guide callback error: {str(e)}",
                label="",
            )


if __name__ == "__main__":
    try:
        manager = SemanticMapGuide()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS interrupt exception.")
