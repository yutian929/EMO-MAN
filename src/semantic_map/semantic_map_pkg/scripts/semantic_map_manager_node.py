#!/usr/bin/env python
import rospy
import ast
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from lib.semantic_map_database import SemanticMapDatabase
from semantic_map_pkg.msg import SemanticObject
from semantic_map_pkg.srv import Show, ShowResponse
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import TransformStamped


class SemanticMapManager:
    def __init__(self):
        rospy.init_node("semantic_map_manager_node")

        # 数据库配置
        self.db_path = rospy.get_param("~db_path", "semantic_map.db")
        self.renew_db = rospy.get_param("~renew_db", False)
        self.last_seen_imgs_dir = rospy.get_param(
            "~last_seen_imgs_dir", "last_seen_imgs"
        )
        self.feature_match_threshold = rospy.get_param("~feature_match_threshold", 0.9)
        rospy.loginfo(f"setting semantic map dataset:")
        rospy.loginfo(f"* db path = {self.db_path}")
        rospy.loginfo(f"* last_seen_imgs_dir = {self.last_seen_imgs_dir}")
        rospy.loginfo(f"* feature_match_threshold = {self.feature_match_threshold}")
        if self.renew_db:
            rospy.logwarn(f"* renew_db = {self.renew_db}")
        else:
            rospy.loginfo(f"* renew_db = {self.renew_db}")
        self.database = SemanticMapDatabase(
            self.db_path, self.last_seen_imgs_dir, 3, self.renew_db
        )

        # ROS配置
        ## 语义对象订阅器
        self.sub = rospy.Subscriber(
            "/semantic_object", SemanticObject, self.semantic_object_callback
        )
        self.bridge = CvBridge()
        ## 语义地图发布器
        self.pub = rospy.Publisher("/semantic_map", PointCloud2, queue_size=10)
        ## 语义地图显示服务
        self.show_markers = rospy.get_param("~show_markers", True)
        self.semantic_map_frame_id = rospy.get_param("~semantic_map_frame_id", "map")
        self.service_show = rospy.Service("/semantic_map_show", Show, self.handle_show)
        self.bbox_pub = rospy.Publisher(
            "/semantic_map_bbox", MarkerArray, queue_size=10
        )
        ## BBox可视化参数
        self.show_bbox = rospy.get_param("~show_bbox", True)
        bbox_colora_str = rospy.get_param("~bbox_colora", "[0.0, 1.0, 0.0, 0.5]")
        bbox_colora = self.colora_param_decode(bbox_colora_str)
        self.bbox_color = bbox_colora[:3]
        self.bbox_alpha = bbox_colora[3]
        self.bbox_line_width = rospy.get_param("~bbox_line_width", 0.01)
        ## 箭头可视化参数
        self.show_arrow = rospy.get_param("~show_arrow", True)
        arrow_color_str = rospy.get_param(
            "~arrow_colora", "[1.0, 0.0, 0.0, 1.0]"
        )  # 默认红色
        arrow_colora = self.colora_param_decode(arrow_color_str)
        self.arrow_color = arrow_colora[:3]
        self.arrow_alpha = arrow_colora[3]
        self.arrow_length = rospy.get_param("~arrow_length", 0.3)  # 默认箭头长度
        self.arrow_scale_x = rospy.get_param("~arrow_scale_x", 0.05)  # 箭头杆直径
        self.arrow_scale_y = rospy.get_param("~arrow_scale_y", 0.1)  # 箭头头部直径
        self.arrow_scale_z = rospy.get_param("~arrow_scale_z", 0.1)  # 箭头头部长度
        ## 标签文本可视化参数
        self.show_label = rospy.get_param("~show_label", True)
        label_color_str = rospy.get_param("~label_colora", "[1.0, 1.0, 1.0, 1.0]")
        label_colora = self.colora_param_decode(label_color_str)
        self.label_color = label_colora[:3]
        self.label_alpha = label_colora[3]
        self.label_scale = rospy.get_param("~label_scale", 0.1)  # 默认字体大小
        self.label_offset = rospy.get_param("~label_offset", 0.1)  # 默认偏移量
        ## 订阅机器人位姿
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.base_link_father = rospy.get_param("~base_link_father", "map")
        self.base_link_child = rospy.get_param("~base_link_child", "camera_link")
        rospy.loginfo(f"semantic map show params: ")
        rospy.loginfo(f"* show_markers = {self.show_markers}")
        rospy.loginfo(f"* show_bbox = {self.show_bbox}")
        rospy.loginfo(f"* show_arrow = {self.show_arrow}")
        rospy.loginfo(f"* show_label = {self.show_label}")
        rospy.loginfo("semantic_map_manager_node initialized complete.")

    def colora_param_decode(self, colora_str):
        """
        将字符串转换为列表
        :param colora_str: 字符串格式的颜色参数
        :return: 转换后的列表
        """
        try:
            colora = ast.literal_eval(colora_str)
            # 验证数据类型和长度
            if not (
                isinstance(colora, list)
                and len(colora) == 4
                and all(isinstance(x, float) for x in colora)
            ):
                raise ValueError
        except (SyntaxError, ValueError) as e:
            rospy.logwarn(
                "Invalid colora %s, using default [0.0, 1.0, 0.0, 0.5]. Error: %s",
                colora_str,
                e,
            )
            colora = [0.0, 1.0, 0.0, 0.5]
        return colora

    def update_db(
        self,
        category,
        bbox,
        count,
        x_list,
        y_list,
        z_list,
        rgb_list,
        cv_image,
        act_list,
        crop_feature,
    ):
        """
        更新数据库:检测是否有冲突，再选择是合并 or 新增 or 移除再新增
        act_list: one-hot编码，表示方向 [+x, +y, -x, -y]
        """
        # 首先提取同类别的所有条目
        # entries = [{"label":str, "bbox": list(float), "time_stamp": str, "x": list(float), "y": list(float), "z": list(float), "rgb": list(int)}, "act": list(int)}, "feature": list(int)..."]
        entries = self.database._get_entries_by_category(category)
        # 如果没有同类别的条目，直接插入
        if not entries:  # 新类别 -> 插入
            label = f"{category}@1"
            self.database._update_entry(
                label, bbox, x_list, y_list, z_list, rgb_list, act_list, crop_feature
            )
            self.database._save_last_seen_img(label, cv_image, act_list)
            return
        else:  # 如果有同类别的条目，用bbox去匹配每一个条目，看是否是同一个语义对象
            for entry in entries:
                label_entry = entry["label"]
                bbox_entry = entry["bbox"]
                act_entry = entry["act"]
                crop_feature_entry = entry["feature"]
                if self.bbox_match(bbox, bbox_entry):  # 同位置 -> 合并
                    x_merged = x_list + entry["x"]
                    y_merged = y_list + entry["y"]
                    z_merged = z_list + entry["z"]
                    rgb_merged = rgb_list + entry["rgb"]
                    bbox_merged = self.bbox_merge(bbox, bbox_entry)
                    act_merged = self.act_merge(act_list, act_entry)
                    self.database._update_entry(
                        label_entry,
                        bbox_merged,
                        x_merged,
                        y_merged,
                        z_merged,
                        rgb_merged,
                        act_merged,
                        crop_feature,
                    )
                    self.database._save_last_seen_img(label_entry, cv_image, act_list)
                    return
                else:
                    if self.feature_match(
                        crop_feature, crop_feature_entry
                    ):  # 不同位置，相同特征 -> 原来物体发生了位移 -> 替换
                        rospy.logwarn(
                            f"Object moved, updating entry {label_entry} with new position."
                        )
                        self.database._update_entry(
                            label_entry,  # 原有标签
                            bbox,
                            x_list,
                            y_list,
                            z_list,
                            rgb_list,
                            act_list,
                            crop_feature,
                        )
                        self.database._save_last_seen_img(
                            label_entry, cv_image, act_list
                        )
                        return

            # 如果所有条目都不匹配，插入新的语义对象
            existed_label_ids = [entry["label"].split("@")[1] for entry in entries]
            new_label_id = int(max(existed_label_ids)) + 1
            label = f"{category}@{new_label_id}"
            self.database._update_entry(
                label, bbox, x_list, y_list, z_list, rgb_list, act_list, crop_feature
            )
            self.database._save_last_seen_img(label, cv_image, act_list)
            return

    def act_merge(self, list1, list2):
        # 按位或操作
        assert len(list1) == len(
            list2
        ), f"ERROR: act_merge, dimensions do not match {len(list1)} != {len(list2)}"
        return [max(a, b) for a, b in zip(list1, list2)]

    def bbox_merge(self, bbox1, bbox2):
        """合并两个bbox"""
        x_min1, y_min1, z_min1, x_max1, y_max1, z_max1 = bbox1
        x_min2, y_min2, z_min2, x_max2, y_max2, z_max2 = bbox2
        x_min = min(x_min1, x_min2)
        y_min = min(y_min1, y_min2)
        z_min = min(z_min1, z_min2)
        x_max = max(x_max1, x_max2)
        y_max = max(y_max1, y_max2)
        z_max = max(z_max1, z_max2)
        return [x_min, y_min, z_min, x_max, y_max, z_max]

    def feature_match(self, feature1, feature2, threshold=0.9):
        ft1 = np.array(feature1)
        ft2 = np.array(feature2)
        # 计算余弦相似度
        cos_sim = np.dot(ft1, ft2) / (np.linalg.norm(ft1) * np.linalg.norm(ft2))
        # 设定一个阈值，判断是否匹配
        if cos_sim > threshold:
            return True
        else:
            return False

    def bbox_match(self, bbox1, bbox2):
        """检查两个bbox是否匹配, 是否有交集"""
        x_min1, y_min1, z_min1, x_max1, y_max1, z_max1 = bbox1
        x_min2, y_min2, z_min2, x_max2, y_max2, z_max2 = bbox2
        if (
            x_min1 > x_max2
            or x_max1 < x_min2
            or y_min1 > y_max2
            or y_max1 < y_min2
            or z_min1 > z_max2
            or z_max1 < z_min2
        ):
            return False
        return True

    def bbox_calc(self, x_list, y_list, z_list):
        """计算AABB"""
        x_min = min(x_list)
        y_min = min(y_list)
        z_min = min(z_list)
        x_max = max(x_list)
        y_max = max(y_list)
        z_max = max(z_list)
        return [x_min, y_min, z_min, x_max, y_max, z_max]

    def get_base_pose(self):
        """
        监听tf，# 获取相机在世界坐标系下的2D位置
        返回(x, y)
        """
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_link_father,
                self.base_link_child,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            camera_position = transform.transform.translation
            camera_x, camera_y = camera_position.x, camera_position.y
            return camera_x, camera_y
        except TransformException as e:
            rospy.logerr(f"Transform error: {str(e)}")
            return

    def semantic_object_callback(self, msg):
        """增加数据有效性检查的写入方法"""
        try:
            # 验证数据长度是否有效, 以及一致性
            if msg.count == 0:
                rospy.logwarn(f"Received Empty data for {msg.category}")
                return
            if not (
                len(msg.x) == len(msg.y) == len(msg.z) == len(msg.rgb) == msg.count
            ):
                rospy.logerr(
                    f"Inconsistent data length, len(x): {len(msg.x)}, len(y): {len(msg.y)}, len(z): {len(msg.z)}, len(rgb): {len(msg.rgb)}, count: {msg.count}"
                )
                return
            # 获取msg
            try:
                category = msg.category
                count = msg.count
                confidence = msg.confidence
                x_list = list(msg.x)
                y_list = list(msg.y)
                z_list = list(msg.z)
                rgb_list = list(msg.rgb)
                cv_image = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="bgr8")
                crop_feature = list(msg.feature)
            except Exception as e:
                rospy.logerr(f"Message unpack error: {str(e)}")
                return
            # 计算bbox， AABB
            bbox = self.bbox_calc(x_list, y_list, z_list)
            object_x = (bbox[0] + bbox[3]) / 2  # bbox中心点x
            object_y = (bbox[1] + bbox[4]) / 2  # bbox中心点y
            # 获取相机在世界坐标系下的2D位置
            camera_x, camera_y = self.get_base_pose()
            if camera_x is None or camera_y is None:
                rospy.logerr("Camera position is None, skipping this object.")
                return
            # 计算方向
            direction_x = object_x - camera_x
            direction_y = object_y - camera_y
            act_list = [
                int(direction_x > 0 and abs(direction_x) > abs(direction_y)),  # +x
                int(direction_y > 0 and abs(direction_y) > abs(direction_x)),  # +y
                int(direction_x < 0 and abs(direction_x) > abs(direction_y)),  # -x
                int(direction_y < 0 and abs(direction_y) > abs(direction_x)),  # -y
            ]
            # 更新语义对象到数据库
            self.update_db(
                category,
                bbox,
                count,
                x_list,
                y_list,
                z_list,
                rgb_list,
                cv_image,
                act_list,
                crop_feature,
            )

        except Exception as e:
            rospy.logerr(f"Cloud callback error: {str(e)}")

    def handle_show(self, req):
        if not self.show_markers:
            rospy.logwarn("Show markers is disabled, please enable it in the config.")
            res = ShowResponse()
            res.success = False
            res.message = "Show markers is disabled"
            return res
        # 处理请求
        data = req.data
        # 对data进行区分,是要show全部，还是只要show特定label，还是show特定category
        show_type = None
        if data == "all":
            show_type = "all"
            self.pub_all()
        elif "@" in data:
            show_type = "label"
            self.pub_label(data)
        else:
            show_type = "category"
            self.pub_category(data)
        res = ShowResponse()
        res.success = True
        res.message = f"Show {show_type} success."
        return res

    def entry_to_points(self, entry: dict) -> list:
        # entry = {"label":str, "bbox": list(float), "time_stamp": str, "x": list(float), "y": list(float), "z": list(float), "rgb": list(int), "act": list(int)}
        # 提取x, y, z, rgb
        x_list = entry["x"]
        y_list = entry["y"]
        z_list = entry["z"]
        rgb_list = entry["rgb"]
        points = []
        for x, y, z, rgb in zip(x_list, y_list, z_list, rgb_list):
            points.append([x, y, z, rgb])
        return points

    def pub_semantic_map(self, points: list):
        try:
            # 转换为ROS消息
            header = Header(stamp=rospy.Time.now(), frame_id=self.semantic_map_frame_id)
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("rgb", 12, PointField.UINT32, 1),
            ]

            # 定义结构化数据类型
            point_dtype = [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ]

            # 转换为结构化数组
            structured_array = np.array(
                [(p[0], p[1], p[2], p[3]) for p in points], dtype=point_dtype
            )

            # 直接使用数组的tobytes()方法
            packed_data = structured_array.tobytes()

            cloud = PointCloud2(
                header=header,
                height=1,
                width=len(structured_array),
                is_dense=True,
                is_bigendian=False,
                fields=fields,
                point_step=16,
                row_step=16 * len(structured_array),
                data=packed_data,
            )

            self.pub.publish(cloud)

        except Exception as e:
            rospy.logerr(f"Publishing failed: {str(e)}")

    def pub_all(self):
        try:
            # 获取所有entry
            # all_entries = [{"label":str, "bbox": list(float), "time_stamp": str, "x": list(float), "y": list(float), "z": list(float), "rgb": list(int)}, ...]
            all_entries = self.database._get_all_entries()
            if not all_entries:
                rospy.logwarn("No entries to publish")
                return
            # 从entry中提取x, y, z, rgb, 并合并所有点云
            all_points = []
            for entry in all_entries:
                points = self.entry_to_points(entry)
                all_points.extend(points)
            if not all_points:
                rospy.logwarn("No points to publish")
                return
            # 发布点云
            self.pub_semantic_map(all_points)
        except Exception as e:
            rospy.logerr(f"Publish all error: {str(e)}")

        if self.show_markers:
            self.pub_markers(all_entries)

    def pub_label(self, label):
        try:
            # 获取指定label的entry
            # entry = {"label":str, "bbox": list(float), "time_stamp": str, "x": list(float), "y": list(float), "z": list(float), "rgb": list(int), "act": list(int)}
            entry = self.database._get_entry_by_label(label)
            if not entry:
                rospy.logwarn(f"No entry with label {label}")
                return
            # 从entry中提取x, y, z, rgb
            points = self.entry_to_points(entry)
            if not points:
                rospy.logwarn(f"No points to publish for label {label}")
                return
            # 发布点云
            self.pub_semantic_map(points)
        except Exception as e:
            rospy.logerr(f"Publish label error: {str(e)}")

        if self.show_markers:
            self.pub_markers(entry)

    def pub_category(self, category):
        try:
            # 获取指定category的所有entry
            # category_entries = [{"label":str, "bbox": list(float), "time_stamp": str, "x": list(float), "y": list(float), "z": list(float), "rgb": list(int)}, ...]
            category_entries = self.database._get_entries_by_category(category)
            if not category_entries:
                rospy.logwarn(f"No entries with category {category}")
                return
            # 从entry中提取x, y, z, rgb, 并合并所有点云
            all_points = []
            for entry in category_entries:
                points = self.entry_to_points(entry)
                all_points.extend(points)
            if not all_points:
                rospy.logwarn(f"No points to publish for category {category}")
                return
            # 发布点云
            self.pub_semantic_map(all_points)
        except Exception as e:
            rospy.logerr(f"Publish category error: {str(e)}")

        if self.show_markers:
            self.pub_markers(category_entries)

    def create_bbox_marker(self, bbox, label):
        """
        根据AABB创建线框Marker
        :param bbox: [x_min, y_min, z_min, x_max, y_max, z_max]
        :param label: 对象标签（用于命名空间）
        :return: Marker对象
        """
        marker = Marker()
        marker.header.frame_id = self.semantic_map_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = label
        marker.id = 0  # 同一个ns下保持唯一即可
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0)  # 永久显示

        # 设置尺寸和颜色
        marker.scale.x = self.bbox_line_width
        marker.color.r = self.bbox_color[0]
        marker.color.g = self.bbox_color[1]
        marker.color.b = self.bbox_color[2]
        marker.color.a = self.bbox_alpha

        # 计算立方体的8个顶点
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        points = [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ]

        # 定义连接线段的顶点索引对
        lines = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # 底面
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # 顶面
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # 侧面
        ]

        # 添加线段到Marker
        for i, j in lines:
            marker.points.append(Point(*points[i]))
            marker.points.append(Point(*points[j]))

        return marker

    def create_arrow_markers(self, bbox, act, label):
        """
        根据方向信息创建箭头Marker
        :param bbox: [x_min, y_min, z_min, x_max, y_max, z_max]
        :param act: 方向信息列表 [1, 0, 1, 0]，分别表示 [+x, +y, -x, -y]
        :param label: 对象标签（用于命名空间）
        :return: 包含箭头Marker的列表
        """
        arrow_markers = []

        # 计算包围盒中心点
        object_x = (bbox[0] + bbox[3]) / 2  # bbox中心点x
        object_y = (bbox[1] + bbox[4]) / 2  # bbox中心点y
        object_z = (bbox[2] + bbox[5]) / 2  # bbox中心点z (假设箭头在平面上)

        # 箭头长度和方向偏移
        arrow_length = self.arrow_length  # 箭头长度
        directions = ["+x", "+y", "-x", "-y"]
        offsets = [
            (arrow_length, 0, 0),  # +x
            (0, arrow_length, 0),  # +y
            (-arrow_length, 0, 0),  # -x
            (0, -arrow_length, 0),  # -y
        ]

        for i, active in enumerate(act):
            if active:  # 如果该方向为1，添加箭头
                offset_x, offset_y, offset_z = offsets[i]
                arrow_start = Point(
                    object_x - offset_x, object_y - offset_y, object_z - offset_z
                )
                arrow_end = Point(object_x, object_y, object_z)

                # 创建箭头Marker
                arrow_marker = Marker()
                arrow_marker.header.frame_id = self.semantic_map_frame_id
                arrow_marker.header.stamp = rospy.Time.now()
                arrow_marker.ns = f"{label}_arrow"
                arrow_marker.id = len(arrow_markers)  # 保证唯一ID
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                arrow_marker.lifetime = rospy.Duration(0)  # 永久显示

                # 设置箭头的起点和终点
                arrow_marker.points.append(arrow_start)
                arrow_marker.points.append(arrow_end)

                # 设置箭头的颜色和尺寸
                arrow_marker.scale.x = self.arrow_scale_x  # 箭头杆的直径
                arrow_marker.scale.y = self.arrow_scale_y  # 箭头头部的直径
                arrow_marker.scale.z = self.arrow_scale_z  # 箭头头部的长度
                arrow_marker.color.r = self.arrow_color[0]
                arrow_marker.color.g = self.arrow_color[1]
                arrow_marker.color.b = self.arrow_color[2]
                arrow_marker.color.a = self.arrow_alpha

                arrow_markers.append(arrow_marker)

        return arrow_markers

    def create_text_marker(self, bbox, label):
        """
        创建显示标签的文本Marker
        :param bbox: [x_min, y_min, z_min, x_max, y_max, z_max]
        :param label: 对象标签
        :return: 文本Marker对象
        """
        marker = Marker()
        marker.header.frame_id = self.semantic_map_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = f"{label}_text"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0)  # 永久显示

        # 将文本放在包围盒顶部
        marker.pose.position.x = (bbox[0] + bbox[3]) / 2  # x中心
        marker.pose.position.y = (bbox[1] + bbox[4]) / 2  # y中心
        marker.pose.position.z = bbox[5] + 0.1  # 略高于包围盒的最高点

        # 设置文本内容和属性
        marker.text = label
        marker.scale.z = self.label_scale  # 字体大小
        marker.color.r = self.label_color[0]
        marker.color.g = self.label_color[1]
        marker.color.b = self.label_color[2]
        marker.color.a = self.label_alpha  # 文本透明度

        return marker

    def pub_markers(self, entries):
        """
        删除旧的并发布新的包围盒Marker和箭头
        :param entries: 语义对象条目列表
        """
        # 删除旧的包围盒Marker
        self.del_all_bbox_markers()

        # 发布新的包围盒Marker和箭头
        markers = MarkerArray()
        if type(entries) is not list:
            entries = [entries]
        for entry in entries:
            label = entry["label"]
            bbox = entry["bbox"]
            act = entry["act"]  # 获取方向信息

            # 创建包围盒Marker
            if self.show_bbox:
                bbox_marker = self.create_bbox_marker(bbox, label)
                markers.markers.append(bbox_marker)

            # 创建箭头Marker
            if self.show_arrow:
                arrow_markers = self.create_arrow_markers(bbox, act, label)
                markers.markers.extend(arrow_markers)

            # 创建标签文本Marker
            if self.show_label:
                text_marker = self.create_text_marker(bbox, label)
                markers.markers.append(text_marker)

        self.bbox_pub.publish(markers)

    def del_all_bbox_markers(self):
        """
        删除所有包围盒Marker
        """
        # 创建删除指令
        delete_marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = self.semantic_map_frame_id
        delete_marker.action = Marker.DELETEALL
        delete_marker_array.markers.append(delete_marker)

        # 发布删除指令
        self.bbox_pub.publish(delete_marker_array)


if __name__ == "__main__":
    manager = SemanticMapManager()
    rospy.spin()
