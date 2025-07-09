import sqlite3
import threading
from datetime import datetime
import numpy as np
import cv2
import os


class SemanticMapDatabase:
    def __init__(
        self,
        db_path: str,
        last_seen_imgs_dir: str,
        last_seen_imgs_max: int = 3,
        renew_db: bool = False,
    ):
        """
        初始化语义地图数据库

        :param db_path: SQLite 数据库文件路径
        :param last_seen_imgs_dir: 最近观测图像的存储目录
        """
        self.db_path = db_path
        self.last_seen_imgs_dir = last_seen_imgs_dir
        self.last_seen_imgs_max = last_seen_imgs_max
        if not os.path.exists(self.last_seen_imgs_dir):
            os.makedirs(self.last_seen_imgs_dir)
        if renew_db:
            self._renew_db()
        self.lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程安全的数据库连接并配置二进制支持"""
        conn = sqlite3.connect(
            self.db_path, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES
        )
        # 注册 BLOB 类型适配器
        sqlite3.register_converter("BLOB", lambda x: x)
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _renew_db(self):
        """删除数据库文件和最近观测图像目录"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.last_seen_imgs_dir):
            os.system(f"rm -rf {self.last_seen_imgs_dir}")

    def _init_db(self):
        """
        初始化数据库表结构和图像库: semantic_objects
        label: category@id
        bbox: AABB, [x_min, y_min, z_min, x_max, y_max, z_max]
        time_stamp: 时间戳
        x_data, y_data, z_data: 二进制数据(float32)
        rgb_data: 二进制数据(uint32)
        act_data: 二进制数据(uint32)
        feature_data: 二进制数据(float32)
        """
        with self.lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS semantic_objects (
                        label TEXT PRIMARY KEY,
                        bbox BLOB,
                        time_stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        x_data BLOB NOT NULL,
                        y_data BLOB NOT NULL,
                        z_data BLOB NOT NULL,
                        rgb_data BLOB NOT NULL,
                        act_data BLOB NOT NULL,
                        feature_data BLOB NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()
        if not os.path.exists(self.last_seen_imgs_dir):
            os.makedirs(self.last_seen_imgs_dir)

    def _float32_to_blob(self, data: list) -> bytes:
        """将 float32 数据转换为二进制数据"""
        return np.array(data, dtype=np.float32).tobytes()

    def _uint32_to_blob(self, data: list) -> bytes:
        """将 uint32 数据转换为二进制数据"""
        return np.array(data, dtype=np.uint32).tobytes()

    def _blob_to_float32(self, blob: bytes) -> list:
        """将二进制反序列化为float32列表"""
        return np.frombuffer(blob, dtype=np.float32).tolist()

    def _blob_to_uint32(self, blob: bytes) -> list:
        """将二进制反序列化为uint32列表"""
        return np.frombuffer(blob, dtype=np.uint32).tolist()

    def _update_entry(
        self,
        label: str,
        bbox: list,
        x: list,
        y: list,
        z: list,
        rgb: list,
        act: list,
        feature: list,
        time_stamp: str = None,
    ):
        """
        更新/插入语义对象条目
        :param label: 格式为 category@id 的标识符 (如 chair@1)
        :param bbox: 边界框 [x_min, y_min, z_min, x_max, y_max, z_max]
        :param x: x坐标列表
        :param y: y坐标列表
        :param z: z坐标列表
        :param rgb: RGB值列表 (uint32 packed)
        :param act: 动作列表 (one-hot编码)
        :param feature: 特征列表 (float32)
        :param time_stamp: 可选时间戳 (格式: YYYY-MM-DD HH:MM:SS)
        """
        if len(act) != 4:
            raise ValueError(f"act_data length must be 4, but got {len(act)}")

        bbox_data = self._float32_to_blob(bbox)
        time_stamp = time_stamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        x_data = self._float32_to_blob(x)
        y_data = self._float32_to_blob(y)
        z_data = self._float32_to_blob(z)
        rgb_data = self._uint32_to_blob(rgb)
        act_data = self._uint32_to_blob(act)
        feature_data = self._float32_to_blob(feature)
        # 插入或更新条目
        with self.lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO semantic_objects
                    (label, bbox, time_stamp, x_data, y_data, z_data, rgb_data, act_data, feature_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        label,
                        bbox_data,
                        time_stamp,
                        x_data,
                        y_data,
                        z_data,
                        rgb_data,
                        act_data,
                        feature_data,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def _get_entry_by_label(self, label: str) -> dict:
        """
        通过标签获取语义对象条目
        :param label: 格式为 category@id 的标识符 (如 chair@1)
        :return: 语义对象条目字典
        """
        with self.lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """
                    SELECT * FROM semantic_objects WHERE label=?
                    """,
                    (label,),
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "label": row[0],
                        "bbox": self._blob_to_float32(row[1]),
                        "time_stamp": row[2],
                        "x": self._blob_to_float32(row[3]),
                        "y": self._blob_to_float32(row[4]),
                        "z": self._blob_to_float32(row[5]),
                        "rgb": self._blob_to_uint32(row[6]),
                        "act": self._blob_to_uint32(row[7]),
                        "feature": self._blob_to_float32(row[8]),
                    }
                return None
            finally:
                conn.close()

    def _get_entries_by_category(self, category: str) -> list:
        """
        通过类别, 获取同一类语义对象条目列表
        :param category: 类别标识符 (如 chair)
        :return: 语义对象条目列表
        """
        with self.lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """
                    SELECT * FROM semantic_objects WHERE label LIKE ?
                    """,
                    (f"{category}@%",),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "label": row[0],
                        "bbox": self._blob_to_float32(row[1]),
                        "time_stamp": row[2],
                        "x": self._blob_to_float32(row[3]),
                        "y": self._blob_to_float32(row[4]),
                        "z": self._blob_to_float32(row[5]),
                        "rgb": self._blob_to_uint32(row[6]),
                        "act": self._blob_to_uint32(row[7]),
                        "feature": self._blob_to_float32(row[8]),
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def _get_all_entries(self) -> list:
        """
        获取所有语义对象条目
        :return: 语义对象条目列表
        """
        with self.lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("SELECT * FROM semantic_objects")
                rows = cursor.fetchall()
                return [
                    {
                        "label": row[0],
                        "bbox": self._blob_to_float32(row[1]),
                        "time_stamp": row[2],
                        "x": self._blob_to_float32(row[3]),
                        "y": self._blob_to_float32(row[4]),
                        "z": self._blob_to_float32(row[5]),
                        "rgb": self._blob_to_uint32(row[6]),
                        "act": self._blob_to_uint32(row[7]),
                        "feature": self._blob_to_float32(row[8]),
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def get_img_paths_by_category(self, category: str) -> dict:
        """
        获取指定类别的所有观测图像路径
        :param category: 类别标识符 (如 chair)
        :return: 观测图像路径字典{label: [img_path1, img_path2, ...]}
        """
        label_dir_names = os.listdir(self.last_seen_imgs_dir)
        label_dir_names.sort()
        img_paths_dict = {}
        for label_dir_name in label_dir_names:
            if label_dir_name.startswith(category):
                # 此时label_dir_name（eg. bed@1)下面还有<=四个朝向目录(eg. xp, yp, xn, yn), 要全部读出来
                act_dir_names = os.listdir(
                    os.path.join(self.last_seen_imgs_dir, label_dir_name)
                )
                act_dir_names.sort()
                img_paths = []
                for act_dir_name in act_dir_names:
                    act_dir_path = os.path.join(
                        self.last_seen_imgs_dir, label_dir_name, act_dir_name
                    )
                    img_files = os.listdir(act_dir_path)
                    img_files.sort()
                    for img_file in img_files:
                        img_paths.append(os.path.join(act_dir_path, img_file))
                img_paths_dict[label_dir_name] = img_paths
        return img_paths_dict

    def get_img_paths_by_label(self, label: str) -> list:
        """
        获取指定标签的所有观测图像路径
        :param label: 格式为 category@id 的标识符 (如 chair@1)
        :return: 观测图像路径字典{"xp": [img_path1, img_path2, ...], "yp": [...], ...}
        """
        act_img_paths = {"xp": [], "yp": [], "xn": [], "yn": []}
        for act_dir_name in act_img_paths.keys():
            act_dir_path = os.path.join(self.last_seen_imgs_dir, label, act_dir_name)
            if os.path.exists(act_dir_path):
                img_files = os.listdir(act_dir_path)
                img_files.sort()
                for img_file in img_files:
                    act_img_paths[act_dir_name].append(
                        os.path.join(act_dir_path, img_file)
                    )
        return act_img_paths

    def _save_last_seen_img(self, label: str, cv_image: np.ndarray, act_list: list):
        """
        保存最近观测图像
        :param label: 格式为 category@id 的标识符 (如 chair@1)
        :param cv_image: OpenCV 图像对象
        :param act_list: one-hot编码，表示方向 [+x, +y, -x, -y]
        """
        act_dir_name = (
            "xp"
            if act_list[0]
            else "yp"
            if act_list[1]
            else "xn"
            if act_list[2]
            else "yn"
        )
        if os.path.exists(os.path.join(self.last_seen_imgs_dir, label, act_dir_name)):
            self._clear_last_seen_imgs(label, act_dir_name)
        else:
            os.makedirs(os.path.join(self.last_seen_imgs_dir, label, act_dir_name))
        time = datetime.now().strftime("%Y%m%d%H%M%S")
        img_path = os.path.join(
            self.last_seen_imgs_dir, label, act_dir_name, f"{label}@{time}.png"
        )
        cv2.imwrite(img_path, cv_image)

    def _clear_last_seen_imgs(self, label: str, act_dir_name: str):
        """
        清除指定label的最久远的观测图像，当超过self.last_seen_imgs_max时
        :param label: 格式为 category@id 的标识符 (如 chair@1)
        :param act_dir_name: 动作目录名称 (如 xp, yp, xn, yn)
        """
        img_dir = os.path.join(self.last_seen_imgs_dir, label, act_dir_name)
        img_files = os.listdir(img_dir)
        if len(img_files) >= self.last_seen_imgs_max:
            img_files.sort()
            os.remove(os.path.join(img_dir, img_files[0]))


if __name__ == "__main__":
    db = SemanticMapDatabase(
        "/home/yutian/YanBot/semantic_map.db", "/home/yutian/YanBot/last_seen_imgs"
    )
    # 测试插入和查询
    print(db.get_img_paths_by_label("bed@1"))
