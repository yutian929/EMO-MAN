import numpy as np
import cv2


class MapCoordinateTransformer:
    def __init__(self, resolution, width, height, origin, data):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin = origin

        # 原始地图数据（行优先存储，左下角原点）
        self.map_data = np.array(data, dtype=np.int8).reshape(self.height, self.width)
        # 创建自由空间二值地图, binary中, 0:障碍/未知, 1:自由
        self.map_data_binary = np.where(self.map_data == 0, 1, 0).astype(np.uint8)

        # 垂直翻转后的图像数据（左上角原点，适配OpenCV）
        self.image_data_binary = cv2.flip(self.map_data_binary, 0)

        # cv2.imwrite("map_data_binary.png", self.map_data_binary*255)
        # cv2.imwrite("image_data_binary.png", self.image_data_binary*255)

    def is_in_map(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    """世界坐标系 -> 地图像素坐标系（ROS原生坐标系）"""

    def world2map_x(self, world_x):
        map_x = int((world_x - self.origin.x) / self.resolution)
        return map_x if 0 <= map_x < self.width else None

    def world2map_y(self, world_y):
        map_y = int((world_y - self.origin.y) / self.resolution)
        return map_y if 0 <= map_y < self.height else None

    def world2map_p(self, world_x, world_y):
        map_x = self.world2map_x(world_x)
        map_y = self.world2map_y(world_y)
        return (map_x, map_y) if map_x is not None and map_y is not None else None

    def world2map_l(self, dis):
        return int(dis / self.resolution)

    """地图像素坐标系 -> 世界坐标系（像素中心点）"""

    def map2world_x(self, map_x):
        world_x = self.origin.x + (map_x + 0.5) * self.resolution
        return world_x

    def map2world_y(self, map_y):
        world_y = self.origin.y + (map_y + 0.5) * self.resolution
        return world_y

    def map2world_p(self, map_x, map_y):
        world_x = self.map2world_x(map_x)
        world_y = self.map2world_y(map_y)
        return (world_x, world_y) if self.is_in_map(map_x, map_y) else None

    """图像坐标系（翻转后） -> 地图像素坐标系"""

    def img2map_x(self, img_x):
        map_x = img_x
        return map_x if 0 <= map_x < self.width else None

    def img2map_y(self, img_y):
        map_y = self.height - img_y - 1
        return map_y if 0 <= map_y < self.height else None

    def img2map_p(self, img_x, img_y):
        map_x = self.img2map_x(img_x)
        map_y = self.img2map_y(img_y)
        return (map_x, map_y) if map_x is not None and map_y is not None else None

    """地图像素坐标系 -> 图像坐标系（翻转后）"""

    def map2img_x(self, map_x):
        img_x = map_x
        return img_x if 0 <= img_x < self.width else None

    def map2img_y(self, map_y):
        img_y = self.height - map_y - 1
        return img_y if 0 <= img_y < self.height else None

    def map2img_p(self, map_x, map_y):
        img_x = self.map2img_x(map_x)
        img_y = self.map2img_y(map_y)
        return (img_x, img_y) if img_x is not None and img_y is not None else None
