# -*- coding: utf-8 -*-
"""
车道偏离预警系统(LDWS)
实时计算车辆相对车道中心的偏移，超阈值报警
"""

import numpy as np
from config import Config


class LDWS:
    """车道偏离预警系统"""

    def __init__(self, lane_width=Config.LANE_WIDTH_METERS):
        """
        初始化LDWS

        参数:
            lane_width: 车道宽度(米), 默认3.7米
        """
        self.lane_width = lane_width
        self.departure_threshold = Config.DEPARTURE_THRESHOLD
        self.history = []
        self.max_history = 5

    def calculate_offset(self, left_x, right_x, image_width):
        """
        计算车辆相对车道中心的偏移

        参数:
            left_x: 左车道线位置(像素)
            right_x: 右车道线位置(像素)
            image_width: 图像宽度(像素)

        返回:
            offset: 偏移量(米), 正值=右偏, 负值=左偏
        """
        if left_x is None or right_x is None:
            return 0

        # 车道中心位置
        lane_center = (left_x + right_x) / 2
        # 图像中心(车辆位置)
        image_center = image_width / 2

        # 像素偏移
        pixel_offset = lane_center - image_center

        # 转换为米: 假设远处3米对应100像素(可根据实际情况调整)
        meters_per_pixel = self.lane_width / 100
        offset_meters = pixel_offset * meters_per_pixel

        return offset_meters

    def check_departure(self, offset_meters):
        """
        检查是否偏离车道

        参数:
            offset_meters: 偏移量(米)

        返回:
            (is_departing, message): 是否偏离, 提示信息
        """
        threshold = self.lane_width * self.departure_threshold

        if abs(offset_meters) > threshold:
            direction = "右偏" if offset_meters > 0 else "左偏"
            distance = abs(offset_meters) - threshold
            message = f"WARNING: 车道{direction}! 偏离 {distance*100:.1f}cm"
            return True, message

        return False, "Normal"

    def update(self, offset_meters):
        """更新历史记录并返回平滑后的偏移"""
        self.history.append(offset_meters)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return np.median(self.history)

    def get_warning_level(self, offset_meters):
        """
        获取警告等级

        参数:
            offset_meters: 偏移量(米)

        返回:
            level: 0=正常, 1=轻微, 2=严重
        """
        threshold = self.lane_width * self.departure_threshold

        if abs(offset_meters) < threshold * 0.5:
            return 0, "Normal"
        elif abs(offset_meters) < threshold:
            return 1, "Caution"
        else:
            return 2, "WARNING"
