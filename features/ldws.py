# -*- coding: utf-8 -*-
"""
LDWS车道偏离预警模块
"""

import numpy as np
from config import Config


class LDWS:
    """车道偏离预警系统"""

    def __init__(self, lane_width=3.7):
        self.lane_width = lane_width
        self.history = []
        self.max_history = 5

    def calculate_offset(self, left_x, right_x, image_width):
        """计算偏移量(米)"""
        if left_x is None or right_x is None:
            return 0
        lane_center = (left_x + right_x) / 2
        pixel_offset = lane_center - image_width / 2
        meters_per_pixel = self.lane_width / 100
        return pixel_offset * meters_per_pixel

    def check_departure(self, offset):
        """检测是否偏离"""
        threshold = self.lane_width * 0.3
        if abs(offset) > threshold:
            direction = "右偏" if offset > 0 else "左偏"
            return True, f"WARNING: 车道{direction}!"
        return False, "Normal"

    def get_smoothed_offset(self, offset):
        """平滑偏移量"""
        self.history.append(offset)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return np.median(self.history)
