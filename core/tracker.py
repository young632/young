# -*- coding: utf-8 -*-
"""
车道线跟踪模块
帧间平滑，减少检测抖动
"""

import numpy as np
from config import Config


class LaneTracker:
    """车道线跟踪器"""

    def __init__(self, n_frames=Config.TRACKER_FRAMES):
        """
        初始化跟踪器

        参数:
            n_frames: 平滑帧数
        """
        self.n_frames = n_frames
        self.left_history = []
        self.right_history = []
        self.confidence_history = []

    def update(self, left_line, right_line, confidence=1.0):
        """
        更新跟踪器

        参数:
            left_line: 左车道线参数 [slope, intercept] 或 None
            right_line: 右车道线参数 [slope, intercept] 或 None
            confidence: 检测置信度 0-1

        返回:
            (smoothed_left, smoothed_right): 平滑后的车道线
        """
        self.left_history.append(left_line)
        self.right_history.append(right_line)
        self.confidence_history.append(confidence)

        # 限制历史长度
        if len(self.left_history) > self.n_frames:
            self.left_history.pop(0)
            self.right_history.pop(0)
            self.confidence_history.pop(0)

        # 计算加权平均
        smoothed_left = self._weighted_average(self.left_history, self.confidence_history)
        smoothed_right = self._weighted_average(self.right_history, self.confidence_history)

        return smoothed_left, smoothed_right

    def _weighted_average(self, lines, weights):
        """
        计算加权平均

        参数:
            lines: 线参数列表
            weights: 权重列表

        返回:
            平均后的线参数
        """
        valid_lines = [l for l in lines if l is not None]
        valid_weights = [w for l, w in zip(lines, weights) if l is not None]

        if not valid_lines:
            return None

        total_weight = sum(valid_weights)
        if total_weight == 0:
            total_weight = 1

        avg_line = np.zeros_like(valid_lines[0], dtype=float)
        for line, weight in zip(valid_lines, valid_weights):
            avg_line += line * (weight / total_weight)

        return avg_line

    def reset(self):
        """重置跟踪器"""
        self.left_history = []
        self.right_history = []
        self.confidence_history = []

    def get_smoothed_lines(self):
        """获取当前平滑后的车道线"""
        if not self.left_history or not self.right_history:
            return None, None

        return self._weighted_average(self.left_history, self.confidence_history), \
               self._weighted_average(self.right_history, self.confidence_history)


class MovingAverageFilter:
    """移动平均滤波器"""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def update(self, value):
        """更新滤波器"""
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if not self.history:
            return 0

        return sum(self.history) / len(self.history)

    def reset(self):
        """重置滤波器"""
        self.history = []
