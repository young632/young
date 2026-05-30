# -*- coding: utf-8 -*-
"""
HUD实时数据显示模块
在画面上叠加显示检测信息
"""

import cv2
import numpy as np


class HUD:
    """抬头显示系统"""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_info(self, frame, fps=0, curvature=0, offset=0, warning=""):
        """
        在图像上绘制检测信息

        参数:
            frame: 输入图像
            fps: 帧率
            curvature: 曲率半径(米)
            offset: 偏移量(米)
            warning: 警告信息

        返回:
            叠加信息后的图像
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        # 背景面板
        overlay = result.copy()
        cv2.fillPoly(overlay, [np.array([[0, 0], [w, 0], [w, 120], [0, 120]])], (0, 0, 0))
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

        # 信息文本
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Curvature: {curvature:.1f}m" if curvature else "Curvature: N/A",
            f"Offset: {offset:.2f}m {'(Right)' if offset > 0 else '(Left)' if offset < 0 else ''}",
        ]

        y_base = 30
        for i, line in enumerate(info_lines):
            cv2.putText(result, line, (15, y_base + i * 28),
                        self.font, 0.7, (0, 255, 255), 2)

        # 警告信息
        if warning and warning != "Normal":
            cv2.putText(result, warning, (15, y_base + len(info_lines) * 28 + 10),
                        self.font, 0.8, (0, 0, 255), 2)

        return result

    def draw_both_lanes(self, frame, left_lane_info, right_lane_info):
        """
        绘制左右车道详细信息

        参数:
            frame: 输入图像
            left_lane_info: 左车道信息 dict
            right_lane_info: 右车道信息 dict

        返回:
            叠加信息后的图像
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        # 左侧面板 - 左车道信息
        overlay = result.copy()
        left_panel = np.array([[0, 0], [200, 0], [200, h], [0, h]])
        cv2.fillPoly(overlay, [left_panel], (30, 30, 30))
        cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)

        # 右侧面板 - 右车道信息
        overlay = result.copy()
        right_panel = np.array([[w-200, 0], [w, 0], [w, h], [w-200, h]])
        cv2.fillPoly(overlay, [right_panel], (30, 30, 30))
        cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)

        # 绘制左车道信息
        y = 30
        cv2.putText(result, "LEFT LANE", (10, y), self.font, 0.6, (0, 255, 0), 2)
        for key, value in left_lane_info.items():
            cv2.putText(result, f"{key}: {value}", (10, y + 25),
                       self.font, 0.5, (200, 200, 200), 1)
            y += 25

        # 绘制右车道信息
        y = 30
        cv2.putText(result, "RIGHT LANE", (w - 190, y), self.font, 0.6, (0, 255, 0), 2)
        for key, value in right_lane_info.items():
            cv2.putText(result, f"{key}: {value}", (w - 190, y + 25),
                       self.font, 0.5, (200, 200, 200), 1)
            y += 25

        return result

    def draw_lane_boundaries(self, frame, left_points, right_points):
        """
        绘制车道边界线

        参数:
            frame: 输入图像
            left_points: 左车道线点列表 [(x,y),...]
            right_points: 右车道线点列表

        返回:
            绘制后的图像
        """
        result = frame.copy()

        # 填充车道区域
        if left_points and right_points:
            # 确保左右点数相同
            min_len = min(len(left_points), len(right_points))
            pts = np.array(left_points[:min_len] + right_points[:min_len][::-1])
            cv2.fillPoly(result, [pts], (0, 100, 0, 0.3))

        # 绘制边界线
        if left_points:
            pts = np.array(left_points, dtype=np.int32)
            cv2.polylines(result, [pts], False, (0, 255, 0), 3)

        if right_points:
            pts = np.array(right_points, dtype=np.int32)
            cv2.polylines(result, [pts], False, (0, 255, 0), 3)

        return result
