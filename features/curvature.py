# -*- coding: utf-8 -*-
"""
曲率计算模块
基于车道线多项式拟合计算曲率半径
"""

import numpy as np


class CurvatureCalculator:
    """曲率计算器"""

    def __init__(self):
        self.ym_per_pix = 30 / 720  # 纵向: 米/像素
        self.xm_per_pix = 3.7 / 900  # 横向: 米/像素(假设车道宽3.7米)

    def calculate_curvature(self, fit_coeffs, y_eval):
        """
        计算曲率半径

        参数:
            fit_coeffs: 多项式拟合系数 [a, b, c] 表示 y = ax^2 + bx + c
            y_eval: 评估点的y坐标(像素)

        返回:
            curvature: 曲率半径(米)
        """
        if fit_coeffs is None or len(fit_coeffs) < 3:
            return None

        # 转换到世界坐标
        y_eval_m = y_eval * self.ym_per_pix

        # 多项式系数(世界坐标)
        coeffs_m = [
            fit_coeffs[0] * self.xm_per_pix / (self.ym_per_pix ** 2),
            fit_coeffs[1] * self.xm_per_pix / self.ym_per_pix,
            fit_coeffs[2] * self.xm_per_pix
        ]

        # 计算曲率半径公式: R = [(1+(dx/dy)^2)^1.5] / |d2x/dy2|
        # dx/dy = 2*a*y + b
        # d2x/dy2 = 2*a
        a, b, c = coeffs_m
        dy = 2 * a * y_eval_m + b
        d2y = 2 * a

        curvature = ((1 + dy ** 2) ** 1.5) / abs(d2y) if abs(d2y) > 1e-6 else float('inf')

        return curvature

    def calculate_offset_to_center(self, left_fit, right_fit, y_eval, image_width):
        """
        计算车辆相对车道中心的偏移

        参数:
            left_fit: 左车道线拟合系数
            right_fit: 右车道线拟合系数
            y_eval: 评估点y坐标
            image_width: 图像宽度

        返回:
            offset: 偏移量(米), 正值=偏右, 负值=偏左
        """
        if left_fit is None or right_fit is None:
            return 0

        # 计算左右车道在评估点的x坐标
        left_x = np.polyval(left_fit, y_eval)
        right_x = np.polyval(right_fit, y_eval)

        # 车道中心
        lane_center = (left_x + right_x) / 2

        # 车辆(图像)中心
        vehicle_center = image_width / 2

        # 像素偏移
        pixel_offset = lane_center - vehicle_center

        # 转换到米
        return pixel_offset * self.xm_per_pix

    def fit_polynomial(self, x_coords, y_coords, degree=2):
        """
        多项式拟合

        参数:
            x_coords: x坐标列表
            y_coords: y坐标列表
            degree: 多项式阶数

        返回:
            coeffs: 拟合系数
        """
        if len(x_coords) < degree + 1:
            return None

        try:
            coeffs = np.polyfit(y_coords, x_coords, degree)
            return coeffs
        except:
            return None
