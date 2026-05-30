# -*- coding: utf-8 -*-
"""
车道线检测系统 - 配置模块
集中管理所有固定参数
"""

import math
import numpy as np


class Config:
    """车道线检测固定参数配置"""

    # ==================== Canny边缘检测 ====================
    CANNY_LOW = 50
    CANNY_HIGH = 150

    # ==================== HSL颜色阈值 ====================
    # 白色车道线 [H, L, S]
    HSL_WHITE_LOW = np.array([0, 200, 0], dtype=np.uint8)
    HSL_WHITE_HIGH = np.array([255, 255, 255], dtype=np.uint8)

    # 黄色车道线
    HSL_YELLOW_LOW = np.array([15, 40, 100], dtype=np.uint8)
    HSL_YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

    # ==================== 霍夫概率变换 ====================
    HOUGH_RHO = 1
    HOUGH_THETA = math.pi / 180
    HOUGH_THRESHOLD = 30
    HOUGH_MIN_LINE_LENGTH = 50
    HOUGH_MAX_LINE_GAP = 10

    # ==================== 融合权重 ====================
    OVERLAY_WEIGHT = 0.7
    ORIGINAL_WEIGHT = 0.3

    # ==================== ROI区域顶点(相对比例) ====================
    ROI_VERTICES = np.array([
        [0.1, 1.0],   # 左下
        [0.45, 0.6],  # 左上
        [0.55, 0.6],  # 右上
        [0.9, 1.0]    # 右下
    ])

    # ==================== 车道线相关 ====================
    # 标准车道宽度(米)
    LANE_WIDTH_METERS = 3.7
    # 偏离预警阈值(车道宽度的百分比)
    DEPARTURE_THRESHOLD = 0.3
    # 最小斜率(过滤水平线)
    MIN_SLOPE = 0.5

    # ==================== 暗光增强 ====================
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)

    # ==================== 鸟瞰图 ====================
    # 透视变换源点(按实际情况调整)
    BIRDVIEW_SRC = np.float32([
        [0.4, 0.6],   # 左上
        [0.6, 0.6],   # 右上
        [0.9, 1.0],   # 右下
        [0.1, 1.0]    # 左下
    ])
    BIRDVIEW_DST = np.float32([
        [0.3, 0.0],   # 左上
        [0.7, 0.0],   # 右上
        [0.9, 1.0],   # 右下
        [0.1, 1.0]    # 左下
    ])

    # ==================== 曲线拟合 ====================
    CURVE_DEGREE = 2  # 多项式阶数
    CURVE_SMOOTH = 50  # 平滑因子

    # ==================== 跟踪器 ====================
    TRACKER_FRAMES = 5  # 平滑帧数
