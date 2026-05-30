# -*- coding: utf-8 -*-
"""
暗光图像增强模块
使用CLAHE算法提升低光照条件下的检测效果
"""

import cv2
import numpy as np
from config import Config


def enhance_low_light(image):
    """
    使用CLAHE增强低光照图像

    参数:
        image: BGR格式图像

    返回:
        增强后的BGR图像
    """
    # 转换到LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 创建CLAHE增强器
    clahe = cv2.createCLAHE(
        clipLimit=Config.CLAHE_CLIP_LIMIT,
        tileGridSize=Config.CLAHE_TILE_SIZE
    )

    # 增强L通道
    l_enhanced = clahe.apply(l)

    # 合并通道
    enhanced = cv2.merge([l_enhanced, a, b])

    # 转换回BGR
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return result


def auto_white_balance(image):
    """
    自动白平衡处理

    参数:
        image: BGR格式图像

    返回:
        白平衡后的图像
    """
    result = cv2.xphoto.createAutoWBhog().balanceWhite(image)
    return result


def enhance_contrast(image, clip_limit=2.0, tile_size=(8, 8)):
    """
    对比度增强

    参数:
        image: BGR格式图像
        clip_limit: CLAHE对比度限制
        tile_size: 网格大小

    返回:
        增强后的图像
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)

    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def detect_low_light(image):
    """
    检测图像是否处于低光照条件

    参数:
        image: BGR格式图像

    返回:
        True if low light, False otherwise
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    # 平均亮度小于50认为是低光照
    return mean_brightness < 50
