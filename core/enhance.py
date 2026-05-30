# -*- coding: utf-8 -*-
"""
暗光增强模块
"""

import cv2
import numpy as np
from config import Config


def enhance_low_light(image):
    """使用CLAHE增强低光照图像"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=Config.CLAHE_TILE_SIZE)
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def detect_low_light(image):
    """检测是否为低光照条件"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < 50
