#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV车道线检测实验 - 基于传统CV的车道线检测
实验环境：Linux、Python3、opencv-python、matplotlib、moviepy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


def imread(image_path):
    """
    跨平台图像读取函数，兼容Linux和Windows路径格式

    参数:
        image_path: 图像文件路径

    返回:
        img: 读取的图像，读取失败返回None
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return None
    return img


def display_images(images, titles, cmap=None):
    """
    使用matplotlib显示多张图像（适配Linux环境）

    参数:
        images: 图像列表
        titles: 标题列表
        cmap: 颜色映射模式（如'gray'用于灰度图）
    """
    num = len(images)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, num, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def convert_hsl(image):
    """
    将BGR图像转换为HSL颜色空间
    HSL色相(Hue)、饱和度(Saturation)、亮度(Lightness)

    参数:
        image: BGR格式图像

    返回:
        hsl: HSL格式图像
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)


def color_filter_hsl(image, hsl_lower, hsl_upper):
    """
    使用HSL颜色空间进行阈值过滤，提取特定颜色区域
    适用于车道线的白色和黄色检测

    参数:
        image: HSL格式图像
        hsl_lower: HSL下限阈值 [H, L, S]
        hsl_upper: HSL上限阈值 [H, L, S]

    返回:
        mask: 二值化掩膜
    """
    lower = np.array(hsl_lower, dtype=np.uint8)
    upper = np.array(hsl_upper, dtype=np.uint8)
    mask = cv2.inRange(image, lower, upper)
    return mask


def color_filter_rgb(image, rgb_lower, rgb_upper):
    """
    使用RGB颜色空间进行阈值过滤

    参数:
        image: BGR格式图像（OpenCV默认格式）
        rgb_lower: RGB下限阈值 [B, G, R]
        rgb_upper: RGB上限阈值 [B, G, R]

    返回:
        mask: 二值化掩膜
    """
    lower = np.array(rgb_lower, dtype=np.uint8)
    upper = np.array(rgb_upper, dtype=np.uint8)
    mask = cv2.inRange(image, lower, upper)
    return mask


def combine_masks(mask1, mask2):
    """
    合并两个掩膜（按位或运算）

    参数:
        mask1: 第一个掩膜
        mask2: 第二个掩膜

    返回:
        combined: 合并后的掩膜
    """
    return cv2.bitwise_or(mask1, mask2)


def grayscale(image):
    """
    将BGR图像转换为灰度图
    灰度化可以减少计算量，保留边缘等重要特征

    参数:
        image: BGR格式图像

    返回:
        gray: 灰度图像
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel_size=(5, 5), sigma=1.0):
    """
    高斯滤波去噪
    高斯模糊可以平滑图像，减少噪声干扰

    参数:
        image: 输入图像
        kernel_size: 卷积核大小，必须是奇数
        sigma: 高斯标准差，0表示根据核大小自动计算

    返回:
        blurred: 高斯模糊后的图像
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Canny边缘检测
    原理：高斯平滑 → 梯度计算 → 非极大值抑制 → 双阈值滞后处理

    参数:
        image: 灰度图像
        low_threshold: 低阈值
        high_threshold: 高阈值

    返回:
        edges: 边缘图像（二值图）
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def create_roi_mask(image_shape, vertices):
    """
    创建ROI（感兴趣区域）掩膜
    使用fillPoly填充多边形区域，只保留车道线所在区域

    参数:
        image_shape: 图像尺寸 (height, width)
        vertices: 多边形顶点坐标列表，如 [[x1,y1], [x2,y2], ...]

    返回:
        mask: 掩膜图像
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    roi_vertices = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    return mask


def apply_roi_mask(image, mask):
    """
    将ROI掩膜应用到图像上

    参数:
        image: 输入图像
        mask: ROI掩膜

    返回:
        masked: 应用掩膜后的图像
    """
    return cv2.bitwise_and(image, mask)


def hough_lines(image, rho=1, theta=np.pi/180, threshold=30, min_line_len=50, max_line_gap=10):
    """
    概率霍夫直线变换检测直线
    HoughLinesP直接检测直线的端点，比HoughLines更高效

    参数:
        image: 边缘检测后的图像
        rho: 距离分辨率（像素）
        theta: 角度分辨率（弧度）
        threshold: 累加器阈值，直线至少经过的点数
        min_line_len: 最小线段长度
        max_line_gap: 最大线段间隙

    返回:
        lines: 检测到的线段集合，每条线段为 [x1, y1, x2, y2]
    """
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)


def filter_lines_by_slope(lines, slope_threshold=0.5):
    """
    根据斜率过滤直线，分离左右车道线
    左车道线斜率为负，右车道线斜率为正

    参数:
        lines: 检测到的线段集合
        slope_threshold: 斜率阈值，绝对值小于此值的线段被过滤

    返回:
        left_lines: 左车道线列表
        right_lines: 右车道线列表
    """
    left_lines = []
    right_lines = []

    for line in lines:
        if line is None:
            continue
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) < slope_threshold:
            continue

        if slope < 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])

    return left_lines, right_lines


def remove_outliers(lines, threshold=30):
    """
    使用简单线性回归的思想剔除离群点
    计算所有线段斜率和截距的中值，将偏离较远的点剔除

    参数:
        lines: 线段列表
        threshold: 残差阈值

    返回:
        filtered: 过滤后的线段列表
    """
    if len(lines) == 0:
        return []

    slopes = []
    intercepts = []

    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        slopes.append(slope)
        intercepts.append(intercept)

    if len(slopes) == 0:
        return []

    slope_median = np.median(slopes)
    intercept_median = np.median(intercepts)

    filtered = []
    for i, (x1, y1, x2, y2) in enumerate(lines):
        slope = slopes[i]
        intercept = intercepts[i]

        if abs(slope - slope_median) < threshold:
            filtered.append((x1, y1, x2, y2))

    return filtered


def fit_line(points):
    """
    对点集进行直线拟合
    使用最小二乘法拟合出一条代表车道线的直线

    参数:
        points: 点坐标列表 [(x1,y1), (x2,y2), ...]

    返回:
        line: [slope, intercept] 或 None
    """
    if len(points) < 2:
        return None

    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])

    if np.sum(x_coords == x_coords[0]) == len(x_coords):
        return None

    coeffs = np.polyfit(x_coords, y_coords, 1)
    return coeffs


def get_line_endpoints(line, y_min, y_max):
    """
    根据拟合的直线参数计算线段端点

    参数:
        line: [slope, intercept]
        y_min: 最小y坐标
        y_max: 最大y坐标

    返回:
        (x1, y1), (x2, y2): 线段端点坐标
    """
    if line is None:
        return None, None

    slope, intercept = line
    x1 = int((y_min - intercept) / slope)
    y1 = int(y_min)
    x2 = int((y_max - intercept) / slope)
    y2 = int(y_max)

    return (x1, y1), (x2, y2)


def draw_line(image, pt1, pt2, color=(0, 255, 0), thickness=5):
    """
    在图像上绘制线段

    参数:
        image: 输入图像
        pt1: 起点坐标 (x, y)
        pt2: 终点坐标 (x, y)
        color: 线条颜色（BGR格式）
        thickness: 线条粗细

    返回:
        image: 绘制了线段的图像
    """
    if pt1 is None or pt2 is None:
        return image
    cv2.line(image, pt1, pt2, color, thickness)
    return image


def process_frame(frame):
    """
    处理单帧图像，检测车道线

    参数:
        frame: 输入帧（BGR格式）

    返回:
        result: 标注了车道线的图像
    """
    height, width = frame.shape[:2]

    hsl = convert_hsl(frame)

    white_lower = [0, 200, 0]
    white_upper = [255, 255, 255]
    white_mask = color_filter_hsl(hsl, white_lower, white_upper)

    yellow_lower = [15, 40, 100]
    yellow_upper = [35, 255, 255]
    yellow_mask = color_filter_hsl(hsl, yellow_lower, yellow_upper)

    color_mask = combine_masks(white_mask, yellow_mask)

    filtered = cv2.bitwise_and(frame, frame, mask=color_mask)

    gray = grayscale(filtered)

    blurred = gaussian_blur(gray, kernel_size=(5, 5), sigma=1.0)

    edges = canny_edge_detection(blurred, low_threshold=50, high_threshold=150)

    roi_vertices = [
        [width * 0.1, height * 0.9],
        [width * 0.45, height * 0.6],
        [width * 0.55, height * 0.6],
        [width * 0.9, height * 0.9]
    ]
    roi_mask = create_roi_mask((height, width), roi_vertices)

    masked_edges = apply_roi_mask(edges, roi_mask)

    lines = hough_lines(masked_edges, rho=1, theta=np.pi/180, threshold=30,
                        min_line_len=50, max_line_gap=10)

    result = frame.copy()

    if lines is not None:
        left_lines, right_lines = filter_lines_by_slope(lines, slope_threshold=0.5)

        left_lines = remove_outliers(left_lines, threshold=30)
        right_lines = remove_outliers(right_lines, threshold=30)

        left_points = [(line[0], line[1]) for line in left_lines] + \
                     [(line[2], line[3]) for line in left_lines]
        right_points = [(line[0], line[1]) for line in right_lines] + \
                       [(line[2], line[3]) for line in right_lines]

        y_min = int(height * 0.6)
        y_max = int(height * 0.9)

        left_line = fit_line(left_points)
        right_line = fit_line(right_points)

        left_pt1, left_pt2 = get_line_endpoints(left_line, y_min, y_max)
        right_pt1, right_pt2 = get_line_endpoints(right_line, y_min, y_max)

        result = draw_line(result, left_pt1, left_pt2, color=(0, 255, 0), thickness=5)
        result = draw_line(result, right_pt1, right_pt2, color=(0, 255, 0), thickness=5)

    return result


def process_image(image_path, output_path=None):
    """
    处理单张图像，检测并显示车道线

    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
    """
    frame = imread(image_path)
    if frame is None:
        return

    result = process_frame(frame)

    display_images([frame, result], ['Original', 'Lane Detection'])

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"结果已保存至: {output_path}")


def process_video(input_path, output_path=None):
    """
    处理视频文件，逐帧检测车道线

    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径（可选）
    """
    clip = VideoFileClip(input_path)

    processed_frames = []

    for frame in clip.iter_frames(fps=clip.fps, dtype='uint8'):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = process_frame(frame_bgr)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        processed_frames.append(result_rgb)

    from moviepy.editor import ImageSequenceClip

    output_clip = ImageSequenceClip(processed_frames, fps=clip.fps)

    if output_path:
        output_clip.write_videofile(output_path, codec='libx264', audio=False)
        print(f"视频已保存至: {output_path}")

    return output_clip


def main():
    """
    主函数 - 演示车道线检测流程
    """
    print("=" * 50)
    print("OpenCV车道线检测实验")
    print("实验环境：Linux、Python3、opencv-python、matplotlib、moviepy")
    print("=" * 50)

    print("\n可用函数：")
    print("  process_image(image_path, output_path) - 处理单张图像")
    print("  process_video(input_path, output_path) - 处理视频")
    print("  process_frame(frame) - 处理单帧")

    print("\n图像处理流程：")
    print("  图像读取 → 颜色空间过滤(HSL) → 灰度化 → 高斯模糊 → Canny边缘检测 → ROI掩膜 → 霍夫直线检测 → 直线拟合 → 绘制")

    print("\n颜色空间说明：")
    print("  HSL: 色相(H)、饱和度(S)、亮度(L)")
    print("  白色阈值: H[0,255], L[200,255], S[0,255]")
    print("  黄色阈值: H[15,35], L[40,255], S[100,255]")


if __name__ == "__main__":
    main()
