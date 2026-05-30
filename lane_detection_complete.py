#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV车道线检测完整库文件
固定参数实现，封装三个对外调用函数
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip


# =============================================================================
# 内部工具函数
# =============================================================================

def rgb_to_hsl(image):
    """RGB转HSL颜色空间"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)


def create_white_mask(hsl_image):
    """HSL白色阈值掩码: [0,200,0] ~ [255,255,255]"""
    lower = np.array([0, 200, 0], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsl_image, lower, upper)


def create_yellow_mask(hsl_image):
    """HSL黄色阈值掩码: [15,40,100] ~ [35,255,255]"""
    lower = np.array([15, 40, 100], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsl_image, lower, upper)


def combine_masks(mask1, mask2):
    """合并两个掩码"""
    return cv2.bitwise_or(mask1, mask2)


def apply_mask(image, mask):
    """应用掩码过滤"""
    return cv2.bitwise_and(image, image, mask=mask)


def grayscale(image):
    """灰度化"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel_size=(5, 5)):
    """高斯模糊降噪"""
    return cv2.GaussianBlur(image, kernel_size, 0)


def canny_edge(image):
    """Canny边缘检测: 低阈值50, 高阈值150"""
    return cv2.Canny(image, 50, 150)


def create_roi_mask(image_shape):
    """创建ROI四边形掩码，只保留图像下半车道区域"""
    height, width = image_shape[:2]
    vertices = np.array([[
        [width * 0.1, height],
        [width * 0.45, height * 0.6],
        [width * 0.55, height * 0.6],
        [width * 0.9, height]
    ]], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    return mask


def apply_roi(edge_image, mask):
    """应用ROI掩码裁剪"""
    return cv2.bitwise_and(edge_image, mask)


def hough_lines(edge_image):
    """霍夫概率变换: rho=1, theta=π/180, threshold=30, minLineLength=50, maxLineGap=10"""
    return cv2.HoughLinesP(edge_image, 1, math.pi / 180, 30, minLineLength=50, maxLineGap=10)


def calculate_slope(line):
    """计算线段斜率"""
    x1, y1, x2, y2 = line
    if x2 == x1:
        return None
    return (y2 - y1) / (x2 - x1)


def separate_lanes(lines):
    """斜率分类区分左右车道: 斜率<0左车道, 斜率>0右车道"""
    left_lines = []
    right_lines = []
    if lines is None:
        return left_lines, right_lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = calculate_slope((x1, y1, x2, y2))
        if slope is None:
            continue
        if abs(slope) < 0.5:
            continue
        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))
    return left_lines, right_lines


def fit_line(points):
    """最小二乘拟合直线"""
    if len(points) < 2:
        return None
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    if np.std(x_coords) < 1:
        return None
    coeffs = np.polyfit(x_coords, y_coords, 1)
    return coeffs


def get_line_endpoints(line, y_min, y_max):
    """根据拟合参数计算线段端点"""
    if line is None:
        return None, None
    slope, intercept = line
    if slope == 0:
        return None, None
    x1 = int((y_min - intercept) / slope)
    y1 = int(y_min)
    x2 = int((y_max - intercept) / slope)
    y2 = int(y_max)
    return (x1, y1), (x2, y2)


def draw_lane_lines_overlay(image, left_line, right_line, y_min, y_max):
    """新建画布绘制车道线overlay"""
    overlay = image.copy()
    if left_line is not None:
        pt1, pt2 = get_line_endpoints(left_line, y_min, y_max)
        if pt1 is not None:
            cv2.line(overlay, pt1, pt2, (0, 255, 0), 5)
    if right_line is not None:
        pt1, pt2 = get_line_endpoints(right_line, y_min, y_max)
        if pt1 is not None:
            cv2.line(overlay, pt1, pt2, (0, 255, 0), 5)
    return overlay


def blend_with_original(result, overlay):
    """透明度加权融合: overlay权重0.7, 原图权重0.3"""
    return cv2.addWeighted(overlay, 0.7, result, 0.3, 0)


# =============================================================================
# 单帧处理核心函数
# =============================================================================

def process_frame(frame):
    """
    单帧图像完整处理流程:
    图像读取 → RGB转HSL → 白/黄双阈值掩码融合 → 灰度化 → 高斯模糊 →
    Canny边缘提取 → ROI四边形掩码裁剪 → HoughLinesP直线检测 →
    斜率分类区分左右车道 → 最小二乘拟合完整车道线 → overlay绘制 → addWeighted融合
    """
    height, width = frame.shape[:2]

    hsl = rgb_to_hsl(frame)

    white_mask = create_white_mask(hsl)
    yellow_mask = create_yellow_mask(hsl)
    color_mask = combine_masks(white_mask, yellow_mask)

    filtered = apply_mask(frame, color_mask)

    gray = grayscale(filtered)

    blurred = gaussian_blur(gray)

    edges = canny_edge(blurred)

    roi_mask = create_roi_mask((height, width))
    masked_edges = apply_roi(edges, roi_mask)

    lines = hough_lines(masked_edges)

    left_lines, right_lines = separate_lanes(lines)

    left_points = [(x1, y1) for x1, y1, x2, y2 in left_lines] + [(x2, y2) for x1, y1, x2, y2 in left_lines]
    right_points = [(x1, y1) for x1, y1, x2, y2 in right_lines] + [(x2, y2) for x1, y1, x2, y2 in right_lines]

    y_min = int(height * 0.6)
    y_max = int(height)

    left_line = fit_line(left_points)
    right_line = fit_line(right_points)

    overlay = draw_lane_lines_overlay(frame, left_line, right_line, y_min, y_max)

    result = blend_with_original(frame, overlay)

    return result


# =============================================================================
# 对外调用函数
# =============================================================================

def imread_chinese(path):
    """支持中文路径的图像读取"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def process_image_file(input_img_path, output_img_path):
    """
    读取单张图片处理，保存输出结果图

    参数:
        input_img_path: 输入图像路径
        output_img_path: 输出图像路径
    """
    frame = imread_chinese(input_img_path)
    if frame is None:
        print(f"错误：无法读取图像 {input_img_path}")
        return
    result = process_frame(frame)
    cv2.imwrite(output_img_path, result)
    print(f"图像已保存: {output_img_path}")


def process_video_file(input_video_path, output_video_path):
    """
    基于moviepy逐帧调用图像处理逻辑，输出处理完成视频

    参数:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
    """
    print(f"开始处理视频: {input_video_path}")
    clip = VideoFileClip(input_video_path)

    processed_frames = []
    for frame in clip.iter_frames(fps=clip.fps, dtype='uint8'):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = process_frame(frame_bgr)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        processed_frames.append(result_rgb)

    from moviepy import ImageSequenceClip
    output_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
    output_clip.write_videofile(output_video_path, codec='libx264', audio=False)
    print(f"视频已保存: {output_video_path}")


def display_processing_steps(input_img_path):
    """
    绘图展示全流程中间效果图: 原图、HSL掩码、边缘图、ROI图、最终检测图

    参数:
        input_img_path: 输入图像路径
    """
    frame = imread_chinese(input_img_path)
    if frame is None:
        print(f"错误：无法读取图像 {input_img_path}")
        return

    height, width = frame.shape[:2]

    hsl = rgb_to_hsl(frame)
    white_mask = create_white_mask(hsl)
    yellow_mask = create_yellow_mask(hsl)
    color_mask = combine_masks(white_mask, yellow_mask)
    filtered = apply_mask(frame, color_mask)
    gray = grayscale(filtered)
    blurred = gaussian_blur(gray)
    edges = canny_edge(blurred)
    roi_mask = create_roi_mask((height, width))
    masked_edges = apply_roi(edges, roi_mask)
    result = process_frame(frame)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(color_mask, cmap='gray')
    plt.title('HSL Color Mask')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(masked_edges, cmap='gray')
    plt.title('ROI Masked Edges')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    roi_display = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
    plt.imshow(cv2.cvtColor(roi_display, cv2.COLOR_BGR2RGB))
    plt.title('Hough Lines')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Lane Detection')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("OpenCV车道线检测库")
    print("使用示例:")
    print('  from lane_detection_complete import *')
    print('  process_image_file(r"D:\\计算机实践\\test_images\\test.jpg", "output.jpg")')
    print('  process_video_file(r"D:\\计算机实践\\test_videos\\test.mp4", "output.mp4")')
    print('  display_processing_steps(r"D:\\计算机实践\\test_images\\test.jpg")')
