# -*- coding: utf-8 -*-
"""
智能交通视觉监测系统 - 测试脚本
"""

import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_monitor import TrafficMonitor


def test_single_image(image_path, model_path=None):
    """测试单张图片检测"""
    print(f"测试图片: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        return
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("错误: 无法读取图片")
        return
    
    monitor = TrafficMonitor(speed_limit=60, fps=24, model_path=model_path)
    result, info = monitor.process_frame(frame)
    
    output_path = "traffic_test_result.jpg"
    cv2.imwrite(output_path, result)
    print(f"检测结果已保存到: {output_path}")
    print(f"检测统计: 车流量={info['count']}, 帧数={info['frame']}")


def test_video(video_path, model_path=None):
    """测试视频检测"""
    print(f"测试视频: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在 - {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误: 无法打开视频")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {total_frames}帧, {fps}FPS, {width}x{height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_output.avi', fourcc, fps, (width, height))
    
    monitor = TrafficMonitor(speed_limit=60, fps=fps, model_path=model_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, info = monitor.process_frame(frame)
        out.write(result)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"处理进度: {int(frame_count / total_frames * 100)}%")
    
    cap.release()
    out.release()
    
    stats = monitor.get_statistics()
    print(f"测试完成! 总帧数: {frame_count}, 车流量: {stats['total_count']}")


if __name__ == "__main__":
    # 检查YOLO模型
    model_paths = [
        r"D:\计算机实践\yolov8n.pt",
        r"D:\计算机实践\yolov8n-seg.pt"
    ]
    
    model_path = None
    for mp in model_paths:
        if os.path.exists(mp):
            model_path = mp
            break
    
    # 检查测试视频
    test_videos = [
        "../traffic_rideo.mp4",
        "../road_redio.mp4",
        "../challenge.mp4",
        "../solidWhiteRight.mp4"
    ]
    
    test_images = [
        "../solidWhiteRight.jpg",
        "../solidYellowCurve.jpg"
    ]
    
    # 查找可用的测试文件
    video_path = None
    for vp in test_videos:
        full_path = os.path.join(os.path.dirname(__file__), vp)
        if os.path.exists(full_path):
            video_path = full_path
            break
    
    image_path = None
    for ip in test_images:
        full_path = os.path.join(os.path.dirname(__file__), ip)
        if os.path.exists(full_path):
            image_path = full_path
            break
    
    print("=" * 50)
    print("智能交通视觉监测系统 - YOLOv8测试")
    print("=" * 50)
    
    if model_path:
        print(f"使用YOLO模型: {model_path}")
    else:
        print("未找到YOLO模型，使用传统背景减法")
    
    if image_path:
        print("\n1. 测试图片检测")
        test_single_image(image_path, model_path)
    
    if video_path:
        print("\n2. 测试视频检测")
        test_video(video_path, model_path)
    
    print("\n测试完成!")
