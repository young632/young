# -*- coding: utf-8 -*-
"""测试视频流是否正常工作"""

import cv2
import time
import os

def test_stream():
    print("=== 测试视频流 ===")
    
    # 测试视频文件
    video_path = r"D:\计算机实践\FlaskTrafficMonitor\uploads\traffic_rideo.mp4"
    if not os.path.exists(video_path):
        print(f"✗ 视频文件不存在: {video_path}")
        return
    
    print(f"✓ 视频文件存在: {video_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ 无法打开视频文件")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = 1.0 / fps
    print(f"✓ 视频帧率: {fps}")
    
    # 读取并处理几帧
    for i in range(5):
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("✗ 无法读取帧")
            break
        
        # 模拟处理
        processed_frame = frame.copy()
        
        # 编码
        ret_jpg, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret_jpg:
            print("✗ 编码失败")
            break
        
        frame_bytes = jpeg.tobytes()
        print(f"✓ 帧 {i+1}: 编码成功，大小={len(frame_bytes)} bytes")
        
        # 模拟帧率控制
        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
    
    cap.release()
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_stream()
