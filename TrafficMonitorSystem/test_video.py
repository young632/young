# -*- coding: utf-8 -*-
"""
视频播放测试脚本
"""
import cv2
import os
import sys

def test_video_playback(video_path):
    print(f"测试视频路径: {video_path}")
    print(f"文件是否存在: {os.path.exists(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("ERROR: 无法打开视频文件")
        return False
    
    print("OK: 视频文件打开成功")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"视频信息:")
    print(f"  FPS: {fps}")
    print(f"  总帧数: {total_frames}")
    print(f"  分辨率: {width} x {height}")
    
    frame_count = 0
    while frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ 读取第 {frame_count} 帧失败")
            break
        
        print(f"✅ 成功读取第 {frame_count} 帧")
        frame_count += 1
    
    cap.release()
    print("测试完成")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # 默认测试路径
        video_path = r"D:\计算机实践\test_videos\test.mp4"
    
    test_video_playback(video_path)
