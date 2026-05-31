# -*- coding: utf-8 -*-
"""测试车辆检测和中文显示"""

import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic_monitor import TrafficMonitor

def test_traffic_monitor():
    # 初始化监测系统
    model_path = r"D:\计算机实践\yolov8n.pt"
    if os.path.exists(model_path):
        monitor = TrafficMonitor(speed_limit=60, fps=24, model_path=model_path)
        print(f"使用YOLO模型: {model_path}")
    else:
        monitor = TrafficMonitor(speed_limit=60, fps=24)
        print("使用传统背景减法")
    
    # 加载测试图像
    test_images = [
        r"D:\计算机实践\solidWhiteCurve.jpg",
        r"D:\计算机实践\solidWhiteRight.jpg",
        r"D:\计算机实践\whiteCarLaneSwitch.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n处理图像: {img_path}")
            frame = cv2.imread(img_path)
            if frame is not None:
                result, info = monitor.process_frame(frame)
                
                # 保存结果
                output_path = f"test_result_{os.path.basename(img_path)}"
                cv2.imwrite(output_path, result)
                print(f"检测结果已保存: {output_path}")
                
                # 显示统计信息
                stats = monitor.get_statistics()
                print(f"车流量: {stats['total_count']}")
                print(f"平均速度: {stats['avg_speed']:.1f} km/h")
                print(f"拥堵指数: {stats['congestion_index']:.1f}%")
                print(f"各车型数量: {stats['vehicle_counts']}")
                
                # 显示结果
                cv2.imshow("检测结果", result)
                cv2.waitKey(2000)
            else:
                print(f"无法读取图像: {img_path}")
        else:
            print(f"图像不存在: {img_path}")
    
    cv2.destroyAllWindows()
    
    # 测试视频处理
    test_videos = [
        r"D:\计算机实践\road_redio.mp4",
        r"D:\计算机实践\traffic_rideo.mp4"
    ]
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            print(f"\n处理视频: {video_path}")
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                frame_count = 0
                save_frame_interval = 10
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    result, info = monitor.process_frame(frame)
                    frame_count += 1
                    
                    # 每10帧保存一次结果
                    if frame_count % save_frame_interval == 0:
                        output_path = f"video_frame_{frame_count}.jpg"
                        cv2.imwrite(output_path, result)
                        print(f"保存帧 {frame_count}")
                    
                    # 显示结果
                    cv2.imshow("视频检测", result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                print(f"视频处理完成，共处理 {frame_count} 帧")
            else:
                print(f"无法打开视频: {video_path}")
        else:
            print(f"视频不存在: {video_path}")

if __name__ == "__main__":
    test_traffic_monitor()
