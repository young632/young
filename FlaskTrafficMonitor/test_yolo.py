# -*- coding: utf-8 -*-
"""测试YOLO模型和视频处理是否正常"""

import cv2
import os

def test_yolo():
    print("=== 测试YOLO模型 ===")
    
    try:
        from ultralytics import YOLO
        print("✓ ultralytics 库已安装")
    except ImportError as e:
        print(f"✗ ultralytics 库未安装: {e}")
        return
    
    # 查找YOLO模型
    model_paths = [
        r"D:\计算机实践\FlaskTrafficMonitor\yolov8n.pt",
        r"D:\计算机实践\yolov8n.pt",
        r"D:\计算机实践\yolov8n-seg.pt"
    ]
    
    model_path = None
    for mp in model_paths:
        if os.path.exists(mp):
            model_path = mp
            print(f"✓ 找到YOLO模型: {model_path}")
            break
    
    if model_path is None:
        print("✗ 未找到YOLO模型")
        return
    
    try:
        model = YOLO(model_path)
        print("✓ YOLO模型加载成功")
    except Exception as e:
        print(f"✗ YOLO模型加载失败: {e}")
        return
    
    # 测试视频文件
    video_path = r"D:\计算机实践\FlaskTrafficMonitor\uploads\traffic_rideo.mp4"
    if os.path.exists(video_path):
        print(f"✓ 视频文件存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print("✓ 视频文件可打开")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  - 帧率: {fps}")
            print(f"  - 帧数: {frame_count}")
            print(f"  - 分辨率: {width}x{height}")
            
            ret, frame = cap.read()
            if ret:
                print("✓ 成功读取视频帧")
                
                # 测试YOLO检测
                print("\n=== 测试YOLO检测 ===")
                try:
                    results = model(frame, verbose=False, conf=0.35, classes=[2,5,7])
                    print(f"✓ YOLO检测成功，检测到 {len(results[0].boxes)} 个目标")
                    
                    for box in results[0].boxes:
                        cls = int(box.cls.item())
                        conf = float(box.conf.item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        class_names = {2: 'car', 5: 'bus', 7: 'truck'}
                        print(f"  - {class_names.get(cls, cls)}: 置信度={conf:.2f}, 位置=({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")
                except Exception as e:
                    print(f"✗ YOLO检测失败: {e}")
            else:
                print("✗ 无法读取视频帧")
            
            cap.release()
        else:
            print("✗ 视频文件无法打开")
    else:
        print(f"✗ 视频文件不存在: {video_path}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_yolo()
