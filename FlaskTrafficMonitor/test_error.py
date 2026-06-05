# -*- coding: utf-8 -*-
"""详细测试错误原因"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("=== 测试导入 ===")
    
    try:
        from app.models.traffic_monitor import TrafficMonitor
        print("✓ TrafficMonitor导入成功")
    except Exception as e:
        print(f"✗ TrafficMonitor导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_model():
    print("\n=== 测试YOLO模型 ===")
    
    model_paths = [
        os.path.join(os.path.dirname(__file__), 'yolov8n.pt'),
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
        return None
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✓ YOLO模型加载成功")
        return model
    except Exception as e:
        print(f"✗ YOLO模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_monitor():
    print("\n=== 测试监测器 ===")
    
    model_paths = [
        os.path.join(os.path.dirname(__file__), 'yolov8n.pt'),
        r"D:\计算机实践\yolov8n.pt",
        r"D:\计算机实践\yolov8n-seg.pt"
    ]
    
    model_path = None
    for mp in model_paths:
        if os.path.exists(mp):
            model_path = mp
            break
    
    try:
        from app.models.traffic_monitor import TrafficMonitor
        monitor = TrafficMonitor(model_path=model_path, fps=25)
        print("✓ 监测器创建成功")
        return monitor
    except Exception as e:
        print(f"✗ 监测器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=== 详细错误测试 ===\n")
    
    # 测试导入
    test_imports()
    
    # 测试模型
    test_model()
    
    # 测试监测器
    test_monitor()
    
    print("\n=== 测试完成 ===")