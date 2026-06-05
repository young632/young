# -*- coding: utf-8 -*-
"""带详细错误日志的测试服务器"""

from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import time
import os
import traceback

app = Flask(__name__)

# 全局状态
stats = {
    "total_flow": 0,
    "car": 0,
    "bus": 0,
    "truck": 0,
    "avg_speed": 0,
    "congestion": 0
}

# YOLO相关
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✓ ultralytics导入成功")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"✗ ultralytics导入失败: {e}")

def find_yolo_model():
    """查找YOLO模型"""
    model_paths = [
        os.path.join(os.path.dirname(__file__), 'yolov8n.pt'),
        r"D:\计算机实践\yolov8n.pt",
        r"D:\计算机实践\yolov8n-seg.pt"
    ]
    for mp in model_paths:
        if os.path.exists(mp):
            print(f"✓ 找到模型: {mp}")
            return mp
    print("✗ 未找到YOLO模型")
    return None

@app.route('/video_feed')
def video_feed():
    """视频流接口"""
    global stats
    
    print("\n=== 收到视频流请求 ===")
    
    try:
        filename = request.args.get('file')
        mode = request.args.get('mode', 'normal')
        
        print(f"文件名: {filename}")
        print(f"模式: {mode}")
        
        if not filename:
            print("错误: 缺少文件名")
            return jsonify({'error': '缺少文件名'}), 400
        
        video_path = os.path.join(os.path.dirname(__file__), 'uploads', filename)
        
        print(f"视频路径: {video_path}")
        
        if not os.path.exists(video_path):
            print("错误: 文件不存在")
            return jsonify({'error': '文件不存在'}), 404
        
        print("✓ 文件存在")
        
        # 查找YOLO模型
        model_path = find_yolo_model()
        
        # 创建YOLO模型实例
        model = None
        if mode == 'detect' and YOLO_AVAILABLE and model_path:
            try:
                model = YOLO(model_path)
                print("✓ YOLO模型加载成功")
            except Exception as e:
                print(f"✗ YOLO模型加载失败: {e}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("错误: 无法打开视频")
            return jsonify({'error': '无法打开视频'}), 500
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = 1.0 / fps
        print(f"✓ 视频打开成功，帧率: {fps}")
        
        def generate():
            global stats
            frame_count = 0
            
            print("开始生成视频流...")
            
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                frame_count += 1
                
                if not ret:
                    print(f"视频读取结束，共处理 {frame_count} 帧")
                    break
                
                processed_frame = frame
                
                if mode == 'detect' and model:
                    try:
                        results = model(frame, verbose=False, conf=0.35, classes=[2, 5, 7])
                        
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                cls = int(box.cls.item())
                                colors = {2: (0, 255, 0), 5: (255, 255, 0), 7: (0, 255, 255)}
                                color = colors.get(cls, (0, 255, 0))
                                cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        stats['total_flow'] += 1
                        stats['car'] = sum(1 for r in results for b in r.boxes if int(b.cls.item()) == 2)
                        stats['bus'] = sum(1 for r in results for b in r.boxes if int(b.cls.item()) == 5)
                        stats['truck'] = sum(1 for r in results for b in r.boxes if int(b.cls.item()) == 7)
                        
                    except Exception as e:
                        print(f"检测错误: {e}")
                        traceback.print_exc()
                
                # 编码
                ret_jpg, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret_jpg:
                    print(f"帧 {frame_count}: 编码失败")
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            
            cap.release()
            print("视频流结束")
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    except Exception as e:
        print(f"视频流接口异常: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/data')
def get_data():
    """获取统计数据"""
    global stats
    return jsonify(stats)

@app.route('/')
def index():
    return "测试服务器运行中"

if __name__ == '__main__':
    print("=== 测试服务器启动 ===")
    print(f"YOLO可用: {YOLO_AVAILABLE}")
    find_yolo_model()
    print("服务器运行: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
