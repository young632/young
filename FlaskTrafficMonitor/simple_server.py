# -*- coding: utf-8 -*-
"""简化版视频流服务器测试"""

from flask import Flask, Response
import cv2
import time
import os

app = Flask(__name__)

@app.route('/test_stream')
def test_stream():
    print("=== 收到视频流请求 ===")
    
    video_path = r"D:\计算机实践\FlaskTrafficMonitor\uploads\traffic_rideo.mp4"
    
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在 {video_path}")
        return "文件不存在", 404
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("错误: 无法打开视频")
        return "无法打开视频", 500
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = 1.0 / fps
    print(f"视频帧率: {fps}")
    
    def generate():
        print("开始生成视频流...")
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                print("视频读取结束")
                break
            
            # 编码
            ret_jpg, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret_jpg:
                continue
            
            # 输出MJPEG流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # 帧率控制
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        
        cap.release()
        print("视频流结束")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<html><body><img src="/test_stream"></body></html>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)