# -*- coding: utf-8 -*-
"""完整的交通监测网页服务器 - 单文件版本"""

from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import time
import os

app = Flask(__name__)

# 全局状态
monitor = None
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
except ImportError:
    YOLO_AVAILABLE = False

def find_yolo_model():
    """查找YOLO模型"""
    model_paths = [
        os.path.join(os.path.dirname(__file__), 'yolov8n.pt'),
        r"D:\计算机实践\yolov8n.pt",
        r"D:\计算机实践\yolov8n-seg.pt"
    ]
    for mp in model_paths:
        if os.path.exists(mp):
            return mp
    return None

# 前端模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交通监测系统</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: white; margin: 0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; color: #00d4ff; margin-bottom: 30px; }
        .controls { display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
        select, button { padding: 10px 20px; font-size: 16px; border: none; border-radius: 5px; }
        select { background: #2d2d44; color: white; }
        button { cursor: pointer; }
        .btn-start { background: #00d4ff; color: black; font-weight: bold; }
        .btn-stop { background: #ff4757; color: white; }
        .video-container { background: #0f0f1a; border-radius: 10px; overflow: hidden; }
        #videoArea { width: 100%; display: none; }
        .placeholder { text-align: center; padding: 100px 0; color: #666; }
        .stats-panel { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 20px; }
        .stat-card { background: #2d2d44; padding: 20px; border-radius: 10px; text-align: center; }
        .stat-value { font-size: 32px; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 14px; color: #888; margin-top: 5px; }
        .status { margin-bottom: 15px; font-size: 18px; }
        .status.detecting { color: #2ed573; }
        .status.playing { color: #00d4ff; }
        .status.stopped { color: #ff4757; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 交通监测系统</h1>
        
        <div class="controls">
            <select id="video-select">
                <option value="">选择视频文件</option>
                {% for video in videos %}
                <option value="{{ video }}">{{ video }}</option>
                {% endfor %}
            </select>
            <select id="stream-mode">
                <option value="detect">带检测</option>
                <option value="normal">仅视频</option>
            </select>
            <button class="btn-start" id="start-stream" disabled>开始播放</button>
            <button class="btn-stop" id="stop-stream" disabled>停止播放</button>
        </div>
        
        <div class="status stopped" id="stream-status">状态: 未连接</div>
        
        <div class="video-container" id="video-container">
            <div class="placeholder" id="placeholder">请选择视频并点击开始播放</div>
            <img id="videoArea" />
        </div>
        
        <div class="stats-panel">
            <div class="stat-card">
                <div class="stat-value" id="total-count">0</div>
                <div class="stat-label">总车流量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-speed">0</div>
                <div class="stat-label">平均车速 (km/h)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="car-count">0</div>
                <div class="stat-label">轿车</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="bus-count">0</div>
                <div class="stat-label">公交</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="truck-count">0</div>
                <div class="stat-label">货车</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="congestion-value">0</div>
                <div class="stat-label">拥堵指数</div>
            </div>
        </div>
    </div>

    <script>
        function play() {
            var file = document.getElementById("video-select").value;
            var mode = document.getElementById("stream-mode").value;
            
            var img = document.getElementById("videoArea");
            img.src = '/video_feed?file=' + encodeURIComponent(file) + '&mode=' + mode;
            img.style.display = 'block';
            document.getElementById('placeholder').style.display = 'none';
            
            document.getElementById('start-stream').disabled = true;
            document.getElementById('stop-stream').disabled = false;
            document.getElementById('video-select').disabled = true;
            document.getElementById('stream-mode').disabled = true;
            
            var status = document.getElementById('stream-status');
            status.textContent = '状态: ' + (mode === 'detect' ? '检测中' : '播放中');
            status.className = 'status ' + (mode === 'detect' ? 'detecting' : 'playing');
        }

        function stop() {
            document.getElementById("videoArea").src = "";
            document.getElementById("videoArea").style.display = 'none';
            document.getElementById('placeholder').style.display = 'block';
            
            document.getElementById('start-stream').disabled = false;
            document.getElementById('stop-stream').disabled = true;
            document.getElementById('video-select').disabled = false;
            document.getElementById('stream-mode').disabled = false;
            
            var status = document.getElementById('stream-status');
            status.textContent = '状态: 已停止';
            status.className = 'status stopped';
        }

        setInterval(function() {
            fetch('/data')
                .then(function(res) { return res.json(); })
                .then(function(data) {
                    document.getElementById('total-count').textContent = data.total_flow || 0;
                    document.getElementById('avg-speed').textContent = data.avg_speed || 0;
                    document.getElementById('car-count').textContent = data.car || 0;
                    document.getElementById('bus-count').textContent = data.bus || 0;
                    document.getElementById('truck-count').textContent = data.truck || 0;
                    document.getElementById('congestion-value').textContent = data.congestion || 0;
                })
                .catch(function(error) { console.error('加载失败:', error); });
        }, 500);

        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('start-stream').addEventListener('click', play);
            document.getElementById('stop-stream').addEventListener('click', stop);
            
            document.getElementById('video-select').addEventListener('change', function() {
                document.getElementById('start-stream').disabled = this.value === '';
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页面"""
    upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
    videos = []
    if os.path.exists(upload_folder):
        for f in os.listdir(upload_folder):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append(f)
    return render_template_string(HTML_TEMPLATE, videos=videos)

@app.route('/video_feed')
def video_feed():
    """视频流接口"""
    global monitor, stats
    
    filename = request.args.get('file')
    mode = request.args.get('mode', 'normal')
    
    if not filename:
        return jsonify({'error': '缺少文件名'}), 400
    
    video_path = os.path.join(os.path.dirname(__file__), 'uploads', filename)
    
    if not os.path.exists(video_path):
        return jsonify({'error': '文件不存在'}), 404
    
    # 查找YOLO模型
    model_path = find_yolo_model()
    
    # 创建YOLO模型实例
    if mode == 'detect' and YOLO_AVAILABLE and model_path:
        try:
            model = YOLO(model_path)
            print(f"✓ 加载YOLO模型: {model_path}")
        except Exception as e:
            print(f"✗ YOLO模型加载失败: {e}")
            model = None
    else:
        model = None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'error': '无法打开视频'}), 500
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = 1.0 / fps
    
    def generate():
        global stats
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                break
            
            processed_frame = frame
            
            if mode == 'detect' and model:
                # 执行YOLO检测
                try:
                    results = model(frame, verbose=False, conf=0.35, classes=[2, 5, 7])
                    
                    # 绘制检测框
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls = int(box.cls.item())
                            conf = float(box.conf.item())
                            
                            # 颜色
                            colors = {2: (0, 255, 0), 5: (255, 255, 0), 7: (0, 255, 255)}
                            class_names = {2: '轿车', 5: '公交', 7: '货车'}
                            
                            color = colors.get(cls, (0, 255, 0))
                            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(processed_frame, f'{class_names.get(cls, cls)} {conf:.2f}', 
                                        (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 更新统计
                    stats['total_flow'] += 1
                    stats['car'] = sum(1 for r in results for b in r.boxes if int(b.cls.item()) == 2)
                    stats['bus'] = sum(1 for r in results for b in r.boxes if int(b.cls.item()) == 5)
                    stats['truck'] = sum(1 for r in results for b in r.boxes if int(b.cls.item()) == 7)
                    stats['avg_speed'] = round(30 + stats['total_flow'] * 0.5, 1)
                    stats['congestion'] = min(100, stats['total_flow'] * 2)
                    
                except Exception as e:
                    print(f"检测错误: {e}")
            
            # 编码
            ret_jpg, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret_jpg:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # 帧率控制
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data():
    """获取统计数据"""
    global stats
    return jsonify(stats)

if __name__ == '__main__':
    print("=== 交通监测系统启动 ===")
    print(f"YOLO可用: {YOLO_AVAILABLE}")
    model_path = find_yolo_model()
    print(f"YOLO模型: {model_path if model_path else '未找到'}")
    print("服务器运行: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
