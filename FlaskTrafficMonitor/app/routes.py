from flask import Blueprint, render_template, request, jsonify, send_from_directory, Response
import os
import cv2
import time
from datetime import datetime

main_bp = Blueprint('main', __name__)

# 全局变量
monitor = None
stats = {
    "total_flow": 0,
    "car": 0,
    "bus": 0,
    "truck": 0,
    "avg_speed": 0,
    "congestion": 0
}

@main_bp.route('/test')
def test():
    """测试视频流页面"""
    return send_from_directory(os.path.dirname(os.path.dirname(__file__)), 'test_stream.html')

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/upload')
def upload_page():
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    videos = []
    if os.path.exists(upload_folder):
        for f in os.listdir(upload_folder):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append(f)
    return render_template('upload.html', videos=videos)

@main_bp.route('/monitor')
def monitor_page():
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    videos = []
    if os.path.exists(upload_folder):
        for f in os.listdir(upload_folder):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append(f)
    return render_template('monitor.html', videos=videos)

@main_bp.route('/statistics')
def statistics_page():
    return render_template('statistics.html')

@main_bp.route('/heatmap')
def heatmap_page():
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    videos = []
    if os.path.exists(upload_folder):
        for f in os.listdir(upload_folder):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append(f)
    return render_template('heatmap.html', videos=videos)

@main_bp.route('/history')
def history_page():
    output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    history_files = []
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            if f.endswith('.mp4') and '_detected' in f:
                original_name = f.replace('_detected', '')
                file_path = os.path.join(output_folder, f)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                history_files.append({
                    'name': f,
                    'original_name': original_name,
                    'size': f'{file_size:.2f} MB',
                    'date': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M')
                })
    return render_template('history.html', files=history_files)

@main_bp.route('/video_feed')
def video_feed():
    """视频流接口 - 使用标准MJPEG格式，完整调用YOLOv8检测算法"""
    global monitor, stats
    
    try:
        filename = request.args.get('file')
        mode = request.args.get('mode', 'normal')
        
        print(f"=== 视频流请求 ===")
        print(f"文件名: {filename}")
        print(f"模式: {mode}")
        
        if not filename:
            print("错误: 缺少文件名")
            return jsonify({'error': '缺少文件名'}), 400
        
        video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads', filename)
        
        print(f"视频路径: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"错误: 文件不存在")
            return jsonify({'error': '文件不存在'}), 404
        
        print(f"✓ 文件存在")
        
        # 延迟导入避免启动时加载模型
        from app.models.traffic_monitor import TrafficMonitor
        
        # 查找YOLO模型
        model_path = None
        model_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov8n.pt'),
            r"D:\计算机实践\yolov8n.pt",
            r"D:\计算机实践\yolov8n-seg.pt"
        ]
        for mp in model_paths:
            if os.path.exists(mp):
                model_path = mp
                print(f"✓ 找到YOLO模型: {model_path}")
                break
        
        if model_path is None:
            print("警告: 未找到YOLO模型，将使用背景减法")
        
        # 创建监测器实例
        if monitor is None:
            print("创建新的监测器实例...")
            monitor = TrafficMonitor(model_path=model_path, fps=25)
            print("✓ 监测器创建成功")
        else:
            monitor.reset()
            print("✓ 监测器已重置")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("错误: 无法打开视频文件")
            return jsonify({'error': '无法打开视频'}), 500
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = 1.0 / fps
        print(f"✓ 视频打开成功，帧率: {fps}")
        
        frame_count = 0
        
        def generate():
            global stats
            nonlocal frame_count
            
            print("开始生成视频流...")
            
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                frame_count += 1
                
                if not ret:
                    print(f"视频读取结束，共处理 {frame_count} 帧")
                    break
                
                processed_frame = frame
                
                if mode == 'detect':
                    print(f"帧 {frame_count}: 执行YOLO检测...")
                    try:
                        processed_frame, frame_stats = monitor.process_frame(frame)
                        print(f"帧 {frame_count}: 检测完成，车流量={frame_stats.get('count', 0)}")
                        
                        # 更新统计数据
                        stats['total_flow'] = frame_stats.get('count', 0)
                        vehicle_counts = frame_stats.get('vehicle_counts', {})
                        stats['car'] = vehicle_counts.get(0, 0)      # 轿车
                        stats['bus'] = vehicle_counts.get(2, 0)      # 公交
                        stats['truck'] = vehicle_counts.get(1, 0)    # 货车
                        stats['avg_speed'] = round(frame_stats.get('avg_speed', 0), 1)
                        stats['congestion'] = int(frame_stats.get('congestion', 0))
                    except Exception as e:
                        print(f"帧 {frame_count}: 检测失败 - {e}")
                        import traceback
                        traceback.print_exc()
                        processed_frame = frame
                
                # 编码为JPEG
                ret_jpg, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret_jpg:
                    print(f"帧 {frame_count}: 编码失败")
                    continue
                
                frame_bytes = jpeg.tobytes()
                # 输出标准MJPEG流
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # 帧率控制，避免倍速
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            
            cap.release()
            print("视频流结束")
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    except Exception as e:
        print(f"视频流接口异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@main_bp.route('/data')
def get_data():
    """获取实时统计数据"""
    global stats
    return jsonify(stats)

@main_bp.route('/upload_video', methods=['POST'])
def upload_video():
    # 支持两种参数名：video（首页使用）和 file（其他页面使用）
    if 'video' in request.files:
        file = request.files['video']
    elif 'file' in request.files:
        file = request.files['file']
    else:
        return jsonify({'status': 'error', 'message': '未选择文件'})
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '未选择文件'})
    
    if file and file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        filename = file.filename
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return jsonify({'status': 'success', 'message': '上传成功', 'filename': filename})
    
    return jsonify({'status': 'error', 'message': '不支持的格式'})

@main_bp.route('/stream_video')
def stream_video():
    """首页大屏视频流接口 - 兼容首页使用的接口名"""
    # 转换参数名：filename -> file
    filename = request.args.get('filename')
    mode = request.args.get('mode', 'detect')
    
    if not filename:
        return jsonify({'error': '缺少文件名'}), 400
    
    # 重定向到现有的video_feed接口
    from flask import redirect, url_for
    return redirect(f'/video_feed?file={filename}&mode={mode}')

@main_bp.route('/api/get_realtime_data')
def get_realtime_data():
    """首页大屏实时数据接口 - 兼容首页使用的接口名和数据格式"""
    global stats
    
    # 转换数据格式以匹配首页期望的格式
    congestion_level = '畅通'
    if stats.get('congestion', 0) > 70:
        congestion_level = '拥堵'
    elif stats.get('congestion', 0) > 40:
        congestion_level = '缓行'
    
    return jsonify({
        'total_count': stats.get('total_flow', 0),
        'avg_speed': stats.get('avg_speed', 0),
        'current_vehicles': stats.get('car', 0) + stats.get('bus', 0) + stats.get('truck', 0),
        'congestion_level': congestion_level,
        'car_count': stats.get('car', 0),
        'truck_count': stats.get('truck', 0),
        'bus_count': stats.get('bus', 0),
        'min_speed': max(0, int(stats.get('avg_speed', 0) - 10)),
        'max_speed': int(stats.get('avg_speed', 0) + 10),
        'processed_frames': stats.get('total_flow', 0) * 10
    })

@main_bp.route('/download/<filename>')
def download_file(filename):
    output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    return send_from_directory(output_folder, filename, as_attachment=True)

@main_bp.route('/delete_video/<filename>')
def delete_video(filename):
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    filepath = os.path.join(upload_folder, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'status': 'success', 'message': '删除成功'})
    return jsonify({'status': 'error', 'message': '文件不存在'})

@main_bp.route('/clear_history')
def clear_history():
    output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            if f.endswith('.mp4'):
                os.remove(os.path.join(output_folder, f))
    return jsonify({'status': 'success', 'message': '已清空'})
