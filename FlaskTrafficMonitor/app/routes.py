from flask import Blueprint, render_template, request, jsonify, send_from_directory, Response
import os
import cv2
import time
import numpy as np
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
    "congestion": 0,
    "speed_records": [],
    "frame_count": 0,
    "congestion_history": [0, 0, 0, 0, 0, 0],
    "car_avg_speed": 0,
    "truck_avg_speed": 0,
    "bus_avg_speed": 0,
    "traffic_trend": [120, 150, 135, 180, 165, 200, 175, 190, 210, 185]
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
    global stats
    return render_template('statistics.html',
        total_count=stats.get('total_flow', 0),
        avg_speed=stats.get('avg_speed', 0),
        congestion=stats.get('congestion', 0),
        processed_frames=stats.get('frame_count', 0),
        car_count=stats.get('car', 0),
        truck_count=stats.get('truck', 0),
        bus_count=stats.get('bus', 0),
        car_ratio=0,
        truck_ratio=0,
        bus_ratio=0,
        car_avg_speed=stats.get('car_avg_speed', 0),
        truck_avg_speed=stats.get('truck_avg_speed', 0),
        bus_avg_speed=stats.get('bus_avg_speed', 0)
    )

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
        
        print(f"[OK] 文件存在")
        
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
                print(f"[OK] 找到YOLO模型: {model_path}")
                break
        
        if model_path is None:
            print("警告: 未找到YOLO模型，将使用背景减法")
        
        # 创建监测器实例
        if monitor is None:
            print("创建新的监测器实例...")
            monitor = TrafficMonitor(model_path=model_path, fps=25)
            print("[OK] 监测器创建成功")
        else:
            monitor.reset()
            print("[OK] 监测器已重置")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("错误: 无法打开视频文件")
            return jsonify({'error': '无法打开视频'}), 500
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = 1.0 / fps
        print(f"[OK] 视频打开成功，帧率: {fps}")
        
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
                        stats['frame_count'] = frame_count
                        
                        # 更新车型平均速度
                        stats['car_avg_speed'] = round(frame_stats.get('car_avg_speed', 0), 1)
                        stats['truck_avg_speed'] = round(frame_stats.get('truck_avg_speed', 0), 1)
                        stats['bus_avg_speed'] = round(frame_stats.get('bus_avg_speed', 0), 1)
                        
                        # 更新速度记录（用于车速分布统计）
                        if stats['avg_speed'] > 0:
                            stats['speed_records'].append({
                                'speed': stats['avg_speed'],
                                'frame': frame_count
                            })
                            # 保持最多1000条记录
                            if len(stats['speed_records']) > 1000:
                                stats['speed_records'] = stats['speed_records'][-1000:]
                        
                        # 更新拥堵历史（用于趋势图）
                        congestion_history = stats.get('congestion_history', [0, 0, 0, 0, 0, 0])
                        congestion_history.append(stats['congestion'])
                        if len(congestion_history) > 6:
                            congestion_history = congestion_history[-6:]
                        stats['congestion_history'] = congestion_history
                        
                        # 更新车流量趋势（每处理一定帧数更新一次）
                        if frame_count % 60 == 0:
                            traffic_trend = stats.get('traffic_trend', [120, 150, 135, 180, 165, 200, 175, 190, 210, 185])
                            current_flow = stats.get('total_flow', 0)
                            # 根据当前流量生成新的趋势数据
                            new_point = current_flow + int((traffic_trend[-1] - current_flow) * 0.3)
                            traffic_trend.append(new_point)
                            if len(traffic_trend) > 10:
                                traffic_trend = traffic_trend[-10:]
                            stats['traffic_trend'] = traffic_trend
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

@main_bp.route('/get_stats')
def get_stats():
    """热力图页面统计数据接口"""
    global monitor, stats
    
    if monitor:
        heatmap_stats = monitor.get_heatmap_stats()
        return jsonify({
            'current_vehicles': stats.get('total_flow', 0),
            'congestion_index': stats.get('congestion', 0),
            'hotspot_count': heatmap_stats.get('hotspot_count', 0),
            'max_heat': heatmap_stats.get('max_heat', 0),
            'avg_heat': heatmap_stats.get('avg_heat', 0)
        })
    else:
        return jsonify({
            'current_vehicles': 0,
            'congestion_index': 0,
            'hotspot_count': 0,
            'max_heat': 0,
            'avg_heat': 0
        })

@main_bp.route('/stream_video')
def stream_video():
    """首页大屏视频流接口 - 兼容首页使用的接口名"""
    # 直接返回视频流，不使用重定向
    global monitor, stats
    
    filename = request.args.get('filename')
    mode = request.args.get('mode', 'detect')
    
    if not filename:
        return jsonify({'error': '缺少文件名'}), 400
    
    video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads', filename)
    
    if not os.path.exists(video_path):
        return jsonify({'error': '文件不存在'}), 404
    
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
            break
    
    # 创建监测器实例
    if monitor is None:
        monitor = TrafficMonitor(model_path=model_path, fps=25)
        # 启用热力图显示
        monitor.show_heatmap = True
    else:
        monitor.reset()
        monitor.show_heatmap = True
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return jsonify({'error': '无法打开视频'}), 500
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = 1.0 / fps
    frame_count = 0
    
    def generate():
        global stats
        nonlocal frame_count
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            frame_count += 1
            
            if not ret:
                break
            
            processed_frame = frame
            
            if mode == 'detect':
                try:
                    processed_frame, frame_stats = monitor.process_frame(frame)
                    
                    # 更新统计数据
                    stats['total_flow'] = frame_stats.get('count', 0)
                    vehicle_counts = frame_stats.get('vehicle_counts', {})
                    stats['car'] = vehicle_counts.get(0, 0)
                    stats['bus'] = vehicle_counts.get(2, 0)
                    stats['truck'] = vehicle_counts.get(1, 0)
                    stats['avg_speed'] = round(frame_stats.get('avg_speed', 0), 1)
                    stats['congestion'] = int(frame_stats.get('congestion', 0))
                    
                    # 更新帧数
                    stats['frame_count'] = frame_count
                    
                    # 保存速度记录
                    speed_records = stats.get('speed_records', [])
                    current_speed = frame_stats.get('avg_speed', 0)
                    if current_speed > 0:
                        speed_records.append({'speed': current_speed, 'timestamp': time.time()})
                        # 保留最近1000条记录
                        if len(speed_records) > 1000:
                            speed_records = speed_records[-1000:]
                    stats['speed_records'] = speed_records
                    
                    # 保存各车型速度记录（用于车速分布统计）
                    car_speed = frame_stats.get('car_avg_speed', 0)
                    truck_speed = frame_stats.get('truck_avg_speed', 0)
                    bus_speed = frame_stats.get('bus_avg_speed', 0)
                    
                    if car_speed > 0:
                        speed_records.append({'speed': car_speed, 'timestamp': time.time(), 'type': 'car'})
                    if truck_speed > 0:
                        speed_records.append({'speed': truck_speed, 'timestamp': time.time(), 'type': 'truck'})
                    if bus_speed > 0:
                        speed_records.append({'speed': bus_speed, 'timestamp': time.time(), 'type': 'bus'})
                    
                    # 更新拥堵历史
                    congestion_history = stats.get('congestion_history', [0, 0, 0, 0, 0, 0])
                    congestion_history.append(stats['congestion'])
                    if len(congestion_history) > 6:
                        congestion_history = congestion_history[-6:]
                    stats['congestion_history'] = congestion_history
                    
                    # 更新各车型平均速度
                    stats['car_avg_speed'] = round(frame_stats.get('car_avg_speed', stats.get('avg_speed', 0)), 1)
                    stats['truck_avg_speed'] = round(frame_stats.get('truck_avg_speed', stats.get('avg_speed', 0) * 0.7), 1)
                    stats['bus_avg_speed'] = round(frame_stats.get('bus_avg_speed', stats.get('avg_speed', 0) * 0.6), 1)
                    
                except Exception as e:
                    print(f"检测失败: {e}")
                    processed_frame = frame
            
            # 编码为JPEG
            ret_jpg, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret_jpg:
                continue
            
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 帧率控制
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    
    # 计算车流量趋势（最近10分钟数据）
    traffic_trend = stats.get('traffic_trend', [120, 150, 135, 180, 165, 200, 175, 190, 210, 185])
    
    # 计算车速分布
    speed_records = stats.get('speed_records', [])
    speed_bins = [0, 20, 40, 60, 80, float('inf')]
    speed_distribution = [0, 0, 0, 0, 0]
    
    for record in speed_records:
        speed = record.get('speed', 0)
        for i in range(len(speed_bins) - 1):
            if speed_bins[i] <= speed < speed_bins[i + 1]:
                speed_distribution[i] += 1
                break
    
    # 如果没有真实数据，使用默认值
    if sum(speed_distribution) == 0:
        speed_distribution = [15, 35, 30, 15, 5]
    
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
        'processed_frames': stats.get('frame_count', 0),
        'traffic_trend': traffic_trend,
        'speed_distribution': speed_distribution
    })

@main_bp.route('/api/get_statistics')
def get_statistics():
    """获取统计数据接口 - 用于统计页面的图表数据"""
    global stats
    
    print(f"[DEBUG] stats: {stats}")
    
    speed_records = stats.get('speed_records', [])
    
    # 计算车速分布（基于真实速度记录）
    speed_bins = [0, 20, 40, 60, 80, 100, float('inf')]
    speed_distribution = [0, 0, 0, 0, 0, 0]
    
    for record in speed_records:
        speed = record.get('speed', 0)
        for i in range(len(speed_bins) - 1):
            if speed_bins[i] <= speed < speed_bins[i + 1]:
                speed_distribution[i] += 1
                break
    
    # 获取车流量趋势（使用实时拥堵历史数据转换）
    congestion_history = stats.get('congestion_history', [10, 10, 10, 10, 10, 10])
    # 将拥堵指数转换为相对车流量（拥堵越高，车流量越大）
    base_traffic = stats.get('total_flow', 100)
    hourly_trend = [
        max(1, int(base_traffic * (congestion_history[0] / 100) * 0.8)),
        max(1, int(base_traffic * (congestion_history[1] / 100) * 0.6)),
        max(1, int(base_traffic * (congestion_history[2] / 100) * 1.2)),
        max(1, int(base_traffic * (congestion_history[3] / 100) * 0.9)),
        max(1, int(base_traffic * (congestion_history[4] / 100) * 1.1)),
        max(1, int(base_traffic * (congestion_history[5] / 100) * 0.7)),
    ]
    
    # 车型分布
    car_count = stats.get('car', 0)
    truck_count = stats.get('truck', 0)
    bus_count = stats.get('bus', 0)
    vehicle_total = car_count + truck_count + bus_count
    
    # 计算平均速度（基于真实记录）
    valid_speeds = [r.get('speed', 0) for r in speed_records if r.get('speed', 0) > 0]
    actual_avg_speed = sum(valid_speeds) / len(valid_speeds) if valid_speeds else stats.get('avg_speed', 0)
    
    return jsonify({
        'total_count': stats.get('total_flow', 0),
        'avg_speed': round(actual_avg_speed, 1),
        'congestion': round(stats.get('congestion', 0), 1),
        'processed_frames': stats.get('frame_count', 0),
        'car_count': car_count,
        'truck_count': truck_count,
        'bus_count': bus_count,
        'car_ratio': round((car_count / vehicle_total) * 100, 1) if vehicle_total > 0 else 0,
        'truck_ratio': round((truck_count / vehicle_total) * 100, 1) if vehicle_total > 0 else 0,
        'bus_ratio': round((bus_count / vehicle_total) * 100, 1) if vehicle_total > 0 else 0,
        'car_avg_speed': round(stats.get('car_avg_speed', actual_avg_speed), 1),
        'truck_avg_speed': round(stats.get('truck_avg_speed', actual_avg_speed * 0.7), 1),
        'bus_avg_speed': round(stats.get('bus_avg_speed', actual_avg_speed * 0.6), 1),
        'speed_distribution': speed_distribution,
        'hourly_trend': hourly_trend,
        'congestion_history': congestion_history,
        'speed_record_count': len(speed_records),
        'last_update': datetime.now().isoformat()
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

@main_bp.route('/process_video', methods=['POST'])
def process_video():
    """处理视频并保存检测结果"""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'status': 'error', 'message': '缺少文件名'})
    
    video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads', filename)
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', filename.replace('.mp4', '_detected.mp4'))
    
    if not os.path.exists(video_path):
        return jsonify({'status': 'error', 'message': '文件不存在'})
    
    try:
        import threading
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
                break
        
        monitor = TrafficMonitor(model_path=model_path, fps=25)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return jsonify({'status': 'error', 'message': '无法打开视频'})
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, _ = monitor.process_frame(frame)
            out.write(processed_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count} 帧")
        
        cap.release()
        out.release()
        
        return jsonify({
            'status': 'success',
            'message': f'处理完成，共 {frame_count} 帧',
            'output_file': filename.replace('.mp4', '_detected.mp4')
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})
