#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能交通视觉监测系统 - Web服务端
集成实时视频流、数据统计、异常检测
"""

import os
import cv2
import json
import time
import threading
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_cors import CORS

# 导入交通监测系统
from traffic_monitor import TrafficMonitor

app = Flask(__name__)
CORS(app)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 全局变量
monitor = None
video_source = None
is_running = False
current_frame = None
frame_lock = threading.Lock()
last_stats = {}
stats_lock = threading.Lock()

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_monitor(model_path=None):
    """初始化交通监测系统"""
    global monitor
    monitor = TrafficMonitor(speed_limit=60, fps=24, model_path=model_path)
    monitor.statistics.start_csv_logging('traffic_log.csv')
    print("交通监测系统初始化完成")

def generate_frames():
    """生成视频帧流 - 优化版"""
    global monitor, video_source, is_running, current_frame
    
    cap = None
    try:
        # 尝试打开视频源
        if video_source is None:
            cap = cv2.VideoCapture(0)  # 默认使用摄像头
        elif video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return
        
        print(f"成功打开视频源: {video_source}")
        
        # 设置摄像头参数以提高性能
        if video_source is None or video_source.isdigit():
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while is_running:
            ret, frame = cap.read()
            if not ret:
                # 视频结束，重新播放
                if video_source and not video_source.isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            # 处理帧
            result, stats = monitor.process_frame(frame)
            
            # 更新统计数据
            with stats_lock:
                global last_stats
                last_stats = stats
            
            # 更新当前帧
            with frame_lock:
                current_frame = result.copy()
            
            # 编码为JPEG - 提高画质
            ret, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 动态延迟，不固定等待
            time.sleep(0.01)
            
    except Exception as e:
        print(f"视频处理错误: {e}")
    finally:
        if cap:
            cap.release()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流端点"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_monitor():
    """启动监测"""
    global is_running, video_source
    
    data = request.get_json()
    video_source = data.get('source', None)
    
    if not is_running:
        # 重新初始化监测器
        init_monitor(model_path='yolov8n.pt')
        is_running = True
        
        # 启动视频处理线程
        threading.Thread(target=generate_frames, daemon=True).start()
        print("监测系统已启动")
    
    return jsonify({'status': 'started', 'source': video_source})

@app.route('/api/stop', methods=['POST'])
def stop_monitor():
    """停止监测"""
    global is_running
    
    is_running = False
    print("监测系统已停止")
    
    # 导出统计数据
    if monitor:
        monitor.export_data()
    
    return jsonify({'status': 'stopped'})

@app.route('/api/status')
def get_status():
    """获取监测状态"""
    with stats_lock:
        return jsonify({
            'is_running': is_running,
            'source': video_source,
            **last_stats
        })

@app.route('/api/statistics')
def get_statistics():
    """获取统计数据"""
    if not monitor:
        return jsonify({})
    
    stats = monitor.get_statistics()
    return jsonify({
        'total_count': stats['total_count'],
        'vehicle_counts': {
            '轿车': stats['vehicle_counts'].get(0, 0),
            '货车': stats['vehicle_counts'].get(1, 0),
            '公交': stats['vehicle_counts'].get(2, 0)
        },
        'avg_speed': round(stats['avg_speed'], 1),
        'max_speed': round(stats['max_speed'], 1),
        'congestion_index': round(stats['congestion_index'], 1),
        'warnings': stats['warnings']
    })

@app.route('/api/config', methods=['POST'])
def set_config():
    """设置配置参数"""
    global monitor
    
    data = request.get_json()
    if 'speed_limit' in data and monitor:
        monitor.set_speed_limit(data['speed_limit'])
    
    return jsonify({'status': 'success'})

@app.route('/api/enhance', methods=['POST'])
def set_enhance():
    """设置图像增强参数"""
    global monitor
    
    data = request.get_json()
    if monitor:
        if 'enable' in data:
            monitor.enable_enhancement = data['enable']
        if 'retinex' in data:
            monitor.enable_retinex = data['retinex']
        if 'clahe' in data:
            monitor.enable_clahe = data['clahe']
    
    return jsonify({'status': 'success'})

@app.route('/api/heatmap')
def get_heatmap():
    """获取热力图"""
    if not monitor or not current_frame:
        return jsonify({'error': 'No data available'})
    
    heatmap, _, _ = monitor.heatmap_gen.update(monitor.tracker.tracks, 
                                                current_frame.shape[1], 
                                                current_frame.shape[0])
    
    if heatmap is None:
        return jsonify({'error': 'Heatmap not available'})
    
    ret, buffer = cv2.imencode('.jpg', heatmap, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_bytes = buffer.tobytes()
    
    return Response(frame_bytes, mimetype='image/jpeg')

# ========== 新增模块 API ==========

@app.route('/api/driver_profiles')
def get_driver_profiles():
    """获取所有车辆的驾驶行为画像"""
    if not monitor:
        return jsonify({'profiles': {}})
    
    profiles = monitor.get_driver_profiles()
    return jsonify({'profiles': profiles})

@app.route('/api/driver_profile/<int:track_id>')
def get_driver_profile(track_id):
    """获取指定车辆的驾驶行为画像"""
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'})
    
    profile = monitor.get_driver_profile(track_id)
    if profile:
        return jsonify(profile)
    return jsonify({'error': 'Profile not found'})

@app.route('/api/behavior_timeline/<int:track_id>')
def get_behavior_timeline(track_id):
    """获取指定车辆的行为时间轴"""
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'})
    
    timeline = monitor.get_behavior_timeline(track_id)
    return jsonify({'timeline': timeline})

@app.route('/api/risk_summary')
def get_risk_summary():
    """获取风险评估汇总"""
    if not monitor:
        return jsonify({'error': 'Monitor not initialized'})
    
    summary = monitor.get_risk_summary()
    return jsonify(summary)

@app.route('/api/high_risk_events')
def get_high_risk_events():
    """获取高风险事件列表"""
    if not monitor:
        return jsonify({'events': []})
    
    events = monitor.get_high_risk_events()
    return jsonify({'events': events})

@app.route('/api/lane_statistics')
def get_lane_statistics():
    """获取车道流量统计"""
    if not monitor:
        return jsonify({'lanes': {}})
    
    stats = monitor.get_lane_statistics()
    return jsonify({'lanes': stats})

@app.route('/api/lane_predictions')
def get_lane_predictions():
    """获取车道流量预测"""
    if not monitor:
        return jsonify({'predictions': {}})
    
    predictions = monitor.get_lane_predictions()
    return jsonify({'predictions': predictions})

@app.route('/api/optimization_suggestions')
def get_optimization_suggestions():
    """获取车道优化建议"""
    if not monitor:
        return jsonify({'suggestions': []})
    
    suggestions = monitor.get_optimization_suggestions()
    return jsonify({'suggestions': suggestions})

@app.route('/api/display_settings', methods=['POST'])
def set_display_settings():
    """设置显示相关参数"""
    global monitor
    
    data = request.get_json()
    if monitor:
        if 'show_trajectory' in data:
            monitor.show_trajectory = data['show_trajectory']
        if 'enable_risk_warning' in data:
            monitor.enable_risk_warning = data['enable_risk_warning']
        if 'heatmap_opacity' in data:
            monitor.heatmap_opacity = max(0.1, min(1.0, data['heatmap_opacity']))
        if 'show_lane' in data:
            monitor.show_lane = data['show_lane']
        if 'show_heatmap' in data:
            monitor.show_heatmap = data['show_heatmap']
    
    return jsonify({'status': 'success'})

@app.route('/api/snapshot', methods=['POST'])
def take_snapshot():
    """截取快照"""
    global current_frame
    
    with frame_lock:
        if current_frame is None:
            return jsonify({'error': 'No frame available'})
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'snapshot_{timestamp}.jpg'
        cv2.imwrite(filename, current_frame)
    
    return jsonify({'status': 'success', 'filename': filename})

@app.route('/api/export', methods=['POST'])
def export_data():
    """导出数据"""
    if monitor:
        summary = monitor.export_data(
            csv_file='traffic_log.csv',
            summary_file='traffic_summary.json'
        )
        return jsonify({'status': 'success', 'summary': summary})
    return jsonify({'status': 'failed', 'message': 'Monitor not initialized'})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """上传视频文件"""
    # 检查是否有文件被上传
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    file = request.files['file']
    
    # 检查文件名
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '文件名不能为空'})
    
    # 检查文件类型
    if file and allowed_file(file.filename):
        # 确保上传目录存在
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # 生成唯一文件名
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f'{timestamp}.{ext}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 保存文件
        file.save(filepath)
        print(f"视频文件已上传: {filepath}")
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'path': filepath,
            'message': '视频上传成功'
        })
    else:
        return jsonify({'status': 'error', 'message': '不支持的文件格式，仅支持: mp4, avi, mov, mkv, webm'})

@app.route('/api/videos')
def get_video_list():
    """获取已上传的视频列表"""
    videos = []
    
    # 检查上传目录
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if allowed_file(filename):
                filepath = os.path.join(os.getcwd(), UPLOAD_FOLDER, filename)
                videos.append({
                    'name': filename,
                    'path': filepath
                })
    
    # 添加测试视频目录中的视频
    test_videos_dir = 'test_videos'
    if os.path.exists(test_videos_dir):
        for filename in os.listdir(test_videos_dir):
            if allowed_file(filename):
                filepath = os.path.join(os.getcwd(), test_videos_dir, filename)
                videos.append({
                    'name': filename,
                    'path': filepath
                })
    
    return jsonify({'videos': videos})

@app.route('/api/delete_video', methods=['POST'])
def delete_video():
    """删除视频文件"""
    data = request.get_json()
    filepath = data.get('path', '')
    
    if not filepath:
        return jsonify({'status': 'error', 'message': '文件路径不能为空'})
    
    # 安全检查：规范化路径并检查是否在允许的目录内
    filepath = os.path.normpath(filepath)
    upload_path = os.path.normpath(UPLOAD_FOLDER)
    test_path = os.path.normpath('test_videos')
    
    # 检查路径是否在允许的目录下
    full_upload_path = os.path.abspath(upload_path)
    full_test_path = os.path.abspath(test_path)
    full_file_path = os.path.abspath(filepath)
    
    if not (full_file_path.startswith(full_upload_path) or 
            full_file_path.startswith(full_test_path)):
        return jsonify({'status': 'error', 'message': '不允许删除此文件'})
    
    if os.path.exists(full_file_path):
        os.remove(full_file_path)
        return jsonify({'status': 'success', 'message': '文件已删除'})
    else:
        return jsonify({'status': 'error', 'message': '文件不存在'})

if __name__ == '__main__':
    # 初始化监测器
    init_monitor(model_path='yolov8n.pt')
    
    # 创建必要目录
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)