# -*- coding: utf-8 -*-
"""
智能交通视觉监测系统 - 完整扩展版
集成：车型分类 + 异常检测 + 数据统计 + 信号灯模拟 + 夜间增强 + Web看板
"""

import cv2
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import csv
import json
from datetime import datetime
import os

# 导入新模块
from driving_behavior import BehaviorLSTM, DriverProfile
from risk_assessment import VehicleInteractionAnalyzer
from lane_analysis import LaneFlowAnalyzer

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_center_distance(box1, box2):
    """计算两个框中心点距离"""
    cx1, cy1 = box1[0] + box1[2]//2, box1[1] + box1[3]//2
    cx2, cy2 = box2[0] + box2[2]//2, box2[1] + box2[3]//2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


class VehicleClassifier:
    """车型分类器 - 严格限定三类车型：轿车、公交、货车"""
    VEHICLE_TYPES = {
        0: {'name': '轿车', 'color': (0, 255, 0), 'speed_factor': 1.0},
        1: {'name': '货车', 'color': (255, 140, 0), 'speed_factor': 0.7},
        2: {'name': '公交', 'color': (0, 200, 255), 'speed_factor': 0.6},
    }
    
    def __init__(self):
        self.type_history = {}  # 记录每辆车的历史分类，防止跳变
        self.history_length = 10  # 增加历史帧数量，提升平滑效果
    
    def classify(self, box, yolo_class=None):
        """根据YOLO类别严格分类，只保留三类车型"""
        x, y, w, h = box
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # 严格基于COCO数据集类别，直接映射
        if yolo_class == 2:  # car -> 轿车
            return 0, 0.95
        elif yolo_class == 5:  # bus -> 公交
            return 2, 0.95
        elif yolo_class == 7:  # truck -> 货车
            return 1, 0.95
        
        # 如果没有YOLO类别，使用尺寸特征作为后备
        if aspect_ratio > 2.2 and area > 15000:
            return 2, 0.85  # 长条形 -> 公交
        elif area > 12000 and aspect_ratio < 1.6:
            return 1, 0.85  # 大型 -> 货车
        else:
            return 0, 0.80  # 中型 -> 轿车
    
    def get_stable_type(self, track_id, current_type):
        """获取稳定的车型，防止帧间跳变 - 使用Counter投票法"""
        from collections import Counter
        
        if track_id not in self.type_history:
            self.type_history[track_id] = []
        
        self.type_history[track_id].append(current_type)
        # 只保留最近N帧，避免内存爆炸
        if len(self.type_history[track_id]) > self.history_length:
            self.type_history[track_id].pop(0)
        
        # 如果历史只有一个类型，直接返回
        if len(self.type_history[track_id]) == 1:
            return current_type
        
        # Counter投票法：出现次数最多的类别作为最终结果
        counter = Counter(self.type_history[track_id])
        # most_common(1) 返回 [(类别, 次数)]
        return counter.most_common(1)[0][0]
    
    def cleanup(self, active_track_ids):
        """清理消失车辆的历史数据"""
        for track_id in list(self.type_history.keys()):
            if track_id not in active_track_ids:
                del self.type_history[track_id]


class LightweightLPR:
    """轻量车牌识别器 - 支持 HyperLPR 和简单模板匹配"""
    def __init__(self):
        self.lpr_available = False
        self.model = None
        self.track_plate = {}  # 绑定车辆ID和车牌
        
        # 尝试导入 HyperLPR
        try:
            from hyperlpr import HyperLPR_plate_recognition
            self.model = HyperLPR_plate_recognition
            self.lpr_available = True
            print("成功加载 HyperLPR 车牌识别模型")
        except ImportError:
            print("警告：未安装 HyperLPR，车牌识别功能不可用")
            print("安装方法: pip install hyperlpr")
    
    def recognize(self, car_crop):
        """识别车辆区域的车牌"""
        if not self.lpr_available or self.model is None:
            return None
        
        try:
            # 确保图像是 BGR 格式
            if len(car_crop.shape) == 2:
                car_crop = cv2.cvtColor(car_crop, cv2.COLOR_GRAY2BGR)
            
            # HyperLPR 识别（新API）
            result = self.model(car_crop)
            if result and len(result) > 0:
                plate_info = result[0]
                plate_number = plate_info[0]  # 车牌号码
                confidence = plate_info[1]    # 置信度
                if confidence > 0.7:  # 置信度阈值
                    return plate_number
            return None
        except Exception as e:
            print(f"LPR识别错误: {e}")
            return None
    
    def get_plate(self, track_id):
        """获取车辆的车牌"""
        return self.track_plate.get(track_id, None)
    
    def set_plate(self, track_id, plate_number):
        """绑定车辆ID和车牌"""
        self.track_plate[track_id] = plate_number
    
    def cleanup(self, active_track_ids):
        """清理消失车辆的车牌记录"""
        for track_id in list(self.track_plate.keys()):
            if track_id not in active_track_ids:
                del self.track_plate[track_id]


class ByteTracker:
    """ByteTrack风格多目标跟踪器 - 增强版"""
    def __init__(self, max_disappeared=8, max_distance=80, fps=24):
        self.next_id = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.fps = fps
    
    def register(self, box, confidence, frame_id, vehicle_type=0):
        track_id = self.next_id
        cx, cy = box[0] + box[2]//2, box[1] + box[3]//2
        self.tracks[track_id] = {
            'box': box,
            'disappeared': 0,
            'positions': deque(maxlen=30),
            'trajectory': deque(maxlen=50),
            'speed': 0.0,
            'speed_history': deque(maxlen=8),
            'first_seen': frame_id,
            'confidence': confidence,
            'vehicle_type': vehicle_type,
            'lane': 0,
            'reverse_count': 0,
            'lane_change_count': 0,
            'stopped_time': 0,
            'overtake_detected': False,
            'warnings': []
        }
        self.tracks[track_id]['positions'].append((cx, cy, frame_id))
        self.tracks[track_id]['trajectory'].append((cx, cy))
        self.next_id += 1
        return track_id
    
    def calculate_adaptive_speed(self, track, frame_id, img_height):
        positions = track['positions']
        if len(positions) < 5:
            return track.get('speed', 0.0)
        
        recent_positions = list(positions)[-10:]
        speed_samples = []
        
        for i in range(len(recent_positions) - 1):
            pos1, pos2 = recent_positions[i], recent_positions[i + 1]
            pixel_dist = abs(pos2[1] - pos1[1])
            time_diff = (pos2[2] - pos1[2]) / self.fps
            if 0.01 < time_diff < 2.0:
                instant_speed = pixel_dist / time_diff
                if instant_speed < 250:
                    speed_samples.append(instant_speed)
        
        if len(speed_samples) < 3:
            return track.get('speed', 0.0) * 0.9
        
        median_speed = np.median(speed_samples)
        current_y = recent_positions[-1][1]
        depth_ratio = current_y / img_height
        
        if depth_ratio < 0.4:
            pixel_to_kmh = 0.35
        elif depth_ratio < 0.6:
            pixel_to_kmh = 0.45
        else:
            pixel_to_kmh = 0.55
        
        raw_speed = median_speed * pixel_to_kmh
        track['speed_history'].append(raw_speed)
        
        if len(track['speed_history']) >= 4:
            recent_speeds = list(track['speed_history'])[-6:]
            mean_speed = np.mean(recent_speeds)
            std_speed = np.std(recent_speeds)
            smoothed_speed = mean_speed if std_speed > 15 else np.mean(recent_speeds[-5:])
        else:
            smoothed_speed = raw_speed
        
        return max(0, smoothed_speed)
    
    def detect_anomalies(self, track_id, track, prev_y):
        """检测异常行为"""
        warnings = []
        
        curr_cx, curr_cy = track['box'][0] + track['box'][2]//2, track['box'][1] + track['box'][3]//2
        track['trajectory'].append((curr_cx, curr_cy))
        
        if prev_y is not None:
            direction = curr_cy - prev_y
            if direction < -5:
                track['reverse_count'] += 1
                if track['reverse_count'] > 5:
                    warnings.append(('逆行', (255, 0, 255)))
            else:
                track['reverse_count'] = max(0, track['reverse_count'] - 1)
        
        speed = track.get('speed', 0)
        if speed > 120:
            warnings.append(('严重超速', (0, 0, 255)))
        elif speed > 80:
            warnings.append(('超速', (0, 165, 255)))
        
        if len(track['trajectory']) >= 10:
            traj = list(track['trajectory'])
            recent_traj = traj[-10:]
            x_changes = [recent_traj[i][0] - recent_traj[i-1][0] for i in range(1, len(recent_traj))]
            if len(x_changes) >= 3:
                lane_change_threshold = 30
                if abs(sum(x_changes[:3])) > lane_change_threshold:
                    track['lane_change_count'] += 1
                    if track['lane_change_count'] > 2:
                        warnings.append(('异常变道', (255, 200, 0)))
        
        if speed < 5 and len(track['positions']) > 50:
            track['stopped_time'] += 1
            if track['stopped_time'] > 120:
                warnings.append(('滞留/停车', (128, 0, 128)))
        else:
            track['stopped_time'] = 0
        
        track['warnings'] = warnings
        return warnings
    
    def update(self, high_conf_boxes, high_conf_scores, low_conf_boxes, low_conf_scores, frame_id, img_height=480, high_conf_types=None, low_conf_types=None):
        used_detections = set()
        matched_ids = set()
        
        if high_conf_types is None:
            high_conf_types = [0] * len(high_conf_boxes)
        if low_conf_types is None:
            low_conf_types = [0] * len(low_conf_boxes)
        
        prev_positions = {tid: track['positions'][-1][1] if track['positions'] else None 
                         for tid, track in self.tracks.items()}
        
        if len(self.tracks) > 0:
            track_ids = list(self.tracks.keys())
            
            for track_id in track_ids:
                if track_id in matched_ids:
                    continue
                track = self.tracks[track_id]
                best_match = None
                best_dist = float('inf')
                best_idx = -1
                
                for i, det in enumerate(high_conf_boxes):
                    if i in used_detections:
                        continue
                    dist = calculate_center_distance(track['box'], det)
                    if dist < best_dist and dist < self.max_distance:
                        best_dist = dist
                        best_match = det
                        best_idx = i
                
                if best_match is not None:
                    box = best_match
                    track['box'] = box
                    track['disappeared'] = 0
                    track['confidence'] = high_conf_scores[best_idx]
                    track['vehicle_type'] = high_conf_types[best_idx]
                    cx, cy = box[0] + box[2]//2, box[1] + box[3]//2
                    track['positions'].append((cx, cy, frame_id))
                    track['speed'] = self.calculate_adaptive_speed(track, frame_id, img_height)
                    self.detect_anomalies(track_id, track, prev_positions.get(track_id))
                    matched_ids.add(track_id)
                    used_detections.add(best_idx)
            
            for track_id in track_ids:
                if track_id in matched_ids:
                    continue
                track = self.tracks[track_id]
                best_match = None
                best_dist = float('inf')
                best_idx = -1
                
                for i, det in enumerate(low_conf_boxes):
                    if i in used_detections:
                        continue
                    dist = calculate_center_distance(track['box'], det)
                    if dist < best_dist and dist < self.max_distance * 0.7:
                        best_dist = dist
                        best_match = det
                        best_idx = i
                
                if best_match is not None:
                    box = best_match
                    track['box'] = box
                    track['disappeared'] = 0
                    track['confidence'] = low_conf_scores[best_idx]
                    track['vehicle_type'] = low_conf_types[best_idx]
                    cx, cy = box[0] + box[2]//2, box[1] + box[3]//2
                    track['positions'].append((cx, cy, frame_id))
                    track['speed'] = self.calculate_adaptive_speed(track, frame_id, img_height)
                    matched_ids.add(track_id)
                    used_detections.add(best_idx)
        
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_ids:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
        
        for i, (box, score, vtype) in enumerate(zip(high_conf_boxes, high_conf_scores, high_conf_types)):
            if i not in used_detections:
                self.register(box, score, frame_id, vtype)
        
        return self.tracks


class TrafficStatistics:
    """交通数据统计与报表生成"""
    def __init__(self):
        self.data = {
            'total_count': 0,
            'vehicle_counts': {0: 0, 1: 0, 2: 0},
            'speed_records': [],
            'avg_speed': 0,
            'max_speed': 0,
            'min_speed': 0,
            'congestion_index': 0,
            'events': [],
            'lane_data': {0: {'count': 0, 'avg_speed': 0}, 1: {'count': 0, 'avg_speed': 0}},
            'hourly_data': [],
            'warnings': []
        }
        self.start_time = datetime.now()
        self.csv_file = None
        self.csv_writer = None
    
    def start_csv_logging(self, filename='traffic_log.csv'):
        self.csv_file = open(filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['时间', '帧号', '车辆ID', '车型', '速度', '车道', '异常行为', 'X坐标', 'Y坐标'])
    
    def log_vehicle(self, track_id, vehicle_type, speed, lane, warnings, x, y, frame_id):
        if self.csv_writer:
            warning_str = ','.join([w[0] for w in warnings]) if warnings else '正常'
            vehicle_name = ['轿车', '货车', '公交'][vehicle_type]
            self.csv_writer.writerow([
                datetime.now().strftime('%H:%M:%S'),
                frame_id, track_id, vehicle_name,
                f'{speed:.1f}', lane, warning_str, x, y
            ])
    
    def update(self, tracks, frame_id):
        speeds = []
        for track_id, track in tracks.items():
            speed = track.get('speed', 0)
            if speed > 5:
                speeds.append(speed)
                self.data['speed_records'].append({
                    'frame': frame_id,
                    'speed': speed,
                    'vehicle_type': track.get('vehicle_type', 0)
                })
                self.log_vehicle(track_id, track.get('vehicle_type', 0), speed,
                               track.get('lane', 0), track.get('warnings', []),
                               track['box'][0], track['box'][1], frame_id)
        
        if speeds:
            self.data['avg_speed'] = np.mean(speeds)
            self.data['max_speed'] = max(speeds)
            self.data['min_speed'] = min(speeds)
            
            slow_count = sum(1 for s in speeds if s < 20)
            self.data['congestion_index'] = (slow_count / len(speeds)) * 100
        
        self.data['total_count'] = sum(self.data['vehicle_counts'].values())
        
        for warning in self.data['warnings'][-20:]:
            if frame_id - warning['frame'] < 300:
                continue
            self.data['warnings'].append({'frame': frame_id, 'warning': warning})
    
    def export_summary(self, filename='traffic_summary.json'):
        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_count': self.data['total_count'],
            'vehicle_counts': {name: count for name, count in 
                             zip(['轿车', '货车', '公交'],
                                self.data['vehicle_counts'].values())},
            'avg_speed': f"{self.data['avg_speed']:.1f}",
            'max_speed': f"{self.data['max_speed']:.1f}",
            'min_speed': f"{self.data['min_speed']:.1f}",
            'congestion_index': f"{self.data['congestion_index']:.1f}%"
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary
    
    def close(self):
        if self.csv_file:
            self.csv_file.close()


class TrafficLightController:
    """智能信号灯控制器"""
    def __init__(self):
        self.current_green_time = 30
        self.min_green_time = 15
        self.max_green_time = 60
        self.green_time = 30
        self.is_green = True
        self.phase = 0
        self.phase_names = ['北向南', '东向西', '南向北', '西向东']
    
    def update(self, vehicle_count, avg_speed):
        if vehicle_count > 15:
            self.green_time = min(self.green_time + 2, self.max_green_time)
        elif vehicle_count < 5:
            self.green_time = max(self.green_time - 2, self.min_green_time)
        else:
            adjustment = (vehicle_count - 10) * 0.5
            self.green_time = max(self.min_green_time, min(self.max_green_time, self.green_time + adjustment))
        
        return {
            'green_time': int(self.green_time),
            'phase': self.phase_names[self.phase],
            'is_green': self.is_green,
            'vehicle_count': vehicle_count
        }
    
    def switch_phase(self):
        self.phase = (self.phase + 1) % 4
        self.is_green = True


class WeatherEnhancer:
    """恶劣天气图像增强器 - 集成 Retinex 去雾 + CLAHE 光照矫正"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def clahe_enhance(self, img):
        """CLAHE 限制对比度自适应直方图均衡 - 处理强光、逆光、夜间太暗"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE 增强亮度通道
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def retinex_dehaze(self, img):
        """简化版 Retinex 去雾 - 处理雾天、雨天模糊"""
        # 转换为灰度图计算光照
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯模糊获取光照分量
        illumination = cv2.GaussianBlur(gray, (75, 75), 0)
        illumination = illumination / 255.0
        # Retinex 公式：R = log(I) - log(L)
        result = img / illumination[..., np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def white_balance(self, img):
        """简单白平衡处理"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    def enhance(self, frame, enable_retinex=True, enable_clahe=True):
        """组合增强：白平衡 → Retinex 去雾 → CLAHE 光照矫正"""
        result = frame.copy()
        
        # 1. 白平衡
        result = self.white_balance(result)
        
        # 2. Retinex 去雾（可选）
        if enable_retinex:
            result = self.retinex_dehaze(result)
        
        # 3. CLAHE 光照矫正（可选）
        if enable_clahe:
            result = self.clahe_enhance(result)
        
        return result
    
    def is_low_light(self, frame):
        """判断是否为低光照场景"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < 80


class NightEnhancer:
    """夜间/恶劣天气图像增强 - 兼容旧接口"""
    def __init__(self):
        self.weather_enhancer = WeatherEnhancer()
        self.contrast_factor = 1.5
        self.brightness_factor = 1.3
    
    def enhance(self, frame):
        # 使用新的天气增强器
        return self.weather_enhancer.enhance(frame)


def blend_layers(img1, img2, alpha=0.5, beta=0.5, gamma=0):
    """安全的图层融合工具函数，自动处理尺寸和通道不匹配
    
    Args:
        img1: 第一层图像
        img2: 第二层图像
        alpha: 第一层权重
        beta: 第二层权重
        gamma: 偏移量
    
    Returns:
        融合后的图像
    """
    if img1 is None or img2 is None:
        return img1 if img1 is not None else img2
    
    # 统一尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2 or w1 != w2:
        img2 = cv2.resize(img2, (w1, h1))
    
    # 统一通道数
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # 确保数据类型一致
    if img1.dtype != img2.dtype:
        img2 = img2.astype(img1.dtype)
    
    return cv2.addWeighted(img1, alpha, img2, beta, gamma)


class HeatmapGenerator:
    """交通热力图生成"""
    def __init__(self):
        self.density_map = None
        self.speed_map = None
        self.count_map = None
        self.width = 640
        self.height = 480
        self.decay_factor = 0.95
    
    def _ensure_size(self, width, height):
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            self.density_map = np.zeros((height, width), dtype=np.float32)
            self.speed_map = np.zeros((height, width), dtype=np.float32)
            self.count_map = np.zeros((height, width), dtype=np.float32)
    
    def update(self, tracks, width, height):
        self._ensure_size(width, height)
        
        if self.density_map is None:
            return None, None, None
        
        self.density_map *= self.decay_factor
        self.speed_map *= self.decay_factor
        self.count_map *= self.decay_factor
        
        for track_id, track in tracks.items():
            if track.get('speed', 0) < 5:
                continue
            cx = int(track['box'][0] + track['box'][2] // 2)
            cy = int(track['box'][1] + track['box'][3] // 2)
            
            if 0 <= cx < self.width and 0 <= cy < self.height:
                cv2.circle(self.density_map, (cx, cy), 30, 1.0, -1)
                speed = track.get('speed', 0)
                cv2.circle(self.speed_map, (cx, cy), 30, speed, -1)
                cv2.circle(self.count_map, (cx, cy), 30, 1.0, -1)
        
        density_normalized = cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX)
        density_colored = cv2.applyColorMap(density_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        return density_colored, self.density_map, self.speed_map
    
    def generate_time_heatmap(self):
        if np.sum(self.count_map) < 1:
            return None
        avg_speed_map = np.divide(self.speed_map, self.count_map, where=self.count_map > 0)
        avg_speed_normalized = cv2.normalize(avg_speed_map, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(avg_speed_normalized.astype(np.uint8), cv2.COLORMAP_JET)


class CollisionWarning:
    """车距监测与追尾预警"""
    def __init__(self):
        self.min_safe_distance = 50
        self.warning_distance = 80
        self.alert_distance = 120
    
    def check_distance(self, track1, track2):
        box1, box2 = track1['box'], track2['box']
        
        x1, y1 = box1[0] + box1[2]//2, box1[1] + box1[3]//2
        x2, y2 = box2[0] + box2[2]//2, box2[1] + box2[3]//2
        
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        speed1, speed2 = track1.get('speed', 0), track2.get('speed', 0)
        relative_speed = abs(speed1 - speed2)
        
        safe_distance = self.min_safe_distance + relative_speed * 0.5
        
        if distance < safe_distance * 0.5:
            return 'RED', distance
        elif distance < safe_distance:
            return 'YELLOW', distance
        elif distance < self.alert_distance:
            return 'WARNING', distance
        return 'SAFE', distance


def cv2_puttext_cn(img, text, org, font_size=20, color=(255, 255, 255)):
    """绘制中文字符 - 增强版，支持更多字体路径"""
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/STHeiti Medium.ttc",
        "C:/Windows/Fonts/STSong.ttf",
        "C:/Windows/Fonts/STKaiti.ttf",
        "C:/Windows/Fonts/KaiTi.ttf",
        "C:/Windows/Fonts/SimSun.ttf",
        "C:/Windows/Fonts/YaHei.Consolas.1.12.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/Library/Fonts/SimHei.ttf",
        "/Library/Fonts/Microsoft YaHei.ttf"
    ]
    
    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
                break
            except Exception as e:
                continue
    
    if font is None:
        print("警告：未找到中文字体，使用默认字体")
        font = ImageFont.load_default()
    
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(org, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"中文绘制错误: {e}")
        return img


class TrafficMonitor:
    """智能交通视觉监测系统 - 完整版"""
    def __init__(self, speed_limit=60, fps=24, model_path=None):
        self.fps = fps
        self.speed_limit = speed_limit
        self.model_path = model_path
        self.model = None
        self.use_yolo = False
        
        if YOLO_AVAILABLE and model_path:
            try:
                self.model = YOLO(model_path)
                self.use_yolo = True
                print(f"成功加载YOLO模型: {model_path}")
            except Exception as e:
                print(f"加载YOLO模型失败: {e}")
                self.use_yolo = False
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=800, varThreshold=40, detectShadows=False
        ) if not self.use_yolo else None
        
        self.tracker = ByteTracker(max_disappeared=8, max_distance=80, fps=fps)
        self.classifier = VehicleClassifier()
        self.statistics = TrafficStatistics()
        self.traffic_light = TrafficLightController()
        self.night_enhancer = NightEnhancer()
        self.heatmap_gen = HeatmapGenerator()
        self.collision_checker = CollisionWarning()
        
        # 新增：轻量车牌识别器
        self.lpr = LightweightLPR()
        # 新增：恶劣天气图像增强器
        self.weather_enhancer = WeatherEnhancer()
        
        # ========== 新增三大核心模块 ==========
        # 1. 驾驶行为画像模块
        self.behavior_detector = BehaviorLSTM()
        self.driver_profile = DriverProfile()
        
        # 2. 会车/跟车风险评估模块
        self.risk_analyzer = VehicleInteractionAnalyzer()
        
        # 3. 车道流量均衡分析模块
        self.lane_analyzer = LaneFlowAnalyzer(num_lanes=4)
        # =====================================
        
        self.frame_count = 0
        self.counted_ids = set()
        self.track_valid_frames = {}
        self.track_prev_pos = {}  # 记录车辆上一帧位置
        
        self.show_heatmap = False
        self.show_lane = True
        self.show_statistics = True
        self.show_warnings = True
        
        # 图像增强开关
        self.enable_enhancement = True
        self.enable_retinex = True
        self.enable_clahe = True
        
        # 新增开关控制
        self.show_trajectory = True  # 轨迹线显示
        self.enable_risk_warning = True  # 风险预警开关
        self.heatmap_opacity = 0.7  # 热力图层透明度
    
    def reset(self):
        self.frame_count = 0
        self.counted_ids = set()
        self.track_valid_frames = {}
        self.track_prev_pos = {}
        self.tracker = ByteTracker(max_disappeared=8, max_distance=80, fps=self.fps)
        self.statistics = TrafficStatistics()
        if not self.use_yolo:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=800, varThreshold=40, detectShadows=False
            )
    
    def set_speed_limit(self, limit):
        self.speed_limit = limit
    
    def detect_with_yolo(self, frame):
        """YOLO检测 - 严格只保留三类车型"""
        if not self.model:
            return [], [], []
        
        height, width = frame.shape[:2]
        
        # 使用CLAHE做光照均衡
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # YOLO检测，只启用car/bus/truck三类
        results = self.model(
            frame_enhanced, 
            verbose=False, 
            conf=0.35,  # 置信度阈值
            iou=0.45,   # IOU阈值
            classes=[2, 5, 7]  # 严格只保留三类：2=car,5=bus,7=truck
        )
        
        boxes, scores, classes = [], [], []
        
        # ROI区域定义（只保留车道内）
        roi_left, roi_right = int(width * 0.10), int(width * 0.90)
        roi_top, roi_bottom = int(height * 0.20), height
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls.item())
                # 严格检查类别
                if cls not in [2, 5, 7]:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                conf = float(box.conf.item())
                
                # ROI过滤：只保留车道内
                cx, cy = x + w//2, y + h//2
                if not (roi_left < cx < roi_right and roi_top < cy < roi_bottom):
                    continue
                
                # 尺寸过滤：过滤太小的远景框
                if w < 30 or h < 30:
                    continue
                
                # 公交车特殊过滤：防止把站牌识别成公交车
                if cls == 5:  # bus
                    # 公交车置信度要求更高
                    if conf < 0.5:
                        continue
                    # 公交车宽高比通常大于1.2（长条形），站牌通常更高或接近正方形
                    aspect_ratio = w / h if h > 0 else 0
                    # 公交车宽度通常明显大于高度，站牌可能更高或接近正方形
                    if aspect_ratio < 1.0:
                        continue
                    # 公交车面积通常较大
                    if w * h < 8000:
                        continue
                
                # 货车特殊过滤
                if cls == 7:  # truck
                    if conf < 0.45:
                        continue
                
                boxes.append((x, y, w, h))
                scores.append(conf)
                classes.append(cls)
        
        return boxes, scores, classes
    
    def detect_with_bgsub(self, frame):
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.005)
        _, fg_mask = cv2.threshold(fg_mask, 185, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=3)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes, scores, classes = [], [], []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300 or area > 25000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 40 or h < 40:
                continue
            boxes.append((x, y, w, h))
            scores.append(0.7)
            classes.append(2)
        
        return boxes, scores, classes
    
    def process_frame(self, frame):
        height, width = frame.shape[:2]
        self.frame_count += 1
        
        # 恶劣天气图像增强预处理
        if self.enable_enhancement:
            frame_enhanced = self.weather_enhancer.enhance(
                frame, 
                enable_retinex=self.enable_retinex, 
                enable_clahe=self.enable_clahe
            )
        else:
            frame_enhanced = frame.copy()
        
        if self.use_yolo:
            boxes, scores, classes = self.detect_with_yolo(frame_enhanced)
        else:
            boxes, scores, classes = self.detect_with_bgsub(frame_enhanced)
        
        # 车型分类
        vehicle_types = []
        for box, cls in zip(boxes, classes):
            vtype, conf = self.classifier.classify(box, cls)
            vehicle_types.append(vtype)
        
        # 高/低置信度划分
        high_conf_idx = [i for i, s in enumerate(scores) if s >= 0.4]
        low_conf_idx = [i for i, s in enumerate(scores) if s < 0.4]
        
        high_conf_boxes = [boxes[i] for i in high_conf_idx]
        high_conf_scores = [scores[i] for i in high_conf_idx]
        high_conf_types = [vehicle_types[i] for i in high_conf_idx]
        
        low_conf_boxes = [boxes[i] for i in low_conf_idx]
        low_conf_scores = [scores[i] for i in low_conf_idx]
        low_conf_types = [vehicle_types[i] for i in low_conf_idx]
        
        # ByteTrack跟踪
        tracks = self.tracker.update(
            high_conf_boxes, high_conf_scores,
            low_conf_boxes, low_conf_scores,
            self.frame_count, height,
            high_conf_types, low_conf_types
        )
        
        # 为跟踪结果设置车型（使用历史平滑）
        active_track_ids = set()
        for track_id, track in tracks.items():
            active_track_ids.add(track_id)
            current_type = track.get('vehicle_type', 0)
            stable_type = self.classifier.get_stable_type(track_id, current_type)
            track['vehicle_type'] = stable_type
        
        # 清理分类器历史
        self.classifier.cleanup(active_track_ids)
        
<<<<<<< HEAD
        # 车流量统计 - 统计线移到更靠下的位置，让车辆有足够时间被稳定跟踪
        count_line_y = int(height * 0.75)
=======
        # 车牌识别：只对稳定跟踪的车辆进行识别
        for track_id, track in tracks.items():
            # 只在车辆稳定跟踪后（5帧以上）且尚未识别到车牌时进行识别
            if self.track_valid_frames.get(track_id, 0) >= 5 and not self.lpr.get_plate(track_id):
                box = track['box']
                x, y, w, h = box
                # 裁剪车辆区域（稍微扩展一点）
                x1, y1 = max(0, x - 5), max(0, y - 5)
                x2, y2 = min(width, x + w + 5), min(height, y + h + 5)
                car_crop = frame[y1:y2, x1:x2]
                
                # 只有车辆区域足够大时才进行识别
                if car_crop.shape[0] > 50 and car_crop.shape[1] > 50:
                    plate_number = self.lpr.recognize(car_crop)
                    if plate_number:
                        self.lpr.set_plate(track_id, plate_number)
        
        # 清理LPR历史
        self.lpr.cleanup(active_track_ids)
        
        # 车流量统计
        count_line_y = int(height * 0.58)
>>>>>>> d869d291353cbd5a4f7a9769ee8a43202c70f7c9
        for track_id, track in tracks.items():
            if track_id not in self.track_valid_frames:
                self.track_valid_frames[track_id] = 0
            self.track_valid_frames[track_id] += 1
            
            box = track['box']
            cy = box[1] + box[3] // 2
            
            # 只有真正穿过检测线才计数
            # 降低稳定帧要求，确保公交车等大型车辆能被正确计数
            if self.track_valid_frames[track_id] >= 2 and track_id not in self.counted_ids:
                if track_id in self.track_prev_pos:
                    prev_cy = self.track_prev_pos[track_id]
                    # 允许一定的抖动范围，防止漏检
                    if prev_cy <= count_line_y + 5 and cy > count_line_y - 5:
                        self.counted_ids.add(track_id)
                        self.statistics.data['total_count'] += 1
                        vtype = track.get('vehicle_type', 0)
                        if vtype in self.statistics.data['vehicle_counts']:
                            self.statistics.data['vehicle_counts'][vtype] += 1
            
            self.track_prev_pos[track_id] = cy
        
        # 清理消失车辆历史
        for track_id in list(self.track_prev_pos.keys()):
            if track_id not in active_track_ids:
                del self.track_prev_pos[track_id]
        
        self.statistics.update(tracks, self.frame_count)
        
        # ========== 新增模块处理 ==========
        # 1. 驾驶行为画像更新
        for track_id, track in tracks.items():
            self.driver_profile.update_profile(track_id, track, self.frame_count, self.fps)
        
        # 2. 会车/跟车风险评估
        conflicts = self.risk_analyzer.analyze_frame(tracks, self.frame_count, self.fps, frame)
        
        # 3. 车道流量均衡分析
        self.lane_analyzer.update(tracks, self.frame_count, self.fps)
        # =================================
        
        # 智能信号灯
        vehicle_count = sum(1 for t in tracks.values() if t.get('speed', 0) > 5)
        avg_speed = np.mean([t.get('speed', 0) for t in tracks.values() if t.get('speed', 0) > 5]) if vehicle_count > 0 else 0
        light_info = self.traffic_light.update(vehicle_count, avg_speed)
        
        result = frame.copy()
        
        # 热力图叠加
        if self.show_heatmap:
            heatmap_colored, _, _ = self.heatmap_gen.update(tracks, width, height)
            if heatmap_colored is not None:
                result = blend_layers(result, heatmap_colored, self.heatmap_opacity, 1 - self.heatmap_opacity, 0)
        
        # 绘制虚拟车道线（已禁用，用户要求移除）
        # if self.show_lane:
        #     for lane in self.lane_analyzer.lane_manager.lanes:
        #         cv2.line(result, (lane['x_start'], 0), (lane['x_start'], height), (0, 255, 255), 1, cv2.LINE_AA)
        
        # 绘制车辆轨迹线
        if self.show_trajectory:
            for track_id, track in tracks.items():
                trajectory = list(track.get('trajectory', []))
                if len(trajectory) >= 2:
                    for i in range(1, len(trajectory)):
                        cv2.line(result, trajectory[i-1], trajectory[i], (200, 200, 200), 1, cv2.LINE_AA)
        
        # 绘制风险标记
        if self.enable_risk_warning:
            for conflict in conflicts:
                if conflict['risk_level'] in ['medium', 'high']:
                    pos = conflict['position']
                    color = (0, 255, 255) if conflict['risk_level'] == 'medium' else (0, 0, 255)
                    cv2.circle(result, pos, 20, color, 2)
                    cv2.putText(result, f"TTC:{conflict['ttc']}s", (pos[0]+25, pos[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制检测框和标注
        for track_id, track in tracks.items():
            if self.track_valid_frames.get(track_id, 0) < 3:
                continue
            
            box = track['box']
            x, y, w, h = box
            x_exp, y_exp = int(max(0, x - 5)), int(max(0, y - 5))
            w_exp = int(min(width - x_exp, w + 10))
            h_exp = int(min(height - y_exp, h + 10))
            
            vtype = track.get('vehicle_type', 0)
            type_info = self.classifier.VEHICLE_TYPES.get(vtype, self.classifier.VEHICLE_TYPES[0])
            color = type_info['color']
            
            valid_frames = self.track_valid_frames.get(track_id, 0)
            
            speed = track.get('speed', 0)
            is_over_speed = speed > self.speed_limit and speed > 25
            
            # 超速红色框，正常车辆车型颜色框
            if is_over_speed:
                cv2.rectangle(result, (x_exp, y_exp), (x_exp + w_exp, y_exp + h_exp), (0, 0, 255), 3)
            elif valid_frames >= 5:
                cv2.rectangle(result, (x_exp, y_exp), (x_exp + w_exp, y_exp + h_exp), color, 2)
            else:
                cv2.rectangle(result, (x_exp, y_exp), (x_exp + w_exp, y_exp + h_exp), color, 1)
            
            # 绘制车型标签
            if valid_frames >= 5:
                label = type_info['name']
                # 标签背景
                label_width = 75
                cv2.rectangle(result, (x_exp, max(0, y_exp - 22)), (x_exp + label_width, max(0, y_exp)), (0, 0, 0), -1)
                # 中文标签
                result = cv2_puttext_cn(result, label, (x_exp + 3, max(5, y_exp - 7)), font_size=12, color=color)
                
                # 速度标签
                if speed >= 20:
                    speed_label = "{}km/h".format(int(speed))
                    cv2.putText(result, speed_label, (x_exp + 3, y_exp + h_exp + 18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
                
                # 车牌显示
                plate_number = self.lpr.get_plate(track_id)
                if plate_number:
                    # 车牌背景
                    plate_bg_width = max(100, len(plate_number) * 18)
                    cv2.rectangle(result, (x_exp, max(0, y_exp - 42)), 
                                  (x_exp + plate_bg_width, max(0, y_exp - 22)), (0, 0, 0), -1)
                    # 车牌文字
                    result = cv2_puttext_cn(result, plate_number, 
                                            (x_exp + 3, max(5, y_exp - 27)), 
                                            font_size=12, color=(255, 255, 0))
            
            # 警告信息
            warnings = track.get('warnings', [])
            if warnings and self.show_warnings:
                for i, (warning_text, warning_color) in enumerate(warnings[:2]):
                    result = cv2_puttext_cn(result, warning_text, (x_exp, y_exp + h_exp + 36 + i*18), font_size=12, color=warning_color)
        

        return result, {
            'count': self.statistics.data['total_count'],
            'frame': self.frame_count,
            'avg_speed': self.statistics.data['avg_speed'],
            'congestion': self.statistics.data['congestion_index'],
            'light_info': light_info
        }
    
    def get_statistics(self):
        return {
            'total_count': self.statistics.data['total_count'],
            'vehicle_counts': self.statistics.data['vehicle_counts'],
            'avg_speed': self.statistics.data['avg_speed'],
            'max_speed': self.statistics.data['max_speed'],
            'congestion_index': self.statistics.data['congestion_index'],
            'warnings': self.statistics.data['warnings'][-10:]
        }
    
    def get_driver_profiles(self):
        """获取所有车辆的驾驶行为画像"""
        return self.driver_profile.profiles
    
    def get_driver_profile(self, track_id):
        """获取指定车辆的驾驶行为画像"""
        return self.driver_profile.get_profile(track_id)
    
    def get_behavior_timeline(self, track_id):
        """获取指定车辆的行为时间轴"""
        return self.driver_profile.get_timeline(track_id)
    
    def get_risk_summary(self):
        """获取风险评估汇总"""
        return self.risk_analyzer.get_risk_summary()
    
    def get_high_risk_events(self):
        """获取高风险事件列表"""
        return self.risk_analyzer.ttc_evaluator.get_high_risk_events()
    
    def get_lane_statistics(self):
        """获取车道流量统计"""
        return self.lane_analyzer.get_lane_statistics()
    
    def get_lane_predictions(self):
        """获取车道流量预测"""
        return self.lane_analyzer.predict_future_flow()
    
    def get_optimization_suggestions(self):
        """获取车道优化建议"""
        return self.lane_analyzer.generate_optimization_suggestions()
    
    def export_data(self, csv_file='traffic_log.csv', summary_file='traffic_summary.json'):
        # 导出基础统计
        self.statistics.export_summary(summary_file)
        
        # 导出驾驶行为画像数据
        behavior_data = {
            'profiles': self.driver_profile.profiles,
            'timelines': self.driver_profile.behavior_timeline
        }
        with open('driver_profiles.json', 'w', encoding='utf-8') as f:
            json.dump(behavior_data, f, ensure_ascii=False, indent=2)
        
        # 导出风险事件数据
        risk_data = {
            'summary': self.risk_analyzer.get_risk_summary(),
            'events': self.risk_analyzer.ttc_evaluator.conflict_history,
            'heatmap': dict(self.risk_analyzer.ttc_evaluator.heatmap_data)
        }
        with open('risk_events.json', 'w', encoding='utf-8') as f:
            json.dump(risk_data, f, ensure_ascii=False, indent=2)
        
        # 导出车道分析数据
        lane_data = {
            'statistics': self.lane_analyzer.get_lane_statistics(),
            'predictions': self.lane_analyzer.predict_future_flow(),
            'suggestions': self.lane_analyzer.generate_optimization_suggestions()
        }
        with open('lane_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(lane_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已导出: {summary_file}, driver_profiles.json, risk_events.json, lane_analysis.json")
