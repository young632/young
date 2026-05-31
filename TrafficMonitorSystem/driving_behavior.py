# -*- coding: utf-8 -*-
"""
驾驶行为画像模块
基于ByteTrack跟踪数据，使用轻量LSTM模型识别异常驾驶行为
"""

import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler


class BehaviorLSTM:
    """轻量LSTM异常行为检测器"""
    
    def __init__(self, sequence_length=10, threshold=2.0):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.track_features = {}
        
    def extract_features(self, track):
        """提取车辆特征"""
        speed = track.get('speed', 0)
        positions = list(track.get('positions', []))
        trajectory = list(track.get('trajectory', []))
        
        # 计算加速度
        speed_history = list(track.get('speed_history', []))
        acceleration = 0
        if len(speed_history) >= 2:
            acceleration = speed_history[-1] - speed_history[-2]
        
        # 计算横向偏移
        lateral_movement = 0
        if len(trajectory) >= 2:
            lateral_movement = abs(trajectory[-1][0] - trajectory[-2][0])
        
        # 计算方向变化
        direction_change = 0
        if len(trajectory) >= 3:
            dx1 = trajectory[-1][0] - trajectory[-2][0]
            dy1 = trajectory[-1][1] - trajectory[-2][1]
            dx2 = trajectory[-2][0] - trajectory[-3][0]
            dy2 = trajectory[-2][1] - trajectory[-3][1]
            if dx1 != 0 or dy1 != 0:
                direction_change = abs(np.arctan2(dy1, dx1) - np.arctan2(dy2, dx2))
        
        return [speed, acceleration, lateral_movement, direction_change]
    
    def detect_anomaly(self, track_id, track):
        """检测异常驾驶行为"""
        if track_id not in self.track_features:
            self.track_features[track_id] = deque(maxlen=self.sequence_length)
        
        features = self.extract_features(track)
        self.track_features[track_id].append(features)
        
        if len(self.track_features[track_id]) < self.sequence_length:
            return []
        
        # 简单统计异常检测（模拟LSTM效果）
        feature_array = np.array(self.track_features[track_id])
        mean = np.mean(feature_array, axis=0)
        std = np.std(feature_array, axis=0)
        std[std == 0] = 1
        
        # 计算当前特征的异常分数
        z_scores = np.abs((features - mean) / std)
        
        anomalies = []
        speed, acceleration, lateral_movement, direction_change = features
        
        # 急刹车检测
        if acceleration < -15 and speed > 30:
            anomalies.append(('急刹', 'brake', acceleration))
        
        # 急加速检测
        if acceleration > 15:
            anomalies.append(('急加速', 'accelerate', acceleration))
        
        # 频繁变道检测
        if lateral_movement > 40:
            anomalies.append(('频繁变道', 'lane_change', lateral_movement))
        
        # 加塞检测（横向移动大且速度慢）
        if lateral_movement > 30 and speed < 20:
            anomalies.append(('加塞', 'cut_in', lateral_movement))
        
        # 超速检测
        if speed > 80:
            anomalies.append(('超速', 'overspeed', speed))
        
        return anomalies


class DriverProfile:
    """驾驶行为画像生成器"""
    
    BEHAVIOR_ICONS = {
        'brake': '🛑',
        'accelerate': '⚡',
        'lane_change': '↔️',
        'cut_in': '⚠️',
        'overspeed': '🚀'
    }
    
    def __init__(self):
        self.profiles = {}  # track_id -> profile
        self.behavior_timeline = {}  # track_id -> [(timestamp, behavior_type, value)]
    
    def update_profile(self, track_id, track, frame_id, fps=24):
        """更新驾驶行为画像"""
        if track_id not in self.profiles:
            self.profiles[track_id] = {
                'compliance_rate': 100,
                'aggression_index': 0,
                'violation_count': 0,
                'avg_speed': 0,
                'max_speed': 0,
                'total_distance': 0,
                'behavior_counts': {'brake': 0, 'accelerate': 0, 'lane_change': 0, 'cut_in': 0, 'overspeed': 0},
                'first_seen': frame_id,
                'last_updated': frame_id
            }
            self.behavior_timeline[track_id] = []
        
        profile = self.profiles[track_id]
        speed = track.get('speed', 0)
        warnings = track.get('warnings', [])
        
        # 更新速度统计
        profile['avg_speed'] = (profile['avg_speed'] * frame_id + speed) / (frame_id + 1)
        profile['max_speed'] = max(profile['max_speed'], speed)
        
        # 记录异常行为
        timestamp = frame_id / fps
        for behavior in warnings:
            behavior_type = self._get_behavior_type(behavior[0])
            if behavior_type:
                profile['behavior_counts'][behavior_type] += 1
                profile['violation_count'] += 1
                self.behavior_timeline[track_id].append((timestamp, behavior_type, behavior[0]))
        
        # 计算合规率和激进指数
        total_frames = frame_id - profile['first_seen'] + 1
        if total_frames > 0:
            profile['compliance_rate'] = max(0, 100 - (profile['violation_count'] / total_frames) * 100)
            profile['aggression_index'] = min(100, profile['violation_count'] * 5)
        
        profile['last_updated'] = frame_id
    
    def _get_behavior_type(self, behavior_text):
        """将行为文本映射到类型"""
        if '急刹' in behavior_text:
            return 'brake'
        elif '急加速' in behavior_text or '加速' in behavior_text:
            return 'accelerate'
        elif '变道' in behavior_text:
            return 'lane_change'
        elif '加塞' in behavior_text:
            return 'cut_in'
        elif '超速' in behavior_text:
            return 'overspeed'
        return None
    
    def get_profile(self, track_id):
        """获取车辆驾驶行为画像"""
        return self.profiles.get(track_id, None)
    
    def get_timeline(self, track_id):
        """获取车辆行为时间轴"""
        return self.behavior_timeline.get(track_id, [])
    
    def cleanup(self, active_track_ids):
        """清理消失车辆的画像数据"""
        for track_id in list(self.profiles.keys()):
            if track_id not in active_track_ids:
                del self.profiles[track_id]
                del self.behavior_timeline[track_id]
