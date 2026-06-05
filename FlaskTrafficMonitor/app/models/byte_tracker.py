from collections import deque
import numpy as np

class ByteTracker:
    """ByteTrack风格多目标跟踪器 - 增强版"""
    
    def __init__(self, max_disappeared=8, max_distance=80, fps=24):
        self.next_id = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.fps = fps
    
    def register(self, box, confidence, frame_id, vehicle_type=0):
        """注册新目标"""
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
        """计算自适应车速"""
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
        cx, cy = track['box'][0] + track['box'][2]//2, track['box'][1] + track['box'][3]//2
        
        # 逆行检测
        if prev_y is not None:
            if cy < prev_y - 20:
                track['reverse_count'] += 1
                if track['reverse_count'] > 10:
                    warnings.append(('逆行', (0, 0, 255)))
            else:
                track['reverse_count'] = 0
        
        # 车道变换检测
        x_changes = [pos[0] for pos in track['trajectory'][-5:]]
        if len(x_changes) >= 3:
            lane_change_threshold = 30
            if abs(sum(x_changes[:3])) > lane_change_threshold:
                track['lane_change_count'] += 1
                if track['lane_change_count'] > 2:
                    warnings.append(('异常变道', (255, 200, 0)))
        
        # 滞留检测
        if track['speed'] < 5 and len(track['positions']) > 50:
            track['stopped_time'] += 1
            if track['stopped_time'] > 120:
                warnings.append(('滞留/停车', (128, 0, 128)))
        else:
            track['stopped_time'] = 0
        
        track['warnings'] = warnings
        return warnings
    
    def update(self, high_conf_boxes, high_conf_scores, low_conf_boxes, low_conf_scores, frame_id, img_height=480, high_conf_types=None, low_conf_types=None):
        """更新跟踪状态"""
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
                    dist = self._calculate_distance(track['box'], det)
                    if dist < best_dist and dist < self.max_distance:
                        best_dist = dist
                        best_match = det
                        best_idx = i
                
                if best_match is not None:
                    track['box'] = best_match
                    track['disappeared'] = 0
                    track['confidence'] = high_conf_scores[best_idx]
                    track['vehicle_type'] = high_conf_types[best_idx]
                    cx, cy = best_match[0] + best_match[2]//2, best_match[1] + best_match[3]//2
                    track['positions'].append((cx, cy, frame_id))
                    track['trajectory'].append((cx, cy))
                    track['speed'] = self.calculate_adaptive_speed(track, frame_id, img_height)
                    self.detect_anomalies(track_id, track, prev_positions.get(track_id))
                    matched_ids.add(track_id)
                    used_detections.add(best_idx)
            
            # 处理低置信度检测
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
                    dist = self._calculate_distance(track['box'], det)
                    if dist < best_dist and dist < self.max_distance:
                        best_dist = dist
                        best_match = det
                        best_idx = i
                
                if best_match is not None:
                    track['box'] = best_match
                    track['disappeared'] = 0
                    track['confidence'] = low_conf_scores[best_idx]
                    track['vehicle_type'] = low_conf_types[best_idx]
                    cx, cy = best_match[0] + best_match[2]//2, best_match[1] + best_match[3]//2
                    track['positions'].append((cx, cy, frame_id))
                    track['trajectory'].append((cx, cy))
                    track['speed'] = self.calculate_adaptive_speed(track, frame_id, img_height)
                    matched_ids.add(track_id)
                    used_detections.add(best_idx)
        
        # 标记未匹配的跟踪
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_ids:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
        
        # 添加新检测
        for i, (box, score, vtype) in enumerate(zip(high_conf_boxes, high_conf_scores, high_conf_types)):
            if i not in used_detections:
                self.register(box, score, frame_id, vtype)
        
        return self.tracks
    
    def _calculate_distance(self, box1, box2):
        """计算两个检测框中心点之间的距离"""
        cx1 = box1[0] + box1[2] // 2
        cy1 = box1[1] + box1[3] // 2
        cx2 = box2[0] + box2[2] // 2
        cy2 = box2[1] + box2[3] // 2
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
