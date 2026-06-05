# -*- coding: utf-8 -*-
"""
会车/跟车风险评估模块
基于TTC时间碰撞阈值模型，实现风险评估和预警
"""

import numpy as np
import os
from datetime import datetime


class TTCRiskEvaluator:
    """TTC时间碰撞阈值风险评估器"""
    
    RISK_LEVELS = {
        'low': {'label': '低风险', 'color': (0, 255, 0), 'threshold': 3.0},
        'medium': {'label': '中风险', 'color': (0, 255, 255), 'threshold': 1.5},
        'high': {'label': '高风险', 'color': (0, 0, 255), 'threshold': 0.0}
    }
    
    def __init__(self):
        self.risk_events = []
        self.conflict_history = []
        self.heatmap_data = {}
        self.screenshot_dir = 'risk_screenshots'
        os.makedirs(self.screenshot_dir, exist_ok=True)
    
    def calculate_ttc(self, track1, track2):
        """计算两车之间的TTC（时间碰撞阈值）"""
        box1 = track1['box']
        box2 = track2['box']
        
        # 计算中心点
        cx1, cy1 = box1[0] + box1[2]//2, box1[1] + box1[3]//2
        cx2, cy2 = box2[0] + box2[2]//2, box2[1] + box2[3]//2
        
        # 计算相对距离
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        
        # 计算相对速度（假设同向行驶，只考虑纵向速度）
        speed1 = track1.get('speed', 0)
        speed2 = track2.get('speed', 0)
        relative_speed = speed1 - speed2  # 前车速度 - 后车速度
        
        # 避免除零
        if abs(relative_speed) < 0.1:
            return float('inf')
        
        # 计算TTC（时间碰撞阈值）
        # 假设速度单位是 km/h，需要转换为像素/秒
        pixel_speed1 = speed1 * 0.4  # 近似转换
        pixel_speed2 = speed2 * 0.4
        
        # TTC = 距离 / 相对速度
        if pixel_speed1 > pixel_speed2 and distance > 0:
            ttc = distance / (pixel_speed1 - pixel_speed2)
        else:
            ttc = float('inf')
        
        return ttc
    
    def evaluate_risk(self, track1, track2):
        """评估两车之间的风险等级"""
        ttc = self.calculate_ttc(track1, track2)
        
        if ttc <= self.RISK_LEVELS['high']['threshold']:
            return 'high', ttc
        elif ttc <= self.RISK_LEVELS['medium']['threshold']:
            return 'medium', ttc
        else:
            return 'low', ttc
    
    def detect_conflicts(self, tracks, frame_id, fps=24):
        """检测所有车辆之间的冲突"""
        track_ids = list(tracks.keys())
        conflicts = []
        
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track_id1 = track_ids[i]
                track_id2 = track_ids[j]
                
                track1 = tracks[track_id1]
                track2 = tracks[track_id2]
                
                # 只考虑同向行驶的车辆
                box1 = track1['box']
                box2 = track2['box']
                cy1, cy2 = box1[1] + box1[3]//2, box2[1] + box2[3]//2
                
                # 检查是否在同一车道附近
                cx1, cx2 = box1[0] + box1[2]//2, box2[0] + box2[2]//2
                if abs(cx1 - cx2) > 150:
                    continue
                
                risk_level, ttc = self.evaluate_risk(track1, track2)
                
                if risk_level != 'low':
                    conflicts.append({
                        'track_id1': track_id1,
                        'track_id2': track_id2,
                        'risk_level': risk_level,
                        'ttc': round(ttc, 2),
                        'frame_id': frame_id,
                        'timestamp': frame_id / fps,
                        'position': ((cx1 + cx2) // 2, (cy1 + cy2) // 2)
                    })
                    
                    # 更新热力图数据
                    grid_x = (cx1 + cx2) // 100
                    grid_y = (cy1 + cy2) // 100
                    key = (grid_x, grid_y)
                    self.heatmap_data[key] = self.heatmap_data.get(key, 0) + 1
        
        return conflicts
    
    def record_conflict(self, conflict_info, frame=None):
        """记录冲突事件并保存截图"""
        self.conflict_history.append(conflict_info)
        
        # 保存截图
        if frame is not None and conflict_info['risk_level'] != 'low':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conflict_{timestamp}_{conflict_info['risk_level']}.jpg"
            filepath = os.path.join(self.screenshot_dir, filename)
            cv2.imwrite(filepath, frame)
            conflict_info['screenshot'] = filepath
    
    def get_high_risk_events(self, limit=10):
        """获取高风险事件列表"""
        high_risk = [c for c in self.conflict_history if c['risk_level'] in ['medium', 'high']]
        return sorted(high_risk, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_heatmap_data(self):
        """获取冲突热力图数据"""
        return self.heatmap_data
    
    def clear_history(self):
        """清空历史数据"""
        self.conflict_history = []
        self.heatmap_data = {}


class VehicleInteractionAnalyzer:
    """车辆交互分析器"""
    
    def __init__(self):
        self.ttc_evaluator = TTCRiskEvaluator()
        self.active_conflicts = {}
    
    def analyze_frame(self, tracks, frame_id, fps=24, frame=None):
        """分析当前帧的车辆交互"""
        conflicts = self.ttc_evaluator.detect_conflicts(tracks, frame_id, fps)
        
        # 记录冲突
        for conflict in conflicts:
            conflict_key = tuple(sorted([conflict['track_id1'], conflict['track_id2']]))
            self.active_conflicts[conflict_key] = conflict
            
            # 如果是中高风险，记录并保存截图
            if conflict['risk_level'] != 'low':
                self.ttc_evaluator.record_conflict(conflict, frame)
        
        # 清理已解决的冲突
        current_keys = set(tuple(sorted([c['track_id1'], c['track_id2']])) for c in conflicts)
        for key in list(self.active_conflicts.keys()):
            if key not in current_keys:
                del self.active_conflicts[key]
        
        return conflicts
    
    def get_active_conflicts(self):
        """获取当前活跃的冲突"""
        return list(self.active_conflicts.values())
    
    def get_risk_summary(self):
        """获取风险汇总"""
        high_risk = self.ttc_evaluator.get_high_risk_events()
        return {
            'total_conflicts': len(self.ttc_evaluator.conflict_history),
            'active_conflicts': len(self.active_conflicts),
            'high_risk_count': sum(1 for c in high_risk if c['risk_level'] == 'high'),
            'medium_risk_count': sum(1 for c in high_risk if c['risk_level'] == 'medium'),
            'recent_events': high_risk[:5]
        }
