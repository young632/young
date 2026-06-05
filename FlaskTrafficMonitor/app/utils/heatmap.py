import cv2
import numpy as np
from collections import defaultdict

class HeatmapGenerator:
    """热力图生成器"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.vehicle_tracks = defaultdict(list)  # 按车型分类的轨迹
        self.max_value = 100.0
    
    def add_detection(self, track_id, x, y, w, h, vehicle_type=0):
        """添加检测结果到热力图"""
        cx, cy = x + w // 2, y + h // 2
        
        # 添加到轨迹
        self.vehicle_tracks[track_id].append((cx, cy))
        if len(self.vehicle_tracks[track_id]) > 50:
            self.vehicle_tracks[track_id].pop(0)
        
        # 更新热力图 - 使用高斯核
        radius = min(w, h) // 2 + 20
        if radius < 30:
            radius = 30
        
        y_min = max(0, cy - radius)
        y_max = min(self.height, cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(self.width, cx + radius)
        
        for y_ in range(y_min, y_max):
            for x_ in range(x_min, x_max):
                dist = np.sqrt((x_ - cx) ** 2 + (y_ - cy) ** 2)
                if dist <= radius:
                    value = np.exp(-dist ** 2 / (2 * (radius / 3) ** 2))
                    self.heatmap[y_, x_] += value * 2
    
    def get_heatmap(self):
        """获取热力图"""
        # 归一化
        heatmap_norm = self.heatmap.copy()
        if np.max(heatmap_norm) > 0:
            heatmap_norm = heatmap_norm / np.max(heatmap_norm) * 255
        
        # 应用颜色映射
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        
        # 添加模糊效果
        heatmap_color = cv2.GaussianBlur(heatmap_color, (15, 15), 0)
        
        # 衰减
        self.heatmap *= 0.95
        self.heatmap = np.clip(self.heatmap, 0, self.max_value)
        
        return heatmap_color
    
    def get_vehicle_counts(self):
        """获取各车型数量"""
        counts = {'car': 0, 'bus': 0, 'truck': 0}
        return counts
