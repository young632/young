from collections import defaultdict

class VehicleClassifier:
    """车型分类器 - 严格限定三类车型：轿车、公交、货车"""
    
    VEHICLE_TYPES = {
        0: {'name': '轿车', 'color': (0, 255, 0), 'speed_factor': 1.0},
        1: {'name': '货车', 'color': (255, 140, 0), 'speed_factor': 0.7},
        2: {'name': '公交', 'color': (0, 200, 255), 'speed_factor': 0.6},
    }
    
    def __init__(self):
        self.type_history = {}  # 记录每辆车的历史分类，防止跳变
        self.history_length = 5
    
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
        """获取稳定的车型，防止帧间跳变"""
        if track_id not in self.type_history:
            self.type_history[track_id] = []
        
        self.type_history[track_id].append(current_type)
        if len(self.type_history[track_id]) > self.history_length:
            self.type_history[track_id].pop(0)
        
        # 如果历史只有一个类型，直接返回
        if len(self.type_history[track_id]) == 1:
            return current_type
        
        # 加权投票机制：最近的检测权重更高
        type_weights = defaultdict(int)
        for i, vtype in enumerate(self.type_history[track_id]):
            # 权重随时间递增，最新的检测权重最高
            weight = i + 1
            type_weights[vtype] += weight
        
        # 返回权重最高的类型
        return max(type_weights, key=type_weights.get)
    
    def cleanup(self, active_track_ids):
        """清理消失车辆的历史数据"""
        for track_id in list(self.type_history.keys()):
            if track_id not in active_track_ids:
                del self.type_history[track_id]
