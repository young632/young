# -*- coding: utf-8 -*-
"""
数据库模型定义
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class TrafficRecord(db.Model):
    """交通监测记录表"""
    __tablename__ = 'traffic_records'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    video_name = db.Column(db.String(255), nullable=False, comment='视频文件名')
    start_time = db.Column(db.DateTime, nullable=False, default=datetime.now, comment='监测开始时间')
    end_time = db.Column(db.DateTime, comment='监测结束时间')

    # 总体统计数据
    total_count = db.Column(db.Integer, default=0, comment='总车流量')
    avg_speed = db.Column(db.Float, default=0.0, comment='平均车速')
    max_speed = db.Column(db.Float, default=0.0, comment='最高车速')
    min_speed = db.Column(db.Float, default=0.0, comment='最低车速')
    congestion_index = db.Column(db.Float, default=0.0, comment='拥堵指数')
    processed_frames = db.Column(db.Integer, default=0, comment='处理帧数')

    # 车型统计
    car_count = db.Column(db.Integer, default=0, comment='轿车数量')
    truck_count = db.Column(db.Integer, default=0, comment='货车数量')
    bus_count = db.Column(db.Integer, default=0, comment='公交数量')

    # 车型平均速度
    car_avg_speed = db.Column(db.Float, default=0.0, comment='轿车平均速度')
    truck_avg_speed = db.Column(db.Float, default=0.0, comment='货车平均速度')
    bus_avg_speed = db.Column(db.Float, default=0.0, comment='公交平均速度')

    # 速度分布（JSON格式存储）
    speed_distribution = db.Column(db.Text, comment='速度分布JSON')
    congestion_history = db.Column(db.Text, comment='拥堵历史JSON')

    # 备注
    notes = db.Column(db.Text, comment='备注信息')

    # 创建时间
    created_at = db.Column(db.DateTime, default=datetime.now, comment='创建时间')

    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'video_name': self.video_name,
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else '',
            'end_time': self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else '',
            'total_count': self.total_count,
            'avg_speed': round(self.avg_speed, 1),
            'max_speed': round(self.max_speed, 1),
            'min_speed': round(self.min_speed, 1),
            'congestion_index': round(self.congestion_index, 1),
            'processed_frames': self.processed_frames,
            'car_count': self.car_count,
            'truck_count': self.truck_count,
            'bus_count': self.bus_count,
            'car_avg_speed': round(self.car_avg_speed, 1),
            'truck_avg_speed': round(self.truck_avg_speed, 1),
            'bus_avg_speed': round(self.bus_avg_speed, 1),
            'speed_distribution': self.speed_distribution,
            'congestion_history': self.congestion_history,
            'notes': self.notes,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }

    def __repr__(self):
        return f'<TrafficRecord {self.id}: {self.video_name}>'
