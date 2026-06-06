import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = 'traffic_monitor_secret_key_2024'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    LOG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(os.path.dirname(os.path.dirname(__file__)), 'traffic_data.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # YOLO模型路径
    YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov8n.pt')
    
    # 车辆类型配置
    VEHICLE_TYPES = {
        0: {'name': '轿车', 'color': (0, 255, 0), 'speed_factor': 1.0},
        1: {'name': '货车', 'color': (255, 140, 0), 'speed_factor': 0.7},
        2: {'name': '公交', 'color': (0, 200, 255), 'speed_factor': 0.6},
    }
    
    # 检测参数
    CONFIDENCE_THRESHOLD = 0.35
    IOU_THRESHOLD = 0.45
    SPEED_LIMIT = 60  # km/h
