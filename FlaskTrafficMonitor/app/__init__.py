from flask import Flask
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__)
    
    # 配置
    app.config.from_object('app.config.Config')
    
    # 跨域
    CORS(app)
    
    # 创建必要目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
    
    # 注册路由
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
