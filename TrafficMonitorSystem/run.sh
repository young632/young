#!/bin/bash
echo "启动智能交通视觉监测系统 Web 服务..."

# 检查是否安装了依赖
if ! pip list | grep -q flask; then
    echo "安装依赖..."
    pip install -r requirements.txt
fi

# 下载YOLO模型（如果不存在）
if [ ! -f "yolov8n.pt" ]; then
    echo "下载YOLOv8模型..."
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -q
fi

# 启动服务
echo "启动 Flask 服务器..."
python app.py