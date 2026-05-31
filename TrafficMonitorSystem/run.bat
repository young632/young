@echo off
chcp 65001 >nul
echo 启动智能交通视觉监测系统 Web 服务...
echo.

:: 检查是否安装了依赖
pip list | findstr /i flask >nul
if %errorlevel% neq 0 (
    echo 安装依赖...
    pip install -r requirements.txt
)

:: 检查YOLO模型是否存在
if not exist yolov8n.pt (
    echo 下载YOLOv8模型...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt' -OutFile 'yolov8n.pt'"
)

:: 启动服务
echo 启动 Flask 服务器...
echo 服务地址: http://localhost:5000
echo 按 Ctrl+C 停止服务
echo.
python app.py