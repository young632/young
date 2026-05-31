# 智能交通视觉监测系统

基于OpenCV的传统视觉算法实现的智能交通监测系统，支持车流量统计、车辆速度检测、交通热力图三大核心功能。

## 📁 项目结构

```
TrafficMonitorSystem/
├── traffic_monitor.py    # 核心系统类
├── config.py             # 配置参数
├── gui.py                # GUI界面
├── requirements.txt      # 依赖清单
└── README.md             # 使用说明
```

## 🎯 核心功能

### 1. 车流量统计
- **虚拟线圈检测**：在图像下方设置虚拟检测线
- **MOG2背景减法**：自动适应环境变化
- **轮廓检测**：准确提取车辆区域
- **车道区分**：区分不同车道的车辆

### 2. 车辆速度检测
- **轨迹跟踪**：基于质心的多目标跟踪
- **瞬时速度**：帧间位移换算
- **超速标记**：红色警告框标注超速车辆
- **速度平滑**：避免速度抖动

### 3. 交通热力图
- **运动轨迹**：记录车辆位置
- **热成像映射**：JET色彩编码
- **实时叠加**：半透明显示
- **拥堵分析**：颜色深浅反映交通密度

## ⚙️ 技术参数

| 模块 | 参数 |
|------|------|
| 背景减法 | MOG2, history=100, varThreshold=50 |
| 轮廓过滤 | 面积500-50000像素 |
| 车速换算 | 0.05米/像素 |
| 热力图 | JET colormap, decay=0.95 |
| 车道线 | Canny(50,150), HoughLinesP |
| 颜色空间 | HSL白[0,200,0]-[255,255,255], 黄[15,40,100]-[35,255,255] |

## 🚀 快速开始

### 方式一：Python API

```python
from traffic_monitor import TrafficMonitor

# 创建监测系统
monitor = TrafficMonitor(show_lane=True, show_heatmap=True, show_speed=True)

# 处理视频
monitor.process_video('input.mp4', 'output.mp4')

# 处理摄像头
monitor.process_camera(0)

# 批量处理图片
monitor.process_images(['img1.jpg', 'img2.jpg'])
```

### 方式二：GUI界面

```bash
cd TrafficMonitorSystem
python gui.py
```

## 📊 使用示例

### 示例1：完整车流量统计

```python
from traffic_monitor import TrafficMonitor

monitor = TrafficMonitor()
monitor.speed_limit = 80  # 设置限速
monitor.process_video('traffic.mp4', 'result.mp4')
```

### 示例2：只显示车流量

```python
monitor = TrafficMonitor(show_lane=False, show_heatmap=False, show_speed=False)
monitor.process_video('traffic.mp4')
```

### 示例3：获取统计数据

```python
monitor = TrafficMonitor()
monitor.process_video('traffic.mp4')

stats = monitor.get_statistics()
print(f"总车流量: {stats['total_count']}")
print(f"平均速度: {stats['avg_speed']:.1f} km/h")
```

## 🎨 系统架构

```
TrafficMonitor
├── VehicleTracker        # 车辆跟踪器
│   ├── 注册/注销车辆
│   ├── 质心匹配
│   └── 速度计算
├── HeatmapGenerator      # 热力图生成器
│   ├── 轨迹更新
│   ├── 热力衰减
│   └── 彩色映射
└── LaneDetector          # 车道线检测器
    ├── 颜色过滤
    ├── Canny边缘
    └── 霍夫变换
```

## 🔧 配置参数

所有参数可在 `config.py` 中修改：

```python
# 车流量统计
MIN_CONTOUR_AREA = 500
MAX_CONTOUR_AREA = 50000

# 车速检测
SPEED_THRESHOLD = 80.0
PIXELS_TO_METERS = 0.05

# 热力图
HEATMAP_DECAY = 0.95
HEATMAP_THRESHOLD = 0.3
```

## 📝 实验报告要点

### 1. 算法原理
- **背景减法**：MOG2高斯混合模型，自适应更新背景
- **轮廓检测**：连通区域分析，提取运动目标
- **多目标跟踪**：基于质心的最近邻匹配
- **热力图**：核密度估计 + JET色彩映射

### 2. 实验步骤
1. 视频读取与预处理
2. MOG2背景建模
3. 前景分割与轮廓提取
4. 车辆跟踪与计数
5. 速度计算与显示
6. 热力图生成
7. 车道线检测

### 3. 结果分析
- 车流量统计准确率
- 速度检测误差分析
- 热力图拥堵区域识别

## ⚠️ 注意事项

1. **光照变化**：MOG2可适应轻微光照变化，极端情况需调整参数
2. **遮挡处理**：车辆遮挡时跟踪可能丢失
3. **相机角度**：建议俯视角度拍摄
4. **分辨率**：推荐1280x720以上

## 📦 依赖安装

```bash
pip install opencv-python numpy moviepy Pillow
```

## 🎓 课程设计亮点

1. **三大功能集成**：车流量+车速+热力图
2. **模块化设计**：各功能独立可复用
3. **实时处理**：支持摄像头实时检测
4. **可视化界面**：GUI操作简便
5. **可扩展架构**：易于添加新功能

## 📄 许可证

MIT License