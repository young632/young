# OpenCV 车道线检测系统 - 拓展方案

## 一、项目背景

已完成基础OpenCV车道线检测实验，实现了基于传统视觉的车道线检测功能。本拓展方案将其包装成完整的课程设计/大作业项目，涵盖工程封装、算法增强、新功能、界面部署四大方向。

---

## 二、拓展方向总览

| 优先级 | 方向 | 难度 | 工作量 | 可行性 |
|:---:|------|:---:|:---:|:---:|
| **P0** | 工程封装 | ⭐ | 中 | 高 |
| **P1** | 算法增强 | ⭐⭐ | 中 | 高 |
| **P2** | 新功能拓展 | ⭐⭐⭐ | 大 | 中 |
| **P3** | 界面/部署 | ⭐⭐ | 中 | 高 |

---

## 三、P0 - 工程封装（基础建设）

### 目标
将散落代码封装成标准Python项目，可import使用，符合课程设计规范。

### 具体任务

#### 1. 项目目录结构
```
LaneDetection/
├── lane_detection/           # 主包
│   ├── __init__.py
│   ├── core/                # 核心算法
│   │   ├── __init__.py
│   │   ├── color_filter.py  # 颜色空间处理
│   │   ├── edge.py          # 边缘检测
│   │   ├── hough.py         # 霍夫变换
│   │   ├── lane_fit.py      # 车道线拟合
│   │   └── pipeline.py      # 处理流水线
│   ├── utils/               # 工具函数
│   │   ├── __init__.py
│   │   ├── io.py            # 文件读写
│   │   └── visualization.py # 可视化
│   └── config.py            # 固定参数配置
├── tests/                   # 单元测试
├── docs/                     # 文档
├── requirements.txt
├── setup.py
└── README.md
```

#### 2. 固定参数配置化
```python
# lane_detection/config.py
class Config:
    # Canny边缘检测
    CANNY_LOW = 50
    CANNY_HIGH = 150

    # HSL颜色阈值
    HSL_WHITE_LOW = [0, 200, 0]
    HSL_WHITE_HIGH = [255, 255, 255]
    HSL_YELLOW_LOW = [15, 40, 100]
    HSL_YELLOW_HIGH = [35, 255, 255]

    # 霍夫变换
    HOUGH_RHO = 1
    HOUGH_THETA = math.pi / 180
    HOUGH_THRESHOLD = 30
    HOUGH_MIN_LINE_LENGTH = 50
    HOUGH_MAX_LINE_GAP = 10

    # 融合权重
    OVERLAY_WEIGHT = 0.7
    ORIGINAL_WEIGHT = 0.3

    # ROI参数
    ROI_VERTICES = [[0.1, 1.0], [0.45, 0.6], [0.55, 0.6], [0.9, 1.0]]
```

#### 3. pipeline处理流水线
```python
# lane_detection/core/pipeline.py
class LaneDetectionPipeline:
    def __init__(self, config=None):
        self.config = config or Config()

    def process(self, frame):
        # 完整处理流程
        pass

    def process_video(self, input_path, output_path):
        # 视频处理
        pass
```

### 交付物
- [ ] 标准Python包结构
- [ ] `pip install -e .` 可安装
- [ ] 单元测试覆盖核心函数
- [ ] README使用文档

---

## 四、P1 - 算法增强（核心优化）

### 目标
提升检测精度和稳定性，添加动态阈值、曲线拟合等高级算法。

### 具体任务

#### 1. 动态阈值适配
**问题**：固定阈值在不同光照下效果差

**实现方案**：
```python
def auto_canny_threshold(gray_image):
    """基于图像统计的动态阈值"""
    median = np.median(gray_image)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray_image, lower, upper)
```

#### 2. 曲线拟合替代直线拟合
**问题**：弯道直线拟合效果差

**实现方案**：
```python
from scipy.interpolate import UnivariateSpline

def fit_curved_line(points, y_coords):
    """B样条曲线拟合"""
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    # 只在后半部分图像拟合
    mask = y > np.mean(y_coords)
    spline = UnivariateSpline(y[mask], x[mask], k=3, s=50)
    return spline
```

#### 3. 滑动窗口检测
**问题**：霍夫变换易受阴影干扰

**实现方案**：
```python
def sliding_window_detection(binary_warped):
    """基于滑动窗口的车道线检测"""
    nwindows = 9
    window_height = binary_warped.shape[0] // nwindows

    # 左右车道线起始点
    leftx_base, rightx_base = find_lane_start_points(binary_warped)

    # 逐窗口追踪
    for window in range(nwindows):
        # 窗口内提取车道线像素
        pass
```

#### 4. 帧间平滑
**问题**：视频检测结果抖动

**实现方案**：
```python
class LaneTracker:
    """车道线追踪器，跨帧平滑"""
    def __init__(self, n_frames=5):
        self.left_lines = []
        self.right_lines = []
        self.n_frames = n_frames

    def update(self, left_line, right_line):
        self.left_lines.append(left_line)
        self.right_lines.append(right_line)
        if len(self.left_lines) > self.n_frames:
            self.left_lines.pop(0)
            self.right_lines.pop(0)

    def get_smoothed(self):
        return average_lines(self.left_lines), average_lines(self.right_lines)
```

### 交付物
- [ ] 动态阈值Canny
- [ ] 曲线拟合（弯道支持）
- [ ] 滑动窗口检测
- [ ] 帧间平滑追踪

---

## 五、P2 - 新功能拓展（亮点功能）

### 目标
增加实用功能，提升项目价值，可写入实验报告"创新点"部分。

### 具体任务

#### 1. 车道偏离预警 (LDW)
```python
def lane_departure_warning(vehicle_offset, lane_width):
    """
    检测车辆是否偏离车道
    vehicle_offset: 车辆中心与车道中心的偏移
    lane_width: 车道宽度
    """
    if abs(vehicle_offset) > lane_width * 0.3:
        return "WARNING: Lane Departure!"
    return "Normal"
```

#### 2. 曲率计算
```python
def calculate_curvature(left_line, right_line, y_eval, m_per_pix):
    """
    计算车道曲率半径
    m_per_pix: 像素到米的转换系数
    """
    # 转换到世界坐标
    leftx = left_line[0] * m_per_pix
    lefty = left_line[1] * m_per_pix

    # 拟合多项式
    left_fit = np.polyfit(lefty, leftx, 2)

    # 计算曲率
    curvature = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    return curvature
```

#### 3. 距离估算
```python
def estimate_distance_to_lane(vehicle_offset, curvature):
    """
    估算到车道线的距离
    基于简单的透视几何
    """
    # 假设：远处车道线在图像上部
    focal_length = 700  # 假设焦距
    lane_width_real = 3.7  # 标准车道宽度 3.7米

    # 计算距离
    distance = (lane_width_real * focal_length) / (2 * abs(vehicle_offset))
    return distance
```

#### 4. 夜间模式
```python
def night_mode_detection(frame):
    """
    夜间场景专用处理
    - 增强对比度
    - 调整白平衡
    - 使用更大阈值
    """
    # CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(frame)

    # 调整饱和度
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.add(hsv[:,:,1], 30)  # 增加饱和度
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

#### 5. 性能统计
```python
def calculate_fps(start_time, frame_count):
    """计算实时FPS"""
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    return fps

def draw_performance_info(frame, fps, processing_time):
    """在画面上绘制性能信息"""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Time: {processing_time*1000:.1f}ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```

### 交付物
- [ ] 车道偏离预警
- [ ] 曲率半径计算
- [ ] 距离估算
- [ ] 夜间模式
- [ ] 实时性能统计

---

## 六、P3 - 界面与部署（用户体验）

### 目标
提供友好的用户界面，支持图片/视频/摄像头多种输入方式。

### 具体任务

#### 1. 桌面GUI增强
```python
class LaneDetectionApp:
    """桌面应用完整功能"""

    def __init__(self):
        self.pipeline = LaneDetectionPipeline()
        self.current_mode = 'image'  # image/video/camera

    # 功能列表
    def select_image(self):
        """选择单张图片"""

    def select_video(self):
        """选择视频文件"""

    def start_camera(self):
        """开启摄像头实时检测"""

    def adjust_parameters(self):
        """滑块调节参数"""

    def export_report(self):
        """导出检测报告"""
```

#### 2. 摄像头实时检测
```python
def camera_realtime_detection():
    """实时摄像头车道线检测"""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = lane_pipeline.process(frame)

        # 显示FPS
        cv2.putText(result, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Lane Detection', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

#### 3. 参数调节面板
```python
class ParameterPanel:
    """可调节参数面板"""

    def __init__(self, parent):
        self.params = {
            'canny_low': tk.Scale(label="Canny Low", from_=0, to=200),
            'canny_high': tk.Scale(label="Canny High", from_=0, to=400),
            'hough_threshold': tk.Scale(label="Hough Threshold", from_=0, to=200),
            'roi_slider': tk.Scale(label="ROI Height", from_=0.3, to=1.0),
        }

    def on_param_change(self, event):
        """参数变化回调"""
        update_live_preview()
```

#### 4. Web部署方案
```python
# app.py - Flask Web服务
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/detect', methods=['POST'])
def detect_lane():
    """上传图片进行车道线检测"""
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    result = pipeline.process(img)

    _, buffer = cv2.imencode('.jpg', result)
    return base64.b64encode(buffer).decode()
```

#### 5. 生成检测报告
```python
def generate_report(detection_result, output_path):
    """
    生成检测报告
    - 检测图片
    - 车道曲率
    - 偏离状态
    - 时间戳
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'lane_curvature': detection_result.curvature,
        'departure_warning': detection_result.warning,
        'fps': detection_result.fps,
        'image_path': output_path
    }

    with open(output_path.replace('.jpg', '_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
```

### 交付物
- [ ] 桌面GUI（支持图片/视频/摄像头）
- [ ] 参数实时调节面板
- [ ] Flask Web服务
- [ ] 检测报告生成

---

## 七、开发步骤（推荐顺序）

### 第一阶段：工程封装（2天）
```
Day 1:
- 创建项目目录结构
- 迁移现有代码到core模块
- 实现Config配置类
- 编写__init__.py和setup.py

Day 2:
- 编写单元测试
- 完善README文档
- 制作requirements.txt
- Git初始化提交
```

### 第二阶段：算法增强（3天）
```
Day 3:
- 实现动态阈值Canny
- 测试不同光照条件

Day 4:
- 实现曲线拟合
- 处理弯道场景

Day 5:
- 实现帧间平滑
- 优化视频检测稳定性
```

### 第三阶段：新功能（3天）
```
Day 6:
- 车道偏离预警
- 曲率计算
- 距离估算

Day 7:
- 夜间模式
- 性能统计显示

Day 8:
- 功能集成测试
- 边界case处理
```

### 第四阶段：界面部署（2天）
```
Day 9:
- 完善GUI界面
- 摄像头实时检测

Day 10:
- Flask Web服务
- 检测报告功能
- 项目文档整理
```

---

## 八、功能清单

### 基础功能（已完成）
| 功能 | 状态 | 说明 |
|------|:---:|------|
| 颜色空间过滤 | ✅ | HSL白/黄阈值 |
| Canny边缘检测 | ✅ | 固定阈值50/150 |
| ROI裁剪 | ✅ | 四边形掩码 |
| 霍夫直线检测 | ✅ | HoughLinesP |
| 车道线拟合 | ✅ | 最小二乘 |
| 视频处理 | ✅ | moviepy逐帧 |

### 拓展功能（按优先级）

#### P0 - 工程封装
| 功能 | 难度 | 预估 |
|------|:---:|:---:|
| 模块化封装 | ⭐ | 2h |
| Config配置类 | ⭐ | 1h |
| 单元测试 | ⭐ | 2h |

#### P1 - 算法增强
| 功能 | 难度 | 预估 |
|------|:---:|:---:|
| 动态阈值 | ⭐⭐ | 2h |
| 曲线拟合 | ⭐⭐ | 3h |
| 帧间平滑 | ⭐⭐ | 2h |

#### P2 - 新功能
| 功能 | 难度 | 预估 |
|------|:---:|:---:|
| 车道偏离预警 | ⭐⭐ | 2h |
| 曲率计算 | ⭐⭐ | 2h |
| 距离估算 | ⭐⭐ | 2h |
| 夜间模式 | ⭐⭐⭐ | 4h |
| 性能统计 | ⭐ | 1h |

#### P3 - 界面部署
| 功能 | 难度 | 预估 |
|------|:---:|:---:|
| GUI增强 | ⭐⭐ | 4h |
| 摄像头检测 | ⭐⭐ | 2h |
| 参数调节面板 | ⭐⭐ | 3h |
| Flask Web | ⭐⭐ | 4h |
| 报告生成 | ⭐⭐ | 2h |

---

## 九、实验报告结构建议

```
1. 摘要
2. 引言
   - 研究背景
   - 车道线检测意义
3. 相关技术与理论基础
   - 颜色空间转换
   - 边缘检测算法
   - 霍夫变换原理
4. 系统设计
   - 整体架构
   - 模块划分
   - 数据流程
5. 核心算法实现
   - 固定参数配置
   - 处理流水线
6. 拓展功能（创新点）
   - 算法增强
   - 新增功能
7. 实验与分析
   - 测试数据
   - 性能指标
   - 结果对比
8. 结论与展望
9. 参考文献
```

---

## 十、立即可实现代码清单

| 序号 | 文件 | 功能 | 优先级 |
|:---:|------|------|:---:|
| 1 | `config.py` | 参数配置化 | P0 |
| 2 | `pipeline.py` | 处理流水线封装 | P0 |
| 3 | `dynamic_canny.py` | 动态阈值 | P1 |
| 4 | `curve_fit.py` | 曲线拟合 | P1 |
| 5 | `lane_tracker.py` | 帧间平滑 | P1 |
| 6 | `ldw.py` | 偏离预警 | P2 |
| 7 | `curvature.py` | 曲率计算 | P2 |
| 8 | `night_mode.py` | 夜间模式 | P2 |
| 9 | `gui_enhanced.py` | 增强GUI | P3 |
| 10 | `web_service.py` | Web服务 | P3 |

---

## 十一、推荐起步顺序

```
1. 先完成 P0工程封装（代码规范化）
   ↓
2. 再做 P1算法增强（提升检测效果）
   ↓
3. 接着做 P2新功能（增加创新点）
   ↓
4. 最后做 P3界面部署（完善用户体验）
```

**核心原则**：每完成一个模块，立即测试并更新README文档，保证代码可运行。

---

*文档版本: v1.0*
*更新时间: 2026-05-30*
