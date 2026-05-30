# OpenCV 车道线检测系统 - 课程设计拓展方案

## 项目定位
基于传统计算机视觉的道路车道线实时检测系统，支持图片/视频/摄像头输入，具备车道偏离预警功能。

## 拓展目标
将基础实验代码封装为完整的课程设计项目，实现：工程化代码结构、算法增强（鸟瞰图+弯道拟合+车道跟踪）、LDWS预警、YOLO目标检测融合、多图对比可视化、批量处理工具。

---

## 功能模块清单

### 核心功能（必做）
| 功能 | 说明 | 文件 |
|------|------|------|
| 车道线检测 | HSL颜色过滤+Canny+霍夫变换+直线拟合 | `core/detection.py` |
| 固定参数配置 | Canny(50,150)、HSL阈值、霍夫参数 | `config.py` |
| 视频处理 | moviepy逐帧处理 | `core/video.py` |
| 图片处理 | 支持中英文路径 | `utils/io.py` |
| 结果可视化 | 车道线叠加+透明度融合 | `core/draw.py` |

### 拓展功能（选做）
| 功能 | 说明 | 文件 |
|------|------|------|
| 鸟瞰图变换 | IPM透视矫正 | `core/birdview.py` |
| 弯道拟合 | 曲线B样条/多项式拟合 | `core/curve_fit.py` |
| 车道跟踪 | 帧间平滑+卡尔曼滤波 | `core/tracker.py` |
| 暗光增强 | CLAHE对比度增强 | `core/enhance.py` |
| LDWS预警 | 偏移量计算+语音/文字报警 | `features/ldws.py` |
| 曲率计算 | 车道曲率半径计算 | `features/curvature.py` |
| 实时数据显示 | FPS+曲率+偏移量叠加 | `features/hud.py` |
| 批量处理 | 多文件批量检测 | `tools/batch.py` |
| 性能统计 | 处理时间统计 | `tools/statistics.py` |
| 日志记录 | 检测日志保存 | `tools/logger.py` |
| 数据导出 | JSON/CSV格式结果 | `tools/export.py` |

### 高阶功能（拔高）
| 功能 | 说明 | 文件 |
|------|------|------|
| YOLO目标检测 | 车辆/行人检测融合 | `features/yolo_detector.py` |
| 多图对比显示 | 处理流程分步对比 | `展示/compare.py` |
| 命令行菜单 | CLI参数选择 | `cli.py` |
| Flask Web服务 | API接口 | `web/app.py` |
| GUI界面 | Tkinter桌面应用 | `gui.py` |

---

## 系统架构

### 目录结构
```
LaneDetection/
├── config.py              # 固定参数配置（必改）
├── cli.py                 # 命令行菜单入口
│
├── core/                  # 核心算法模块
│   ├── __init__.py
│   ├── detection.py       # 车道线检测主流程
│   ├── birdview.py       # 鸟瞰图变换
│   ├── curve_fit.py       # 曲线拟合
│   ├── tracker.py        # 车道跟踪
│   ├── enhance.py        # 暗光增强
│   ├── draw.py           # 绘制可视化
│   └── video.py          # 视频处理
│
├── features/              # 拓展功能模块
│   ├── __init__.py
│   ├── ldws.py           # 车道偏离预警
│   ├── curvature.py      # 曲率计算
│   ├── hud.py            # 实时数据显示
│   └── yolo_detector.py  # YOLO目标检测
│
├── tools/                 # 工具模块
│   ├── __init__.py
│   ├── batch.py          # 批量处理
│   ├── statistics.py     # 性能统计
│   ├── logger.py         # 日志记录
│   └── export.py         # 数据导出
│
├── display/               # 展示模块
│   ├── __init__.py
│   └── compare.py        # 多图对比显示
│
├── gui.py                 # GUI界面
├── web/                   # Web服务
│   └── app.py
│
├── tests/                 # 单元测试
├── test_images/           # 测试图片
├── test_videos/           # 测试视频
├── requirements.txt
└── README.md
```

### 模块分工
```
config.py        → 所有固定参数集中管理
core/detection.py → 图像处理主流程（pipeline）
core/birdview.py → 鸟瞰图IPM变换
core/curve_fit.py → 弯道曲线拟合
core/tracker.py  → 车道线跟踪平滑
core/enhance.py  → 低光照图像增强
features/ldws.py → 车道偏离预警逻辑
features/yolo_detector.py → YOLO融合
tools/batch.py   → 批量文件处理
display/compare.py → matplotlib多子图对比
gui.py           → Tkinter图形界面
cli.py           → 命令行交互菜单
```

---

## 开发顺序（优先级：必做→选做→拔高）

### 第一批：必做（工程化封装）
```
优先级P0 - 必须完成，课程设计基础分

1. config.py 创建
   - Canny阈值: 50, 150
   - HSL白色: [0,200,0]-[255,255,255]
   - HSL黄色: [15,40,100]-[35,255,255]
   - 霍夫参数: rho=1, theta=π/180, threshold=30, minLineLength=50, maxLineGap=10
   - 融合权重: 0.7/0.3

2. core/detection.py 重构
   - LaneDetector类封装
   - process(frame)主方法
   - process_image()/process_video()对外接口

3. cli.py 命令行菜单
   - 1. 处理单张图片
   - 2. 处理视频
   - 3. 批量处理
   - 4. 摄像头实时
   - 5. 展示流程
   - 0. 退出
```

### 第二批：选做（算法增强）
```
优先级P1 - 提升检测效果，增加创新点

1. core/birdview.py - 鸟瞰图
   实现：cv2.getPerspectiveTransform + cv2.warpPerspective
   参数：源点4个(前方消失点区域)，目标点矩形

2. core/curve_fit.py - 弯道拟合
   实现：np.polyfit多项式拟合(degree=2)
   替代直线拟合，处理弯道场景

3. core/tracker.py - 车道跟踪
   实现：滑动平均/指数平滑
   目的：减少视频抖动

4. core/enhance.py - 暗光增强
   实现：cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   目的：提升夜间/阴影检测效果
```

### 第三批：选做（核心拓展功能）
```
优先级P1 - 功能亮点，写入实验报告

1. features/ldws.py - LDWS车道偏离预警
   实现：
   - 计算车道中心与车辆中心偏移量
   - 偏移阈值报警(>0.3倍车道宽)
   - 文字/语音提示

2. features/curvature.py - 曲率计算
   实现：基于拟合多项式计算曲率半径
   R = [(1+(dx/dy)^2)^1.5]/|d²x/dy²|

3. features/hud.py - 实时数据显示
   实现：在画面上叠加显示
   - 当前FPS
   - 车道曲率
   - 偏离距离
   - 车道宽度
```

### 第四批：选做（YOLO融合）
```
优先级P2 - 高阶功能，锦上添花

1. features/yolo_detector.py
   实现：
   - 加载YOLO权重(yolov3/yolov4)
   - 目标检测(车辆/行人/交通标志)
   - 与车道线检测结果叠加显示

   注意：需要下载yolo权重文件
   - yolov3.weights
   - yolov3.cfg
```

### 第五批：拔高（工具+展示）
```
优先级P2 - 完善项目体验

1. tools/batch.py - 批量处理
   实现：遍历文件夹，处理所有图片/视频

2. tools/statistics.py - 性能统计
   实现：统计每帧处理时间，计算平均FPS

3. tools/logger.py - 日志记录
   实现：检测时间、输入输出路径、处理帧数

4. tools/export.py - 数据导出
   实现：JSON格式导出检测结果

5. display/compare.py - 多图对比
   实现：plt.subplot(2,3,...)展示6个中间步骤
   - 原图
   - HSL掩码
   - 边缘图
   - ROI图
   - 霍夫直线
   - 最终结果
```

---

## 技术实现要点

### 1. config.py - 固定参数配置
```python
class Config:
    # Canny边缘检测
    CANNY_LOW = 50
    CANNY_HIGH = 150

    # HSL白色阈值 [H, L, S]
    HSL_WHITE_LOW = np.array([0, 200, 0], dtype=np.uint8)
    HSL_WHITE_HIGH = np.array([255, 255, 255], dtype=np.uint8)

    # HSL黄色阈值
    HSL_YELLOW_LOW = np.array([15, 40, 100], dtype=np.uint8)
    HSL_YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

    # 霍夫概率变换
    HOUGH_RHO = 1
    HOUGH_THETA = math.pi / 180
    HOUGH_THRESHOLD = 30
    HOUGH_MIN_LINE_LENGTH = 50
    HOUGH_MAX_LINE_GAP = 10

    # 融合权重
    OVERLAY_WEIGHT = 0.7
    ORIGINAL_WEIGHT = 0.3

    # ROI顶点(相对比例)
    ROI_VERTICES = np.array([
        [0.1, 1.0], [0.45, 0.6], [0.55, 0.6], [0.9, 1.0]
    ])
```

### 2. core/detection.py - 车道线检测
```python
class LaneDetector:
    def __init__(self, config=None):
        self.config = config or Config()

    def process(self, frame):
        """完整检测流程"""
        # 1. HSL颜色过滤
        hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(hsl, self.config.HSL_WHITE_LOW, self.config.HSL_WHITE_HIGH)
        yellow_mask = cv2.inRange(hsl, self.config.HSL_YELLOW_LOW, self.config.HSL_YELLOW_HIGH)
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        filtered = cv2.bitwise_and(frame, frame, mask=color_mask)

        # 2. 灰度化+高斯模糊
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Canny边缘检测
        edges = cv2.Canny(blurred, self.config.CANNY_LOW, self.config.CANNY_HIGH)

        # 4. ROI掩码
        mask = self._create_roi_mask(frame.shape)
        masked = cv2.bitwise_and(edges, mask)

        # 5. 霍夫变换
        lines = cv2.HoughLinesP(masked,
            self.config.HOUGH_RHO,
            self.config.HOUGH_THETA,
            self.config.HOUGH_THRESHOLD,
            minLineLength=self.config.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=self.config.HOUGH_MAX_LINE_GAP)

        # 6. 分离左右车道+拟合
        left_line, right_line = self._fit_lines(lines)

        # 7. 绘制结果
        result = self._draw_lanes(frame, left_line, right_line)
        return result
```

### 3. core/birdview.py - 鸟瞰图
```python
def birdview_transform(frame):
    """生成鸟瞰图"""
    h, w = frame.shape[:2]

    # 源点：前方道路四边形(按实际情况调整)
    src_points = np.float32([
        [w*0.4, h*0.6],   # 左上
        [w*0.6, h*0.6],   # 右上
        [w*0.9, h*1.0],   # 右下
        [w*0.1, h*1.0]    # 左下
    ])

    # 目标点：矩形
    dst_points = np.float32([
        [w*0.3, 0],
        [w*0.7, 0],
        [w*0.9, h],
        [w*0.1, h]
    ])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, (w, h))
    return warped, M
```

### 4. core/curve_fit.py - 曲线拟合
```python
def fit_curved_line(points, y_coords):
    """多项式曲线拟合"""
    if len(points) < 3:
        return None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # 过滤下半部分(近处)
    mask = y > np.mean(y_coords)
    if np.sum(mask) < 3:
        return None

    # 2次多项式拟合
    coeffs = np.polyfit(y[mask], x[mask], 2)
    return coeffs

def draw_curved_line(frame, coeffs, color=(0, 255, 0), thickness=5):
    """绘制曲线"""
    h, w = frame.shape[:2]
    y = np.linspace(h*0.6, h-1, 50)
    x = np.polyval(coeffs, y)

    points = [(int(xi), int(yi)) for xi, yi in zip(x, y) if 0 <= xi < w]
    for i in range(len(points)-1):
        cv2.line(frame, points[i], points[i+1], color, thickness)
```

### 5. core/tracker.py - 车道跟踪
```python
class LaneTracker:
    """滑动平均车道跟踪"""
    def __init__(self, n_frames=5):
        self.left_history = []
        self.right_history = []
        self.n_frames = n_frames

    def update(self, left_line, right_line):
        """更新跟踪"""
        self.left_history.append(left_line)
        self.right_history.append(right_line)

        if len(self.left_history) > self.n_frames:
            self.left_history.pop(0)
            self.right_history.pop(0)

    def get_smoothed(self):
        """获取平滑后的车道线"""
        if not self.left_history:
            return None, None

        left_avg = np.mean(self.left_history, axis=0)
        right_avg = np.mean(self.right_history, axis=0)
        return left_avg, right_avg
```

### 6. core/enhance.py - 暗光增强
```python
def enhance_low_light(frame):
    """CLAHE暗光增强"""
    # 转换到LAB颜色空间
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE增强L通道
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # 合并通道
    enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return result
```

### 7. features/ldws.py - 车道偏离预警
```python
class LDWS:
    """车道偏离预警系统"""
    def __init__(self, lane_width=3.7):  # 标准车道宽度3.7米
        self.lane_width = lane_width

    def check_departure(self, vehicle_offset, lane_curvature=None):
        """
        检测是否偏离
        vehicle_offset: 偏移量(米)，正值为右偏，负值为左偏
        """
        threshold = self.lane_width * 0.3  # 30%车道宽度

        if abs(vehicle_offset) > threshold:
            direction = "右偏" if vehicle_offset > 0 else "左偏"
            return True, f"WARNING: 车辆{direction}偏离!"
        return False, "Normal"

    def calculate_offset(self, left_line, right_line, focal_length=700):
        """
        计算车辆相对车道中心的偏移
        返回: 偏移量(米)
        """
        if left_line is None or right_line is None:
            return 0

        # 图像下半部分计算
        lane_center = (left_line + right_line) / 2
        image_center = 640  # 假设图像宽度1280

        pixel_offset = lane_center - image_center
        # 简化转换：假设远处3米对应100像素
        meters_per_pixel = 3.0 / 100
        return pixel_offset * meters_per_pixel
```

### 8. features/yolo_detector.py - YOLO融合
```python
class YOLODetector:
    """YOLO目标检测"""
    def __init__(self, weights_path, config_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = ["car", "person", "truck", "bus"]  # 目标类别

    def detect(self, frame):
        """检测车辆和行人"""
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
        self.net.setInput(blob)

        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        results = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and self.classes[class_id] in ["car", "person"]:
                    results.append({
                        'class': self.classes[class_id],
                        'confidence': float(confidence),
                        'bbox': detection[:4]  # x,y,w,h
                    })
        return results

    def draw_detections(self, frame, detections):
        """绘制检测框"""
        for det in detections:
            x, y, w, h = det['bbox']
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### 9. tools/batch.py - 批量处理
```python
def batch_process(input_dir, output_dir, detector):
    """批量处理目录下所有图片/视频"""
    import os
    from pathlib import Path

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for file in input_path.glob('*'):
        if file.suffix.lower() in ['.jpg', '.png', '.bmp']:
            frame = cv2.imread(str(file))
            result = detector.process(frame)
            cv2.imwrite(str(output_path / file.name), result)
            results.append({'file': file.name, 'status': 'success'})

        elif file.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # 视频处理...
            results.append({'file': file.name, 'status': 'success'})

    return results
```

### 10. display/compare.py - 多图对比
```python
def display_processing_steps(frame, steps_data):
    """
    显示处理流程对比图
    steps_data = {
        'Original': frame,
        'HSL Mask': hsl_mask,
        'Edges': edges,
        'ROI': roi_masked,
        'Hough': hough_result,
        'Final': final_result
    }
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    titles = list(steps_data.keys())
    images = list(steps_data.values())

    for ax, title, img in zip(axes, titles, images):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
```

---

## 实验报告可直接用的亮点总结

### 1. 工程化亮点
- 采用面向对象设计，LancDetector类封装完整检测流程
- 配置与逻辑分离，Config类集中管理所有固定参数
- 模块化分层：core/features/tools分离
- 支持命令行菜单、GUI界面、Web服务三种使用方式

### 2. 算法增强亮点
- **鸟瞰图变换**：IPM透视矫正，提升弯道检测效果
- **曲线拟合**：多项式拟合替代直线拟合，支持弯道场景
- **车道跟踪**：滑动平均平滑，减少视频抖动
- **暗光增强**：CLAHE算法，提升低光照检测能力

### 3. 功能创新亮点
- **LDWS车道偏离预警**：实时计算偏移量，超阈值报警
- **曲率计算**：基于多项式拟合计算车道曲率半径
- **YOLO融合**：目标检测与车道检测协同，检测车辆行人
- **实时HUD**：FPS、曲率、偏移量实时叠加显示

### 4. 工具完备亮点
- 批量处理：文件夹批量检测多张图片/视频
- 性能统计：FPS、处理时间详细统计
- 日志记录：检测全过程日志保存
- 数据导出：JSON格式导出检测结果

### 5. 展示效果亮点
- 处理流程可视化：6步骤分图对比展示
- 原图vs检测结果对比
- 视频逐帧处理进度显示

---

## 风险与注意事项

### 技术风险
| 风险 | 等级 | 应对措施 |
|------|:---:|----------|
| YOLO权重文件下载失败 | 中 | 预先下载，或使用简化版tiny-yolo |
| 夜间模式效果不稳定 | 中 | 增加CLAHE的clipLimit参数调节 |
| 弯道拟合点数不足 | 高 | 设置最小点数阈值，不足时回退到直线拟合 |
| 视频处理速度慢 | 低 | 使用多线程/减少输出帧 |

### 开发风险
| 风险 | 等级 | 应对措施 |
|------|:---:|----------|
| 模块间耦合过高 | 中 | 保持core模块独立，features依赖core |
| 参数调试困难 | 中 | CLI菜单增加参数调节选项 |
| 报告内容不足 | 低 | 预留足够实验数据分析章节 |

### 使用注意
1. **固定参数不可随意修改**：Canny阈值、HSL阈值等严格按照给定值
2. **ROI区域调整**：根据实际视频调整四边形顶点
3. **YOLO模型选择**：推荐使用yolov3-tiny减少计算量
4. **视频格式兼容**：moviepy对某些编码格式支持不好，可使用opencv替代

### 调试建议
1. 先用单张图片调试各模块
2. 视频处理时每隔30帧显示一次预览
3. 保存中间结果图片便于分析问题
4. 使用display/compare.py逐流程排查

---

*文档版本: v2.0*
*适配: OpenCV + moviepy + matplotlib + YOLO*
