import cv2
import numpy as np
import os

def blend_layers(img1, img2, alpha=0.5, beta=0.5, gamma=0):
    """安全的图层融合工具函数，自动处理尺寸和通道不匹配"""
    if img1 is None or img2 is None:
        return img1 if img1 is not None else img2
    
    # 统一尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2 or w1 != w2:
        img2 = cv2.resize(img2, (w1, h1))
    
    # 统一通道数
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # 确保数据类型一致
    if img1.dtype != img2.dtype:
        img2 = img2.astype(img1.dtype)
    
    return cv2.addWeighted(img1, alpha, img2, beta, gamma)

def cv2_puttext_cn(image, text, position, font_size=12, color=(255, 255, 255)):
    """在图像上绘制中文文字"""
    from PIL import Image, ImageDraw, ImageFont
    
    # 将OpenCV图像转换为PIL图像
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    # 尝试加载中文字体
    try:
        font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    except:
        try:
            font = ImageFont.truetype("msyh.ttc", font_size, encoding="utf-8")
        except:
            font = ImageFont.load_default()
    
    draw.text(position, text, fill=(color[2], color[1], color[0]), font=font)
    
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def calculate_center_distance(box1, box2):
    """计算两个检测框中心点之间的距离"""
    cx1 = box1[0] + box1[2] // 2
    cy1 = box1[1] + box1[3] // 2
    cx2 = box2[0] + box2[2] // 2
    cy2 = box2[1] + box2[3] // 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

def allowed_file(filename, allowed_extensions):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_output_filename(input_filename):
    """生成输出文件名"""
    name, ext = os.path.splitext(input_filename)
    return f"{name}_detected{ext}"
