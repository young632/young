# -*- coding: utf-8 -*-
"""测试中文显示功能"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def cv2_puttext_cn(img, text, org, font_size=20, color=(255, 255, 255)):
    """绘制中文字符 - 增强版"""
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/STHeiti Medium.ttc",
        "C:/Windows/Fonts/STSong.ttf",
        "C:/Windows/Fonts/STKaiti.ttf",
        "C:/Windows/Fonts/KaiTi.ttf",
        "C:/Windows/Fonts/SimSun.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    
    font = None
    found_font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
                found_font = font_path
                print(f"成功加载字体: {font_path}")
                break
            except Exception as e:
                print(f"加载字体失败 {font_path}: {e}")
                continue
    
    if font is None:
        print("警告：未找到中文字体，使用默认字体")
        font = ImageFont.load_default()
        found_font = "default"
    
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(org, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), found_font
    except Exception as e:
        print(f"中文绘制错误: {e}")
        return img, found_font

def test_chinese_display():
    # 创建一个测试图像
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    img[:, :] = (50, 50, 50)
    
    # 测试中文绘制
    test_texts = [
        "智能交通监测系统",
        "车流量: 123",
        "速度: 60 km/h",
        "拥堵指数: 45%",
        "车型: 轿车",
        "警告: 超速",
        "测试中文显示"
    ]
    
    y_pos = 30
    for text in test_texts:
        img, font_used = cv2_puttext_cn(img, text, (20, y_pos), font_size=22, color=(0, 255, 0))
        y_pos += 35
    
    # 保存测试图像
    output_path = "test_chinese_output.png"
    cv2.imwrite(output_path, img)
    print(f"测试图像已保存: {output_path}")
    
    # 显示图像
    cv2.imshow("中文显示测试", img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    return font_used

if __name__ == "__main__":
    font = test_chinese_display()
    print(f"\n测试完成！使用字体: {font}")
