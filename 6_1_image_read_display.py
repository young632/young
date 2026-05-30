#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6.1 图像读取与显示
实验步骤：图像读取(matplotlib.imread)、子图显示(plt.subplot)、列表推导式应用
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_image_matplotlib(image_path):
    """
    使用matplotlib.imread读取图像
    读取后自动归一化为[0,1]范围的float32数组

    参数:
        image_path: 图像文件路径（支持Linux/Unix路径格式）

    返回:
        img: 读取的图像数组，读取失败返回None
    """
    try:
        img = mpimg.imread(image_path)
        print(f"图像读取成功: {image_path}")
        print(f"图像形状: {img.shape}")
        print(f"数据类型: {img.dtype}")
        return img
    except FileNotFoundError:
        print(f"错误：文件不存在 {image_path}")
        return None
    except Exception as e:
        print(f"错误：读取图像失败 {e}")
        return None


def load_images_from_directory(image_paths):
    """
    使用列表推导式批量读取多张图像

    参数:
        image_paths: 图像路径列表

    返回:
        images: 读取成功的图像列表
    """
    images = [mpimg.imread(path) for path in image_paths]
    return images


def load_images_with_validation(image_paths):
    """
    使用列表推导式和条件过滤读取图像，只返回成功读取的图像

    参数:
        image_paths: 图像路径列表

    返回:
        valid_images: 成功读取的图像列表
        failed_paths: 读取失败的路径列表
    """
    valid_images = [mpimg.imread(path) for path in image_paths if mpimg.imread(path) is not None]
    failed_paths = [path for path in image_paths if mpimg.imread(path) is None]
    return valid_images, failed_paths


def display_single_image(img, title="Image", cmap=None):
    """
    显示单张图像

    参数:
        img: 图像数组
        title: 图像标题
        cmap: 颜色映射，RGB图像用None，灰度图用'gray'
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_images_subplot(images, titles, rows=1, cols=None, cmap=None):
    """
    使用plt.subplot在子图中显示多张图像
    列表推导式用于构建子图索引

    参数:
        images: 图像列表
        titles: 标题列表（与images长度一致）
        rows: 子图行数
        cols: 子图列数，None则自动计算
        cmap: 颜色映射
    """
    n = len(images)

    if cols is None:
        cols = n

    indices = [i for i in range(1, n + 1)]

    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, indices[idx])
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def display_images_grid(images, titles, cmap=None):
    """
    使用列表推导式自动计算网格布局并显示图像

    参数:
        images: 图像列表
        titles: 标题列表
        cmap: 颜色映射
    """
    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols

    subplot_indices = [i for i in range(1, n + 1)]

    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, subplot_indices[idx])
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def display_with_bgr_to_rgb(images):
    """
    如果图像是BGR格式，使用列表推导式转换为RGB显示
    适用于OpenCV读取的BGR图像

    参数:
        images: BGR图像列表
    """
    rgb_images = [img[:, :, ::-1] if len(img.shape) == 3 and img.shape[2] == 3 else img for img in images]
    return rgb_images


def get_image_info(img):
    """
    使用字典推导式获取图像基本信息

    参数:
        img: 图像数组

    返回:
        info: 包含图像信息的字典
    """
    info = {
        'shape': img.shape,
        'dtype': img.dtype,
        'min': float(img.min()),
        'max': float(img.max()),
        'mean': float(img.mean())
    }
    return info


def main():
    """
    主函数 - 演示6.1节图像读取与显示
    """
    print("=" * 50)
    print("6.1 图像读取与显示实验")
    print("matplotlib.imread + plt.subplot + 列表推导式")
    print("=" * 50)

    print("\n实验说明：")
    print("1. matplotlib.imread: 读取图像并归一化为[0,1]的float数组")
    print("2. plt.subplot: 创建子图显示多张图像")
    print("3. 列表推导式: 批量处理图像列表")

    example_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']

    print(f"\n示例路径列表: {example_paths}")

    print("\n可用函数：")
    print("  load_image_matplotlib(path) - 读取单张图像")
    print("  load_images_from_directory(paths) - 批量读取图像")
    print("  display_single_image(img) - 显示单张图像")
    print("  display_images_subplot(images, titles, rows, cols) - 子图显示")
    print("  display_images_grid(images, titles) - 网格自动布局显示")


if __name__ == "__main__":
    main()
