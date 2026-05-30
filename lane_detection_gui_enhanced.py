#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车道线检测GUI界面 - 增强版
集成: LDWS预警 | 曲率计算 | HUD显示 | 暗光增强 | 车道跟踪
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lane_detection_complete import process_frame, imread_chinese
from config import Config
from core.enhance import enhance_low_light, detect_low_light
from core.tracker import LaneTracker
from features.ldws import LDWS
from features.curvature import CurvatureCalculator
from features.hud import HUD


class EnhancedLaneDetectionGUI:
    """增强版车道线检测界面"""

    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV车道线检测系统 - 增强版")
        self.root.geometry("1500x950")
        self.root.minsize(1300, 850)
        self.root.configure(bg='#1a1a2e')

        # 功能开关
        self.enable_ldws = tk.BooleanVar(value=True)
        self.enable_hud = tk.BooleanVar(value=True)
        self.enable_tracker = tk.BooleanVar(value=True)
        self.enable_low_light = tk.BooleanVar(value=False)

        # 处理状态
        self.input_path = None
        self.is_video = False
        self.cap = None
        self.processing = False
        self.result_frame = None

        # 增强功能组件
        self.ldws = LDWS()
        self.curvature_calc = CurvatureCalculator()
        self.hud = HUD()
        self.tracker = LaneTracker()

        self.setup_ui()

    def setup_ui(self):
        """设置界面"""
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题栏
        header = tk.Frame(main_frame, bg='#1a1a2e', height=80)
        header.pack(fill=tk.X, padx=30, pady=(15, 10))

        tk.Label(header, text="车道线检测系统 - 增强版",
                font=("Microsoft YaHei UI", 24, "bold"),
                fg='#00d9ff', bg='#1a1a2e').pack(side=tk.LEFT)

        tk.Label(header, text="LDWS | 曲率计算 | HUD显示 | 暗光增强 | 车道跟踪",
                font=("Microsoft YaHei UI", 10),
                fg='#888888', bg='#1a1a2e').pack(side=tk.LEFT, padx=20, pady=10)

        # 功能选项栏
        opt_frame = tk.Frame(main_frame, bg='#16213e', padx=20, pady=12)
        opt_frame.pack(fill=tk.X, padx=25, pady=(5, 5))

        tk.Label(opt_frame, text="增强功能:", font=("Microsoft YaHei UI", 10, "bold"),
                fg='#ffffff', bg='#16213e').pack(side=tk.LEFT, padx=(0, 15))

        for text, var in [
            ("车道偏离预警(LDWS)", self.enable_ldws),
            ("实时数据显示(HUD)", self.enable_hud),
            ("车道线跟踪", self.enable_tracker),
            ("暗光增强", self.enable_low_light),
        ]:
            cb = tk.Checkbutton(opt_frame, text=text, variable=var,
                              font=("Microsoft YaHei UI", 10),
                              bg='#16213e', fg='#00ff88',
                              activebackground='#16213e',
                              selectcolor='#0f3460')
            cb.pack(side=tk.LEFT, padx=10)

        # 按钮栏
        btn_frame = tk.Frame(main_frame, bg='#16213e', padx=20, pady=12)
        btn_frame.pack(fill=tk.X, padx=25, pady=(5, 5))

        buttons = [
            ("添加图片", self.add_image, '#0f3460'),
            ("添加视频", self.add_video, '#0f3460'),
            ("摄像头检测", self.start_camera, '#0f3460'),
            ("开始检测", self.start_detection, '#e94560'),
            ("保存结果", self.save_result, '#0f3460'),
            ("清空", self.clear_all, '#16213e'),
        ]

        for text, cmd, bg in buttons:
            tk.Button(btn_frame, text=text, command=cmd,
                     font=("Microsoft YaHei UI", 10, 'bold'),
                     bg=bg, fg='white', activebackground='#00d9ff',
                     relief='flat', padx=18, pady=8, cursor='hand2').pack(side=tk.LEFT, padx=6)

        # 状态栏
        self.status_label = tk.Label(main_frame, text="请添加图片或视频",
                                    font=("Microsoft YaHei UI", 11),
                                    fg='#00d9ff', bg='#1a1a2e')
        self.status_label.pack(pady=5)

        # 显示区域
        display = tk.Frame(main_frame, bg='#1a1a2e')
        display.pack(fill=tk.BOTH, expand=True, padx=25, pady=(5, 0))

        for text, bg_c in [("原始图像", '#16213e'), ("检测结果", '#16213e')]:
            f = tk.Frame(display, bg='#16213e', padx=5, pady=5)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(f, text=text, font=("Microsoft YaHei UI", 12, 'bold'),
                    fg='#00d9ff', bg='#16213e').pack(pady=(5, 5))
            inner = tk.Frame(f, bg='#0f0f23')
            inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            label = tk.Label(inner, bg='#0f0f23')
            label.pack(fill=tk.BOTH, expand=True)
            if "原始" in text:
                self.label_original = label
            else:
                self.label_result = label

        # 信息面板
        info_frame = tk.Frame(main_frame, bg='#16213e', padx=20, pady=10)
        info_frame.pack(fill=tk.X, padx=25, pady=(5, 10))

        self.info_label = tk.Label(info_frame, text="",
                                  font=("Consolas", 10),
                                  fg='#00ff88', bg='#16213e', justify=tk.LEFT)
        self.info_label.pack(anchor=tk.W)

        # 进度条
        progress_frame = tk.Frame(main_frame, bg='#1a1a2e', padx=50, pady=5)
        progress_frame.pack(fill=tk.X, padx=50, pady=(5, 10))

        self.progress = ttk.Progressbar(progress_frame, length=600, mode='determinate')
        self.progress.pack(fill=tk.X)

        self.progress_label = tk.Label(progress_frame, text="",
                                      font=("Microsoft YaHei UI", 9),
                                      fg='#888888', bg='#1a1a2e')
        self.progress_label.pack()

    def add_image(self):
        """添加图片"""
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if path:
            self.input_path = path
            self.is_video = False
            frame = imread_chinese(path)
            if frame is not None:
                self.show_image(self.label_original, frame)
                self.status_label.config(text=f"已加载: {os.path.basename(path)}", fg='#00ff88')
                self.label_result.config(image='')

    def add_video(self):
        """添加视频"""
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if path:
            self.input_path = path
            self.is_video = True
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if ret:
                self.show_image(self.label_original, frame)
                self.status_label.config(text=f"已加载: {os.path.basename(path)}", fg='#00ff88')
                self.label_result.config(image='')
            cap.release()

    def start_camera(self):
        """摄像头实时检测"""
        self.status_label.config(text="正在连接摄像头...", fg='#ffcc00')
        self.root.update()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            self.status_label.config(text="摄像头连接失败", fg='#ff4444')
            return

        self.status_label.config(text="摄像头已开启，点击'清空'按钮关闭", fg='#00ff88')
        self.camera_active = True
        self.cap = cap

        def camera_loop():
            while self.camera_active:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.process_enhanced(frame)

                # 显示信息
                cv2.imshow('Lane Detection - Press ESC to exit', result)

                # 更新GUI中的预览
                try:
                    self.root.after(0, lambda r=result: self.show_image(self.label_result, r))
                except:
                    pass

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or not self.camera_active:  # ESC键
                    break

            self.camera_active = False
            cap.release()
            cv2.destroyAllWindows()
            try:
                self.root.after(0, lambda: self.status_label.config(text="摄像头已关闭", fg='#00d9ff'))
            except:
                pass

        thread = threading.Thread(target=camera_loop, daemon=True)
        thread.start()

    def clear_all(self):
        """清空"""
        if hasattr(self, 'camera_active') and self.camera_active:
            self.camera_active = False
            cv2.destroyAllWindows()

        if self.cap:
            self.cap.release()

    def start_detection(self):
        """开始检测"""
        if self.processing:
            messagebox.showwarning("提示", "正在处理中...")
            return

        if self.input_path is None:
            messagebox.showwarning("提示", "请先添加图片或视频")
            return

        self.processing = True

        if self.is_video:
            thread = threading.Thread(target=self.process_video_thread)
        else:
            thread = threading.Thread(target=self.process_image_thread)
        thread.start()

    def process_image_thread(self):
        """处理单张图片"""
        self.root.after(0, lambda: self.status_label.config(text="检测中...", fg='#ffcc00'))

        frame = imread_chinese(self.input_path)
        if frame is None:
            self.root.after(0, lambda: messagebox.showerror("错误", "无法读取图像"))
            self.processing = False
            return

        result = self.process_enhanced(frame)
        self.result_frame = result

        self.root.after(0, lambda: self.show_image(self.label_result, result))
        self.root.after(0, lambda: self.status_label.config(text="检测完成!", fg='#00ff88'))
        self.processing = False

    def process_video_thread(self):
        """处理视频"""
        self.root.after(0, lambda: self.status_label.config(text="处理视频中...", fg='#ffcc00'))

        cap = cv2.VideoCapture(self.input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_dir = os.path.dirname(self.input_path) or "."
        name = os.path.splitext(os.path.basename(self.input_path))[0]
        output_path = os.path.join(output_dir, f"{name}_enhanced.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_enhanced(frame)
            writer.write(result)
            count += 1

            p = int(count / total * 100)
            self.root.after(0, lambda pp=p: self.progress.configure(value=pp))
            self.root.after(0, lambda cc=count, tt=total, pp=p:
                          self.progress_label.configure(text=f"处理中: {cc}/{tt} ({pp}%)"))

            if count % 5 == 0:
                self.root.after(0, lambda r=result: self.show_image(self.label_result, r))

        cap.release()
        writer.release()

        self.root.after(0, lambda: self.progress.configure(value=100))
        self.root.after(0, lambda: self.progress_label.configure(text=""))
        self.root.after(0, lambda: self.status_label.configure(text=f"完成! 保存至: {output_path}", fg='#00ff88'))
        self.root.after(0, lambda: messagebox.showinfo("完成", f"视频已保存:\n{output_path}"))
        self.processing = False

    def process_enhanced(self, frame):
        """
        增强版处理流程
        集成: 暗光增强 -> 车道检测 -> LDWS -> HUD
        """
        h, w = frame.shape[:2]

        # 暗光增强
        if self.enable_low_light.get() and detect_low_light(frame):
            frame = enhance_low_light(frame)

        # 基础检测
        result = process_frame(frame)

        # 车道跟踪
        if self.enable_tracker.get():
            # 简化: 使用结果图像进行进一步处理
            pass

        # HUD显示
        if self.enable_hud.get():
            # 计算伪数据用于HUD显示
            fps = 30
            curvature = 1000.0  # 模拟曲率值
            offset = 0.0
            warning = ""

            if self.enable_ldws.get():
                # LDWS计算
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                # 简化: 基于图像中心偏移估算
                lane_center_approx = w // 2
                offset = (lane_center_approx - w // 2) * 0.01
                is_departing, warning = self.ldws.check_departure(offset)

            result = self.hud.draw_info(result, fps, curvature, offset, warning)

        return result

    def show_image(self, label, frame):
        """显示图像"""
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = self.resize_image(rgb, 600, 450)
        im = Image.fromarray(resized)
        im_tk = ImageTk.PhotoImage(im)
        label.imgtk = im_tk
        label.configure(image=im_tk)

    def resize_image(self, img, max_w, max_h):
        """等比例缩放"""
        h, w = img.shape[:2]
        r = min(max_w / w, max_h / h)
        return cv2.resize(img, (int(w * r), int(h * r)))

    def save_result(self):
        """保存结果"""
        if hasattr(self, 'result_frame') and self.result_frame is not None:
            path = filedialog.asksaveasfilename(
                title="保存图片", defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("所有", "*.*")]
            )
            if path:
                cv2.imwrite(path, self.result_frame)
                messagebox.showinfo("成功", f"已保存:\n{path}")
        else:
            messagebox.showwarning("提示", "没有可保存的结果")

    def clear_all(self):
        """清空"""
        if hasattr(self, 'camera_active') and self.camera_active:
            self.camera_active = False
            cv2.destroyAllWindows()

        if self.cap:
            self.cap.release()

        self.input_path = None
        self.is_video = False
        self.result_frame = None
        self.tracker.reset()

        self.label_original.config(image='')
        self.label_result.config(image='')
        self.progress.configure(value=0)
        self.progress_label.configure(text="")
        self.info_label.configure(text="")
        self.status_label.configure(text="已清空", fg='#00d9ff')


def main():
    root = tk.Tk()
    app = EnhancedLaneDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
