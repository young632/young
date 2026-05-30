#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车道线检测GUI界面 - 现代化设计
支持图片和视频输入，实时显示检测结果
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lane_detection_complete import process_frame, imread_chinese


class LaneDetectionGUI:
    """车道线检测图形界面 - 现代化设计"""

    def __init__(self, root):
        self.root = root
        self.root.title("🚗 OpenCV 车道线检测系统")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg='#1a1a2e')

        self.input_path = None
        self.output_path = None
        self.is_video = False
        self.cap = None
        self.video_writer = None
        self.current_frame = None
        self.processing = False

        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        """设置现代化样式"""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Title.TLabel',
                       background='#1a1a2e',
                       foreground='white',
                       font=('Microsoft YaHei UI', 28, 'bold'))

        style.configure('Status.TLabel',
                       background='#16213e',
                       foreground='#e94560',
                       font=('Microsoft YaHei UI', 12))

        style.configure('Modern.TButton',
                       font=('Microsoft YaHei UI', 11, 'bold'),
                       padding=(20, 10))

        style.configure('Modern.Horizontal.TProgressbar',
                       thickness=8,
                       borderwidth=0)

    def setup_ui(self):
        """设置界面布局"""
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = tk.Frame(main_frame, bg='#1a1a2e', height=100)
        header_frame.pack(fill=tk.X, padx=30, pady=(20, 10))

        title_label = tk.Label(
            header_frame,
            text="🚗 OpenCV 车道线检测系统",
            font=("Microsoft YaHei UI", 28, "bold"),
            fg='#00d9ff',
            bg='#1a1a2e'
        )
        title_label.pack(side=tk.LEFT)

        subtitle = tk.Label(
            header_frame,
            text="基于传统计算机视觉的车道线检测",
            font=("Microsoft YaHei UI", 12),
            fg='#888888',
            bg='#1a1a2e'
        )
        subtitle.pack(side=tk.LEFT, padx=20, pady=15)

        btn_frame = tk.Frame(main_frame, bg='#16213e', padx=20, pady=15)
        btn_frame.pack(fill=tk.X, padx=30, pady=(10, 5))

        buttons = [
            ("📷 添加图片", self.add_image, '#0f3460'),
            ("🎬 添加视频", self.add_video, '#0f3460'),
            ("▶️ 开始检测", self.start_detection, '#e94560'),
            ("💾 保存结果", self.save_result, '#0f3460'),
            ("🔄 清空", self.clear_all, '#16213e')
        ]

        for i, (text, cmd, bg) in enumerate(buttons):
            btn = tk.Button(
                btn_frame,
                text=text,
                command=cmd,
                font=("Microsoft YaHei UI", 11, 'bold'),
                bg=bg,
                fg='white',
                activebackground='#00d9ff',
                activeforeground='white',
                relief='flat',
                padx=20,
                pady=12,
                cursor='hand2'
            )
            btn.pack(side=tk.LEFT, padx=8)

        self.status_label = tk.Label(
            main_frame,
            text="✨ 请添加图片或视频文件开始检测",
            font=("Microsoft YaHei UI", 12),
            fg='#00d9ff',
            bg='#1a1a2e'
        )
        self.status_label.pack(pady=8)

        display_container = tk.Frame(main_frame, bg='#1a1a2e', padx=30)
        display_container.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        left_frame = tk.Frame(display_container, bg='#16213e', padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            left_frame,
            text="📷 原始图像",
            font=("Microsoft YaHei UI", 14, 'bold'),
            fg='#00d9ff',
            bg='#16213e'
        ).pack(pady=(5, 10))

        left_inner = tk.Frame(left_frame, bg='#0f0f23')
        left_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.label_original = tk.Label(left_inner, bg='#0f0f23')
        self.label_original.pack(fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(display_container, bg='#16213e', padx=10, pady=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 0))

        tk.Label(
            right_frame,
            text="🎯 检测结果",
            font=("Microsoft YaHei UI", 14, 'bold'),
            fg='#00ff88',
            bg='#16213e'
        ).pack(pady=(5, 10))

        right_inner = tk.Frame(right_frame, bg='#0f0f23')
        right_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.label_result = tk.Label(right_inner, bg='#0f0f23')
        self.label_result.pack(fill=tk.BOTH, expand=True)

        progress_container = tk.Frame(main_frame, bg='#1a1a2e', padx=50, pady=15)
        progress_container.pack(fill=tk.X)

        self.progress = ttk.Progressbar(
            progress_container,
            style='Modern.Horizontal.TProgressbar',
            length=800,
            mode='determinate'
        )
        self.progress.pack(fill=tk.X)

        self.progress_label = tk.Label(
            progress_container,
            text="",
            font=("Microsoft YaHei UI", 10),
            fg='#888888',
            bg='#1a1a2e'
        )
        self.progress_label.pack(pady=(5, 0))

        info_frame = tk.Frame(main_frame, bg='#16213e', padx=20, pady=10)
        info_frame.pack(fill=tk.X, padx=30, pady=(5, 15))

        info_text = "📌 参数说明: Canny(50,150) | HSL白[0,200,0]-[255,255,255] | HSL黄[15,40,100]-[35,255,255] | 霍夫变换"
        tk.Label(
            info_frame,
            text=info_text,
            font=("Microsoft YaHei UI", 9),
            fg='#666666',
            bg='#16213e'
        ).pack()

    def add_image(self):
        """添加图片文件"""
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )
        if path:
            self.input_path = path
            self.is_video = False
            self.show_original_image(path)
            self.status_label.config(text=f"✅ 已加载图片: {os.path.basename(path)}", fg='#00ff88')
            self.label_result.config(image='')

    def add_video(self):
        """添加视频文件"""
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*")
            ]
        )
        if path:
            self.input_path = path
            self.is_video = True
            self.cap = cv2.VideoCapture(path)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.show_original_image(frame)
                self.cap.release()
            self.status_label.config(text=f"✅ 已加载视频: {os.path.basename(path)}", fg='#00ff88')
            self.label_result.config(image='')

    def show_original_image(self, path_or_frame):
        """显示原始图像"""
        if isinstance(path_or_frame, str):
            frame = imread_chinese(path_or_frame)
        else:
            frame = path_or_frame

        if frame is None:
            messagebox.showerror("错误", "无法读取图像")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = self.resize_image(frame_rgb, 550, 400)
        im = Image.fromarray(frame_resized)
        im_tk = ImageTk.PhotoImage(im)
        self.label_original.imgtk = im_tk
        self.label_original.configure(image=im_tk)

    def resize_image(self, img, max_width, max_height):
        """等比例缩放图像"""
        h, w = img.shape[:2]
        ratio = min(max_width / w, max_height / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        return cv2.resize(img, (new_w, new_h))

    def start_detection(self):
        """开始车道线检测"""
        if self.processing:
            messagebox.showwarning("提示", "正在处理中，请稍候...")
            return

        if self.input_path is None:
            messagebox.showwarning("提示", "请先添加图片或视频文件")
            return

        self.processing = True

        if self.is_video:
            thread = threading.Thread(target=self.process_video_thread)
            thread.start()
        else:
            thread = threading.Thread(target=self.process_image_thread)
            thread.start()

    def process_image_thread(self):
        """处理单张图片（后台线程）"""
        self.root.after(0, lambda: self.status_label.config(text="⏳ 正在检测...", fg='#ffcc00'))

        frame = imread_chinese(self.input_path)
        if frame is None:
            self.root.after(0, lambda: messagebox.showerror("错误", "无法读取图像"))
            self.processing = False
            return

        result = process_frame(frame)
        self.result_frame = result

        self.root.after(0, lambda: self.show_result_image(result))
        self.root.after(0, lambda: self.status_label.config(text="✅ 检测完成！", fg='#00ff88'))
        self.processing = False

    def process_video_thread(self):
        """处理视频（后台线程）"""
        self.root.after(0, lambda: self.status_label.config(text="⏳ 正在处理视频...", fg='#ffcc00'))

        self.cap = cv2.VideoCapture(self.input_path)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        output_dir = os.path.dirname(self.input_path) or "."
        base_name = os.path.splitext(os.path.basename(self.input_path))[0]
        self.output_path = os.path.join(output_dir, f"{base_name}_detected.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        self.cap.release()
        self.cap = cv2.VideoCapture(self.input_path)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            result = process_frame(frame)
            self.video_writer.write(result)

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)

            self.root.after(0, lambda p=progress, fc=frame_count, tf=total_frames, self=self:
                          self._update_progress(p, fc, tf))

            if frame_count % 5 == 0:
                self.root.after(0, lambda r=result, self=self: self.show_result_image(r))

        self.cap.release()
        self.video_writer.release()

        self.root.after(0, lambda: self.progress.configure(value=100))
        self.root.after(0, lambda: self.progress_label.configure(text=""))
        self.root.after(0, lambda: self.status_label.configure(
            text=f"✅ 视频检测完成！已保存至: {os.path.basename(self.output_path)}", fg='#00ff88'))
        self.root.after(0, lambda: messagebox.showinfo("完成", f"视频已保存至:\n{self.output_path}"))
        self.processing = False

    def _update_progress(self, progress_val, frame_count, total_frames):
        """更新进度条"""
        self.progress['value'] = progress_val
        self.progress_label.config(text=f"处理中: {frame_count}/{total_frames} 帧 ({progress_val}%)")

    def show_result_image(self, frame):
        """显示检测结果图像"""
        if frame is None:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = self.resize_image(frame_rgb, 550, 400)
        im = Image.fromarray(frame_resized)
        im_tk = ImageTk.PhotoImage(im)
        self.label_result.imgtk = im_tk
        self.label_result.configure(image=im_tk)

    def save_result(self):
        """保存结果"""
        if hasattr(self, 'result_frame') and self.result_frame is not None:
            path = filedialog.asksaveasfilename(
                title="保存图片",
                defaultextension=".jpg",
                filetypes=[
                    ("JPEG图片", "*.jpg"),
                    ("PNG图片", "*.png"),
                    ("所有文件", "*.*")
                ]
            )
            if path:
                cv2.imwrite(path, self.result_frame)
                messagebox.showinfo("成功", f"图片已保存至:\n{path}")
        elif self.output_path and os.path.exists(self.output_path):
            messagebox.showinfo("提示", f"视频已保存至:\n{self.output_path}")
        else:
            messagebox.showwarning("提示", "没有可保存的检测结果")

    def clear_all(self):
        """清空所有"""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()

        self.input_path = None
        self.output_path = None
        self.is_video = False
        self.current_frame = None
        if hasattr(self, 'result_frame'):
            self.result_frame = None

        self.label_original.config(image='')
        self.label_result.config(image='')
        self.progress['value'] = 0
        self.progress_label.config(text="")
        self.status_label.config(text="✨ 请添加图片或视频文件开始检测", fg='#00d9ff')


def main():
    """主函数"""
    root = tk.Tk()
    app = LaneDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
