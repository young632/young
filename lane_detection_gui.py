#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车道线检测GUI界面 - 稳定版
功能: LDWS预警 | HUD显示 | 暗光增强 | 图片/视频处理
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
from core.enhance import enhance_low_light, detect_low_light
from features.ldws import LDWS
from features.hud import HUD


class LaneDetectionApp:
    """车道线检测应用"""

    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV车道线检测系统")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')

        self.input_path = None
        self.is_video = False
        self.processing = False
        self.result_frame = None

        self.ldws = LDWS()
        self.hud = HUD()

        self.enable_ldws = tk.BooleanVar(value=True)
        self.enable_hud = tk.BooleanVar(value=True)
        self.enable_low_light = tk.BooleanVar(value=False)

        self.setup_ui()

    def setup_ui(self):
        main = tk.Frame(self.root, bg='#1a1a2e')
        main.pack(fill=tk.BOTH, expand=True)

        # 标题
        tk.Label(main, text="车道线检测系统",
                font=("Microsoft YaHei", 22, "bold"),
                fg='#00d9ff', bg='#1a1a2e').pack(pady=15)

        # 功能选项
        opt = tk.Frame(main, bg='#16213e', padx=15, pady=10)
        opt.pack(fill=tk.X, padx=20, pady=(0, 10))

        for txt, var in [
            ("车道偏离预警", self.enable_ldws),
            ("数据显示", self.enable_hud),
            ("暗光增强", self.enable_low_light),
        ]:
            tk.Checkbutton(opt, text=txt, variable=var,
                          bg='#16213e', fg='#00ff88',
                          selectcolor='#0f3460',
                          font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=15)

        # 按钮
        btn = tk.Frame(main, bg='#16213e', padx=15, pady=10)
        btn.pack(fill=tk.X, padx=20, pady=(0, 10))

        for txt, cmd, bg in [
            ("添加图片", self.add_image, '#0f3460'),
            ("添加视频", self.add_video, '#0f3460'),
            ("摄像头", self.start_camera, '#0f3460'),
            ("开始检测", self.start_detection, '#e94560'),
            ("保存", self.save_result, '#0f3460'),
            ("清空", self.clear_all, '#16213e'),
        ]:
            tk.Button(btn, text=txt, command=cmd,
                     font=("Microsoft YaHei", 10, 'bold'),
                     bg=bg, fg='white', relief='flat',
                     padx=15, pady=6).pack(side=tk.LEFT, padx=5)

        # 状态
        self.status = tk.Label(main, text="请添加图片或视频",
                              font=("Microsoft YaHei", 11),
                              fg='#00d9ff', bg='#1a1a2e')
        self.status.pack(pady=5)

        # 显示区
        disp = tk.Frame(main, bg='#1a1a2e')
        disp.pack(fill=tk.BOTH, expand=True, padx=20)

        for txt, idx in [("原始", 0), ("结果", 1)]:
            f = tk.Frame(disp, bg='#16213e', padx=5, pady=5)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(f, text=txt, font=("Microsoft YaHei", 12),
                    fg='#00d9ff', bg='#16213e').pack(pady=5)
            inner = tk.Frame(f, bg='#0f0f23')
            inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            lbl = tk.Label(inner, bg='#0f0f23')
            lbl.pack(fill=tk.BOTH, expand=True)
            if idx == 0:
                self.label_orig = lbl
            else:
                self.label_res = lbl

        # 进度
        self.progress = ttk.Progressbar(main, length=500, mode='determinate')
        self.progress.pack(pady=(10, 5))
        self.prog_lbl = tk.Label(main, text="", font=("Microsoft YaHei", 9),
                                fg='#888888', bg='#1a1a2e')
        self.prog_lbl.pack()

        self.camera_active = False

    def start_camera(self):
        """摄像头实时检测"""
        if self.camera_active:
            messagebox.showwarning("提示", "摄像头已在运行")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return

        self.camera_active = True
        self.cap = cap
        self.status.config(text="摄像头已开启，点击'清空'按钮关闭", fg='#00ff88')

        def camera_loop():
             while self.camera_active:
                 ret, frame = self.cap.read()
                 if not ret:
                     break

                 result = self.process_result(frame)

                 try:
                     self.root.after(0, lambda r=result: self.show_img(self.label_res, r))
                 except:
                     pass

                 import time
                 time.sleep(0.03)

             if self.camera_active:
                 self.cap.release()
                 cv2.destroyAllWindows()

        thread = threading.Thread(target=camera_loop, daemon=True)
        thread.start()

    def add_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", "*.jpg *.png *.bmp"), ("所有", "*.*")]
        )
        if path:
            self.input_path = path
            self.is_video = False
            frame = imread_chinese(path)
            if frame is not None:
                self.show_img(self.label_orig, frame)
                self.label_res.config(image='')
                self.status.config(text=f"已加载: {os.path.basename(path)}", fg='#00ff88')

    def add_video(self):
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频", "*.mp4 *.avi *.mov"), ("所有", "*.*")]
        )
        if path:
            self.input_path = path
            self.is_video = True
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if ret:
                self.show_img(self.label_orig, frame)
                self.label_res.config(image='')
                self.status.config(text=f"已加载: {os.path.basename(path)}", fg='#00ff88')
            cap.release()

    def start_detection(self):
        if self.processing:
            messagebox.showwarning("提示", "正在处理中")
            return
        if self.input_path is None:
            messagebox.showwarning("提示", "请先添加文件")
            return

        self.processing = True
        thread = threading.Thread(target=self.process_thread)
        thread.daemon = True
        thread.start()

    def process_thread(self):
        try:
            if self.is_video:
                self.process_video()
            else:
                self.process_image()
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.processing = False

    def process_image(self):
        self.root.after(0, lambda: self.status.config(text="处理中...", fg='#ffcc00'))

        frame = imread_chinese(self.input_path)
        if frame is None:
            self.root.after(0, lambda: messagebox.showerror("错误", "无法读取"))
            return

        result = self.process_result(frame)
        self.result_frame = result

        self.root.after(0, lambda: self.show_img(self.label_res, result))
        self.root.after(0, lambda: self.status.config(text="完成!", fg='#00ff88'))

    def process_video(self):
        self.root.after(0, lambda: self.status.config(text="处理视频中...", fg='#ffcc00'))

        cap = cv2.VideoCapture(self.input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out_dir = os.path.dirname(self.input_path) or "."
        name = os.path.splitext(os.path.basename(self.input_path))[0]
        out_path = os.path.join(out_dir, f"{name}_result.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_result(frame)
            writer.write(result)
            cnt += 1

            p = int(cnt / total * 100)
            self.root.after(0, lambda pp=p: self.progress.configure(value=pp))
            self.root.after(0, lambda c=cnt, t=total, pp=p:
                          self.prog_lbl.configure(text=f"{c}/{t} ({pp}%)"))

            if cnt % 10 == 0:
                self.root.after(0, lambda r=result: self.show_img(self.label_res, r))

        cap.release()
        writer.release()

        self.root.after(0, lambda: self.progress.configure(value=100))
        self.root.after(0, lambda: self.prog_lbl.configure(text=""))
        self.root.after(0, lambda: self.status.configure(text=f"已保存: {out_path}", fg='#00ff88'))
        self.root.after(0, lambda: messagebox.showinfo("完成", f"视频已保存:\n{out_path}"))

    def process_result(self, frame):
        """处理单帧"""
        if self.enable_low_light.get() and detect_low_light(frame):
            frame = enhance_low_light(frame)

        result, offset, _, _ = process_frame(frame)

        if self.enable_hud.get():
            warning = ""
            if self.enable_ldws.get():
                is_dep, warning = self.ldws.check_departure(offset)
            result = self.hud.draw_info(result, fps=30, curvature=0, offset=offset, warning=warning)

        return result

    def show_img(self, label, frame):
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = self.resize(rgb, 500, 400)
        im = Image.fromarray(resized)
        im_tk = ImageTk.PhotoImage(im)
        label.imgtk = im_tk
        label.configure(image=im_tk)

    def resize(self, img, max_w, max_h):
        h, w = img.shape[:2]
        r = min(max_w / w, max_h / h)
        return cv2.resize(img, (int(w * r), int(h * r)))

    def save_result(self):
        if self.result_frame is not None:
            path = filedialog.asksaveasfilename(
                title="保存", defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
            )
            if path:
                cv2.imwrite(path, self.result_frame)
                messagebox.showinfo("成功", f"已保存:\n{path}")
        else:
            messagebox.showwarning("提示", "没有结果")

    def clear_all(self):
        if self.camera_active:
            self.camera_active = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

        self.input_path = None
        self.is_video = False
        self.result_frame = None
        self.label_orig.config(image='')
        self.label_res.config(image='')
        self.progress.configure(value=0)
        self.prog_lbl.configure(text="")
        self.status.configure(text="已清空", fg='#00d9ff')


def main():
    root = tk.Tk()
    LaneDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
