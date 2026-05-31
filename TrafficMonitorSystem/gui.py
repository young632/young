# -*- coding: utf-8 -*-
"""
智能交通视觉监测系统 - GUI界面
修复视频播放问题、中文显示问题
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys
import cv2
from PIL import Image, ImageTk
import queue
import time

try:
    from traffic_monitor import TrafficMonitor
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from traffic_monitor import TrafficMonitor


class TrafficMonitorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("智能交通视觉监测系统")
        self.root.geometry("1200x750")
        self.root.configure(bg="#2c3e50")
        self.root.resizable(True, True)

        self.model_path = self.find_yolo_model()
        self.monitor = TrafficMonitor(speed_limit=60, fps=24, model_path=self.model_path)
        self.is_playing = False
        self.is_paused = False
        self.cap = None
        self.video_thread = None
        self.stop_flag = False
        self.pause_flag = False
        self.playback_error = False

        self.frame_queue = queue.Queue(maxsize=50)
        self.photo_left = None
        self.photo_right = None
        self.input_path = None
        self.is_camera = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30

        self.setup_ui()
    
    def find_yolo_model(self):
        """查找YOLO模型文件"""
        model_paths = [
            r"D:\计算机实践\yolov8n.pt",
            r"D:\计算机实践\yolov8n-seg.pt"
        ]
        for mp in model_paths:
            if os.path.exists(mp):
                print(f"找到YOLO模型: {mp}")
                return mp
        print("未找到YOLO模型，使用传统背景减法")
        return None

    def setup_ui(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        title_frame = tk.Frame(self.root, bg="#1a252f", height=50)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame, text="智能交通视觉监测系统",
            font=("SimHei", 18, "bold"), fg="#3498db", bg="#1a252f"
        ).pack(pady=10)

        control_frame = tk.Frame(self.root, bg="#2c3e50", padx=10, pady=10)
        control_frame.pack(fill="x")

        btn_style = {
            "font": ("SimHei", 12),
            "padx": 15,
            "pady": 8,
            "bd": 0,
            "relief": "raised"
        }

        self.add_video_btn = tk.Button(
            control_frame, text="添加视频",
            bg="#3498db", fg="white",
            command=self.add_video, **btn_style
        )
        self.add_video_btn.pack(side="left", padx=5)

        self.add_image_btn = tk.Button(
            control_frame, text="添加图片",
            bg="#9b59b6", fg="white",
            command=self.add_image, **btn_style
        )
        self.add_image_btn.pack(side="left", padx=5)

        self.camera_btn = tk.Button(
            control_frame, text="摄像头",
            bg="#8e44ad", fg="white",
            command=self.toggle_camera, **btn_style
        )
        self.camera_btn.pack(side="left", padx=5)

        self.play_btn = tk.Button(
            control_frame, text="播放",
            bg="#27ae60", fg="white",
            command=self.play, **btn_style
        )
        self.play_btn.pack(side="left", padx=5)

        self.pause_btn = tk.Button(
            control_frame, text="暂停",
            bg="#f39c12", fg="white",
            command=self.pause, state="disabled", **btn_style
        )
        self.pause_btn.pack(side="left", padx=5)

        self.stop_btn = tk.Button(
            control_frame, text="停止",
            bg="#e74c3c", fg="white",
            command=self.stop_all, **btn_style
        )
        self.stop_btn.pack(side="left", padx=5)

        self.reset_btn = tk.Button(
            control_frame, text="重置计数",
            bg="#7f8c8d", fg="white",
            command=self.reset_counter, **btn_style
        )
        self.reset_btn.pack(side="left", padx=5)

        self.heatmap_btn = tk.Button(
            control_frame, text="热力图",
            bg="#e67e22", fg="white",
            command=self.toggle_heatmap, **btn_style
        )
        self.heatmap_btn.pack(side="left", padx=5)

        self.export_btn = tk.Button(
            control_frame, text="导出数据",
            bg="#16a085", fg="white",
            command=self.export_data, **btn_style
        )
        self.export_btn.pack(side="left", padx=5)

        status_frame = tk.Frame(self.root, bg="#2c3e50", padx=10)
        status_frame.pack(fill="x")

        self.status_var = tk.StringVar()
        self.status_var.set("就绪 - 请添加视频或打开摄像头")
        self.status_label = tk.Label(
            status_frame, textvariable=self.status_var,
            font=("SimHei", 10), fg="#ecf0f1", bg="#2c3e50"
        )
        self.status_label.pack(side="left")

        self.stats_frame = tk.Frame(status_frame, bg="#2c3e50")
        self.stats_frame.pack(side="left", padx=30)

        self.stats_labels = []
        stats_items = ["车流量", "轿车", "货车", "公交", "均速", "拥堵"]
        for item in stats_items:
            label = tk.Label(self.stats_frame, text=f"{item}: -", 
                            font=("SimHei", 9), fg="#95a5a6", bg="#2c3e50")
            label.pack(side="left", padx=15)
            self.stats_labels.append(label)

        self.progress_var = tk.StringVar()
        self.progress_var.set("进度: 0%")
        self.progress_label = tk.Label(
            status_frame, textvariable=self.progress_var,
            font=("SimHei", 10), fg="#3498db", bg="#2c3e50"
        )
        self.progress_label.pack(side="right")

        display_frame = tk.Frame(self.root, bg="#1a252f", padx=10, pady=10)
        display_frame.pack(fill="both", expand=True)

        main_content_frame = tk.Frame(display_frame, bg="#1a252f")
        main_content_frame.pack(fill="both", expand=True)

        left_frame = tk.Frame(main_content_frame, bg="#2c3e50")
        left_frame.pack(side="left", fill="both", expand=True)

        tk.Label(
            left_frame, text="原始画面",
            font=("SimHei", 12), fg="#ecf0f1", bg="#2c3e50"
        ).pack(pady=5)

        self.left_canvas = tk.Canvas(left_frame, bg="#1a252f")
        self.left_canvas.pack(fill="both", expand=True)

        right_frame = tk.Frame(main_content_frame, bg="#2c3e50")
        right_frame.pack(side="right", fill="both", expand=True)

        tk.Label(
            right_frame, text="检测结果",
            font=("SimHei", 12), fg="#ecf0f1", bg="#2c3e50"
        ).pack(pady=5)

        self.right_canvas = tk.Canvas(right_frame, bg="#1a252f")
        self.right_canvas.pack(fill="both", expand=True)

    def add_video(self):
        self.stop_all()
        
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("所有文件", "*.*")]
        )
        
        if path:
            self.monitor.reset()
            self.input_path = path
            self.is_camera = False
            self.current_frame = 0
            self.playback_error = False
            
            try:
                temp_cap = cv2.VideoCapture(path)
                if temp_cap.isOpened():
                    self.total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.fps = int(temp_cap.get(cv2.CAP_PROP_FPS))
                    if self.fps <= 0 or self.fps > 100:
                        self.fps = 30
                    temp_cap.release()
                else:
                    self.total_frames = 0
                    self.fps = 30
            except Exception as e:
                print(f"Error reading video info: {e}")
                self.total_frames = 0
                self.fps = 30
            
            self.status_var.set("已加载: {}".format(os.path.basename(path)))
            self.show_first_frame()

    def add_image(self):
        self.stop_all()
        
        path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        
        if path:
            self.monitor.reset()
            self.input_path = path
            self.is_camera = False
            self.status_var.set("已加载: {}".format(os.path.basename(path)))
            
            frame = cv2.imread(path)
            if frame is not None:
                result, info = self.monitor.process_frame(frame)
                self.show_frames(frame, result)
            else:
                messagebox.showerror("错误", "无法读取图片文件！")

    def toggle_camera(self):
        if self.is_camera:
            self.stop_all()
            self.is_camera = False
            self.status_var.set("摄像头已关闭")
            self.camera_btn.config(bg="#8e44ad")
        else:
            self.stop_all()
            self.monitor.reset()
            
            try:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            except:
                try:
                    self.cap = cv2.VideoCapture(0)
                except:
                    messagebox.showerror("错误", "无法打开摄像头！")
                    return
            
            if self.cap.isOpened():
                self.is_camera = True
                self.input_path = None
                self.total_frames = 0
                self.fps = 30
                self.status_var.set("摄像头已启动")
                self.camera_btn.config(bg="#e74c3c")
                self.play()
            else:
                messagebox.showerror("错误", "无法打开摄像头！")
                self.cap = None

    def show_first_frame(self):
        if not self.input_path:
            return
        
        try:
            cap = cv2.VideoCapture(self.input_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.show_frames(frame, frame)
                cap.release()
        except Exception as e:
            self.status_var.set("错误: {}".format(str(e)))

    def play(self):
        if self.is_playing and not self.is_paused:
            return

        if self.is_paused:
            self.is_paused = False
            self.pause_flag = False
            self.play_btn.config(state="disabled")
            self.pause_btn.config(state="normal")
            self.status_var.set("继续播放中...")
            return

        if self.is_camera and self.cap is not None:
            self.start_playback()
        elif self.input_path and os.path.exists(self.input_path):
            try:
                self.cap = cv2.VideoCapture(self.input_path)
                if self.cap.isOpened():
                    self.start_playback()
                else:
                    messagebox.showerror("错误", "无法打开视频文件！")
            except Exception as e:
                messagebox.showerror("错误", f"打开视频失败: {str(e)}")
        else:
            messagebox.showwarning("警告", "请先添加视频或打开摄像头！")

    def pause(self):
        if not self.is_playing:
            return
        
        self.is_paused = True
        self.pause_flag = True
        self.play_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.status_var.set("已暂停")

    def start_playback(self):
        self.stop_flag = False
        self.playback_error = False
        self.is_playing = True
        self.play_btn.config(state="disabled")
        self.pause_btn.config(state="normal")

        if self.video_thread and self.video_thread.is_alive():
            self.stop_flag = True
            try:
                self.video_thread.join(timeout=2.0)
            except:
                pass

        self.video_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.video_thread.start()
        self.update_display()

    def process_loop(self):
        frame_count = self.current_frame
        frame_delay = 1.0 / self.fps
        last_time = time.time()
        frame_skip = 0
        
        if not self.is_camera and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        while not self.stop_flag and self.cap is not None:
            while self.pause_flag and not self.stop_flag:
                time.sleep(0.01)
            
            if self.stop_flag:
                break

            try:
                ret, frame = self.cap.read()
                if not ret:
                    if self.is_camera:
                        time.sleep(0.03)
                        continue
                    else:
                        break

                frame_count += 1
                frame_skip += 1
                
                if frame_skip % 1 == 0:
                    result, info = self.monitor.process_frame(frame)

                    while self.frame_queue.qsize() > 25:
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    
                    self.frame_queue.put((frame.copy(), result.copy(), frame_count))

                self.current_frame = frame_count

                if not self.is_camera and self.total_frames > 0:
                    progress = min(100, int((frame_count / self.total_frames) * 100))
                    self.root.after(0, lambda p=progress: self.progress_var.set(f"进度: {p}%"))

                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                last_time = current_time
                
            except Exception as e:
                print(f"Playback error: {e}")
                self.playback_error = True
                self.root.after(0, lambda err=str(e): self.status_var.set(f"播放错误: {err}"))
                break

        if not self.is_camera and self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        if not self.stop_flag and not self.playback_error:
            self.is_playing = False
            self.play_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.root.after(0, lambda: self.status_var.set("播放完成"))
            self.root.after(0, lambda: self.progress_var.set("进度: 100%"))

    def update_display(self):
        try:
            while not self.frame_queue.empty():
                frame, result, count = self.frame_queue.get_nowait()
                self.show_frames(frame, result)
                stats = self.monitor.get_statistics()
                status_text = f"检测中 - 帧:{count}, 车流量:{stats['total_count']}"
                self.root.after(0, lambda t=status_text: self.status_var.set(t))
                
                self.update_stats_panel(stats)
        except Exception as e:
            pass

        if self.is_playing and not self.playback_error:
            self.root.after(10, self.update_display)
    
    def update_stats_panel(self, stats):
        try:
            if len(self.stats_labels) >= 6:
                self.stats_labels[0].config(text=f"总车流量: {stats['total_count']}")
                self.stats_labels[1].config(text=f"轿车: {stats['vehicle_counts'].get(0, 0)}")
                self.stats_labels[2].config(text=f"货车: {stats['vehicle_counts'].get(1, 0)}")
                self.stats_labels[3].config(text=f"公交: {stats['vehicle_counts'].get(2, 0)}")
                self.stats_labels[4].config(text=f"均速: {stats['avg_speed']:.0f} km/h")
                self.stats_labels[5].config(text=f"拥堵: {stats['congestion_index']:.0f}%")
        except Exception as e:
            print(f"Stats panel error: {e}")

    def show_frames(self, original, processed):
        try:
            h, w = original.shape[:2]
            canvas_w = self.left_canvas.winfo_width()
            canvas_h = self.left_canvas.winfo_height()

            if canvas_w < 10 or canvas_h < 10:
                canvas_w, canvas_h = 540, 380

            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            orig_rgb = cv2.cvtColor(cv2.resize(original, (new_w, new_h)), cv2.COLOR_BGR2RGB)
            proc_rgb = cv2.cvtColor(cv2.resize(processed, (new_w, new_h)), cv2.COLOR_BGR2RGB)

            self.photo_left = ImageTk.PhotoImage(image=Image.fromarray(orig_rgb))
            self.photo_right = ImageTk.PhotoImage(image=Image.fromarray(proc_rgb))

            x_off = (canvas_w - new_w) // 2
            y_off = (canvas_h - new_h) // 2

            self.left_canvas.delete("all")
            self.right_canvas.delete("all")
            self.left_canvas.create_image(x_off, y_off, anchor=tk.NW, image=self.photo_left)
            self.right_canvas.create_image(x_off, y_off, anchor=tk.NW, image=self.photo_right)
        except Exception as e:
            print(f"Display error: {e}")

    def reset_counter(self):
        self.monitor.reset()
        self.status_var.set("计数器已重置")

    def toggle_heatmap(self):
        self.monitor.show_heatmap = not self.monitor.show_heatmap
        if self.monitor.show_heatmap:
            self.heatmap_btn.config(bg="#d35400")
            self.status_var.set("热力图已开启")
        else:
            self.heatmap_btn.config(bg="#e67e22")
            self.status_var.set("热力图已关闭")

    def export_data(self):
        try:
            self.monitor.export_data()
            self.status_var.set("数据已导出到CSV和JSON文件")
            messagebox.showinfo("成功", "数据已导出:\n- traffic_log.csv\n- traffic_summary.json")
        except Exception as e:
            self.status_var.set(f"导出失败: {str(e)}")
            messagebox.showerror("错误", f"导出失败:\n{str(e)}")

    def stop_all(self):
        self.stop_flag = True
        self.pause_flag = False
        
        if self.video_thread and self.video_thread.is_alive():
            try:
                self.video_thread.join(timeout=2.0)
            except:
                pass
        
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        self.is_playing = False
        self.is_paused = False
        self.play_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.status_var.set("已停止")

    def on_closing(self):
        self.stop_all()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TrafficMonitorGUI()
    app.run()
