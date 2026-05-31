# -*- coding: utf-8 -*-
"""
完整测试脚本 - 处理视频并保存结果
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from traffic_monitor import TrafficMonitor

print('=' * 60)
print('Traffic Monitor System - Full Test')
print('=' * 60)
print()

# 初始化系统
print('1. Initializing TrafficMonitor...', end='')
monitor = TrafficMonitor(show_lane=True, show_heatmap=True, show_speed=True)
print(' OK')
print()

# 检查视频
test_video = r'D:\计算机实践\solidWhiteRight.mp4'
if not os.path.exists(test_video):
    print('Error: Test video not found!')
    print('Expected:', test_video)
    sys.exit(1)

print('2. Opening video:', os.path.basename(test_video))
cap = cv2.VideoCapture(test_video)
if not cap.isOpened():
    print('Error: Cannot open video!')
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print('   Total frames:', total_frames)
print('   FPS:', fps)
print()

# 准备输出
output_path = r'D:\计算机实践\TrafficMonitorSystem\test_output.avi'
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 处理前30帧
print('3. Processing first 30 frames...')
print()

frame_count = 0
save_preview = True

while cap.isOpened() and frame_count < 30:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理
    result, info = monitor.process_frame(frame)
    
    # 保存
    out.write(result)
    
    # 保存预览图
    if save_preview and frame_count == 15:
        cv2.imwrite(r'D:\计算机实践\TrafficMonitorSystem\preview_frame.jpg', result)
        print('   Saved preview at frame 15')
        save_preview = False
    
    # 更新
    frame_count += 1
    if frame_count % 5 == 0:
        stats = monitor.get_statistics()
        print('   Processed', frame_count, '/30 | Count:', stats['total_count'])

cap.release()
out.release()

# 显示结果
print()
print('=' * 60)
print('Test completed successfully!')
print('=' * 60)
print()
print('Output files:')
print(' - Video:', output_path)
print(' - Preview:', r'D:\计算机实践\TrafficMonitorSystem\preview_frame.jpg')
print()

final_stats = monitor.get_statistics()
print('Final statistics:')
print(' - Total vehicles counted:', final_stats['total_count'])
print(' - Frames processed:', final_stats['frame_count'])
print(' - Average speed:', '{:.1f}'.format(final_stats['avg_speed']), 'km/h')
print()
print('Next step: On your computer, run GUI:')
print('   cd "d:\\计算机实践\\TrafficMonitorSystem"')
print('   python gui.py')
print()