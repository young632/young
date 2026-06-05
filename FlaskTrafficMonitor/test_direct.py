# -*- coding: utf-8 -*-
"""直接测试视频流生成"""

import cv2
import time

def test_direct():
    print("=== 直接测试视频流 ===")
    
    video_path = r"D:\计算机实践\FlaskTrafficMonitor\uploads\traffic_rideo.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("✗ 无法打开视频")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = 1.0 / fps
    print(f"帧率: {fps}, 帧间隔: {frame_interval:.4f}s")
    
    for i in range(10):
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            print("✗ 无法读取帧")
            break
        
        # 编码
        ret_jpg, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        if not ret_jpg:
            print("✗ 编码失败")
            break
        
        frame_bytes = jpeg.tobytes()
        print(f"帧 {i+1}: {len(frame_bytes)} bytes")
        
        # 模拟流输出
        stream_data = (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
    
    cap.release()
    print("\n✓ 测试完成")

if __name__ == "__main__":
    test_direct()
