# -*- coding: utf-8 -*-
"""用test_client测试视频流"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

app = create_app()

# 测试视频流
with app.test_client() as client:
    print("=== 测试视频流 ===")
    
    # 设置更大的超时
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    
    try:
        # 发送请求
        response = client.get('/video_feed?file=traffic_rideo.mp4&mode=normal', 
                             follow_redirects=True)
        
        print(f"状态码: {response.status_code}")
        print(f"Content-Type: {response.content_type}")
        
        if response.status_code != 200:
            print(f"响应数据长度: {len(response.data)}")
            print(f"响应数据: {response.data[:1000]}")
            
    except Exception as e:
        print(f"请求异常: {e}")
        import traceback
        traceback.print_exc()