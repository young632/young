# -*- coding: utf-8 -*-
"""测试原始服务器错误"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

app = create_app()

# 测试视频流
with app.test_client() as client:
    print("=== 测试视频流 ===")
    
    try:
        response = client.get('/video_feed?file=traffic_rideo.mp4&mode=normal')
        print(f"状态码: {response.status_code}")
        print(f"Content-Type: {response.content_type}")
        
        if response.status_code != 200:
            print(f"响应数据: {response.data[:500]}")
            
    except Exception as e:
        print(f"请求异常: {e}")
        import traceback
        traceback.print_exc()