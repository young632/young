# -*- coding: utf-8 -*-
"""详细调试视频流错误"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

app = create_app()

# 启用详细错误信息
app.config['PROPAGATE_EXCEPTIONS'] = True

# 测试视频流
with app.test_request_context():
    print("=== 测试视频流 ===")
    
    # 模拟请求上下文
    from flask import request
    
    # 手动设置请求参数
    class FakeArgs:
        def get(self, key, default=None):
            if key == 'file':
                return 'traffic_rideo.mp4'
            elif key == 'mode':
                return 'normal'
            return default
    
    request.args = FakeArgs()
    
    # 直接调用视频流函数
    from app.routes import video_feed
    
    try:
        response = video_feed()
        print(f"响应类型: {type(response)}")
        print(f"成功!")
    except Exception as e:
        print(f"异常: {e}")
        import traceback
        traceback.print_exc()