# -*- coding: utf-8 -*-
"""测试视频流API"""

import requests

def test_video_feed():
    print("=== 测试视频流API ===")
    
    try:
        url = 'http://localhost:5000/video_feed?file=traffic_rideo.mp4&mode=normal'
        print(f"请求URL: {url}")
        
        r = requests.get(url, stream=True, timeout=30)
        print(f"状态码: {r.status_code}")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
        
        if r.status_code == 200:
            data = b''
            count = 0
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    data += chunk
                    count += 1
                    if count > 10:
                        break
            
            print(f"收到数据: {len(data)} 字节")
            print(f"数据开头: {data[:50]}")
            
            if b'--frame' in data:
                print("✓ 数据包含MJPEG边界标记")
            else:
                print("✗ 数据不包含MJPEG边界标记")
        else:
            print(f"请求失败: {r.status_code}")
            
    except Exception as e:
        print(f"请求异常: {e}")

if __name__ == "__main__":
    test_video_feed()