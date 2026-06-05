# -*- coding: utf-8 -*-
import urllib.request

try:
    url = 'http://localhost:5000/video_feed?file=traffic_rideo.mp4&mode=normal'
    print(f"请求: {url}")
    
    response = urllib.request.urlopen(url, timeout=10)
    print(f"状态码: {response.status}")
    print(f"Content-Type: {response.headers['Content-Type']}")
    
    data = response.read(1024)
    print(f"收到数据: {len(data)} bytes")
    print(f"开头: {data[:50]}")
    
except Exception as e:
    print(f"错误: {e}")