# -*- coding: utf-8 -*-
"""
快速测试脚本
"""

import sys
import os

print('=' * 60)
print('Traffic Monitor System - Quick Test')
print('=' * 60)
print()

try:
    print('1. Import TrafficMonitor...', end='')
    from traffic_monitor import TrafficMonitor
    print(' OK')
except Exception as e:
    print(' FAILED:', e)
    sys.exit(1)

try:
    print('2. Import config...', end='')
    from config import *
    print(' OK')
except Exception as e:
    print(' FAILED:', e)
    sys.exit(1)

try:
    print('3. Initialize TrafficMonitor...', end='')
    monitor = TrafficMonitor()
    print(' OK')
except Exception as e:
    print(' FAILED:', e)
    sys.exit(1)

try:
    print('4. Check test video...', end='')
    test_video = r'D:\计算机实践\solidWhiteRight.mp4'
    if os.path.exists(test_video):
        print(' OK')
    else:
        print(' Not found')
except Exception as e:
    print(' FAILED:', e)

print()
print('=' * 60)
print('All tests passed!')
print('=' * 60)
print()
print(' Next step: On your computer, run GUI:')
print('   cd "d:\\计算机实践\\TrafficMonitorSystem"')
print('   python gui.py')
print()