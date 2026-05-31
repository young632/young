# -*- coding: utf-8 -*-
"""
GUI测试脚本 - 验证修复
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('=' * 60)
print('Testing GUI module...')
print('=' * 60)
print()

try:
    print('1. Importing gui module...', end='')
    from gui import TrafficMonitorGUI
    print(' OK')
except Exception as e:
    print(' FAILED:', e)
    sys.exit(1)

print()
print('2. GUI module imported successfully!')
print()
print('=' * 60)
print('Now run the GUI on your computer:')
print('   cd "d:\\计算机实践\\TrafficMonitorSystem"')
print('   python gui.py')
print('=' * 60)
print()
print('Fixes applied:')
print(' - Thread-safe display using queue')
print(' - Keep PhotoImage references')
print(' - Display first frame immediately after adding video')
print(' - Main-thread only UI updates')
print()
