"""
GPIO 引脚扫测脚本 — CanMV K230D
扫描所有可能的 Pin 编号，找到驱动三色灯/蜂鸣器灯的真实引脚。

用法:
    1. 把这个文件上传到 /sdcard/test_gpio.py
    2. 在 CanMV IDE 里运行 (点运行按钮)
    3. 观察哪个灯亮，把亮的 Pin 编号告诉我

注意：会按顺序逐个把每个 Pin 拉高 0.5 秒再拉低。
接线按之前约定：
    - 三色灯 R, G, Y 引脚 (三色灯模块)
    - 蜂鸣器灯 R 引脚
    - 四个引脚分别接 12Pin 排针的 4 个 IO 脚
    - GND 共接 2x4 排针的 GND
"""

import time
from machine import Pin

print("=" * 50)
print("K230D BOX GPIO Pin Scan Test")
print("=" * 50)
print("这个脚本会按顺序测试多个 Pin 编号")
print("找到能让 LED 亮的那个编号")
print("=" * 50)

# 候选 Pin 编号列表 (K230D BOX 的 12Pin 排针是 IO0-IO11,
# 内部映射到 GPIO14-25, 但 CanMV 实际用的编号需要扫)
CANDIDATE_PINS = list(range(0, 12)) + list(range(14, 26))

# 先做一次高电平 LED 扫描 (假设 LED 共阴/active high)
print("\n[1/2] 测试 ACTIVE HIGH (高电平点亮)")
print("如果你看到灯亮 = LED 是共阴，记住 Pin 编号")
print("-" * 50)

for pin_num in CANDIDATE_PINS:
    try:
        p = Pin(pin_num, Pin.OUT)
        p.value(1)  # 高电平
        print("Pin " + str(pin_num) + " = HIGH")
        time.sleep_ms(300)
        p.value(0)  # 拉低
        time.sleep_ms(150)
    except Exception as e:
        # 静默失败
        pass

time.sleep_ms(500)

# 再做一次低电平 LED 扫描 (假设 LED 共阳/active low)
print("\n[2/2] 测试 ACTIVE LOW (低电平点亮)")
print("如果你看到灯亮 = LED 是共阳，记住 Pin 编号")
print("-" * 50)

for pin_num in CANDIDATE_PINS:
    try:
        p = Pin(pin_num, Pin.OUT)
        p.value(0)  # 先拉低
        print("Pin " + str(pin_num) + " = LOW")
        time.sleep_ms(300)
        p.value(0)  # 保持低
        time.sleep_ms(150)
    except Exception as e:
        pass

print()
print("=" * 50)
print("扫描完成!")
print("=" * 50)
print()
print("告诉我:")
print("  1. 在第一阶段 (HIGH) 哪些 Pin 编号让灯亮了?")
print("  2. 在第二阶段 (LOW) 哪些 Pin 编号让灯亮了?")
print("  3. 亮的灯的颜色 (R/G/Y/白)?")
print()
print("我会根据你的反馈更新 config.py")
