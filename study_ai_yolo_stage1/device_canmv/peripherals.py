"""
Stage 2 外设控制 — CanMV K230D
控制三色 LED (蓝/绿/红)。

接线 (实测对应):
    LED_BLUE_GPIO  = 22  → 蓝灯 - 检测到人时亮
    LED_GREEN_GPIO = 24  → 绿灯 - 没人时亮 (程序正常运行)
    LED_RED_GPIO   = 25  → 红灯 - 检测到看手机时亮

蜂鸣器灯暂不接。

API 适配:
    K230D CanMV 固件使用 machine.Pin 类 (类似 ESP32 API)
    from machine import Pin
    p = Pin(pin_num, Pin.OUT)
    p.value(1)  # 高电平
"""

from config import (
    LED_BLUE_GPIO,
    LED_GREEN_GPIO,
    LED_RED_GPIO,
    LED_ACTIVE_HIGH,
    LED_PATTERNS,
)
from utils import debug_print


class PeripheralsController:
    """
    控制 3 色 LED (蓝/绿/红)。
    使用查表模式，根据状态名直接查 LED_PATTERNS 设置灯的开关。
    """

    def __init__(self):
        self._pins = {}             # name -> Pin object
        self._current_pattern = (0, 0, 0)
        self._initialized = False

    def init(self):
        """
        初始化 3 个 Pin 为 OUT，初始全部熄灭。
        必须在 main loop 开始之前调用一次。
        """
        if self._initialized:
            return

        try:
            from machine import Pin
        except ImportError as e:
            print("[PERIPHERALS] machine.Pin not found:", e)
            return

        pin_map = {
            'B': LED_BLUE_GPIO,
            'G': LED_GREEN_GPIO,
            'R': LED_RED_GPIO,
        }

        for name, pin_num in pin_map.items():
            try:
                p = Pin(pin_num, Pin.OUT)
                p.value(1 if not LED_ACTIVE_HIGH else 0)
                self._pins[name] = p
            except Exception as e:
                msg = "[PERIPHERALS] Failed to init Pin" + str(pin_num) + " (" + name + "): " + str(e)
                print(msg)

        self._initialized = True
        msg = "[PERIPHERALS] Initialized. Pins: B=" + str(LED_BLUE_GPIO)
        msg += ", G=" + str(LED_GREEN_GPIO)
        msg += ", R=" + str(LED_RED_GPIO)
        msg += ", active_high=" + str(LED_ACTIVE_HIGH)
        print(msg)

    def set_state(self, state):
        """
        根据状态名设置 LED 开关。从 config.LED_PATTERNS 查表。
        不识别的状态名会全关。
        """
        if not self._initialized:
            return

        pattern = LED_PATTERNS.get(state, (0, 0, 0))
        if pattern == self._current_pattern:
            return  # 没变，避免无谓写入

        b, g, r = pattern
        self._apply('B', b)
        self._apply('G', g)
        self._apply('R', r)
        self._current_pattern = pattern
        # 状态变化时打印 (简洁)
        print("[LED]", state, "-> B=", b, " G=", g, " R=", r)

    def _apply(self, name, on):
        """把 0/1 映射到实际电平并写入 Pin。"""
        pin = self._pins.get(name)
        if pin is None:
            return
        # LED_ACTIVE_HIGH=True: on=1→输出高; on=0→输出低
        # LED_ACTIVE_HIGH=False: on=1→输出低; on=0→输出高
        level = 1 if (on if LED_ACTIVE_HIGH else not on) else 0
        try:
            pin.value(level)
        except Exception as e:
            debug_print(2, "[PERIPHERALS] Pin " + name + " write failed:", e)

    def all_off(self):
        """全部熄灭 (强制清零)。"""
        if not self._initialized:
            return
        for name in self._pins:
            self._apply(name, 0)
        self._current_pattern = (0, 0, 0)

    def deinit(self):
        """
        清理资源：先关所有灯，再释放 Pin。
        退出 main loop 时调用。
        """
        self.all_off()
        self._pins.clear()
        self._initialized = False
        print("[PERIPHERALS] Deinitialized.")
