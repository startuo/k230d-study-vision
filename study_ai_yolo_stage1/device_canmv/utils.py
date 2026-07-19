"""
Stage 1 工具函数 — CanMV K230D
FPS 计算、调试输出、计时辅助。
"""

import time
import gc

# ============================================================
# FPS 计数器
# ============================================================
class FPSCounter:
    """简单的帧率计算器。"""
    def __init__(self, window=10):
        self._window = window
        self._ticks = []
        self._last_tick = 0
        self._fps = 0.0

    def tick(self):
        """记录一帧，返回当前 FPS (float)。"""
        now = time.ticks_ms()
        self._ticks.append(now)
        if len(self._ticks) > self._window:
            self._ticks.pop(0)
        if len(self._ticks) >= 2:
            elapsed_ms = time.ticks_diff(self._ticks[-1], self._ticks[0])
            if elapsed_ms > 0:
                self._fps = (len(self._ticks) - 1) * 1000.0 / elapsed_ms
        self._last_tick = now
        return self._fps

    @property
    def fps(self):
        return self._fps

    def reset(self):
        self._ticks = []
        self._fps = 0.0


# ============================================================
# 推断耗时估算
# ============================================================
class InferTimer:
    """记录单次推断耗时 (ms)。需要手动在 yolo.run 前后调用 start/stop。"""
    def __init__(self):
        self._start = 0
        self._elapsed_ms = 0

    def start(self):
        self._start = time.ticks_ms()

    def stop(self):
        self._elapsed_ms = time.ticks_diff(time.ticks_ms(), self._start)
        return self._elapsed_ms

    @property
    def ms(self):
        return self._elapsed_ms


# ============================================================
# 调试打印
# ============================================================
def debug_print(level, *args, **kwargs):
    """
    按 DEBUG_MODE 等级打印。
    level=1: 基本状态信息
    level=2: 详细检测信息
    """
    from config import DEBUG_MODE
    if DEBUG_MODE >= level:
        print("[STAGE1]", *args, **kwargs)


def force_gc():
    """手动触发垃圾回收，返回释放的内存字节数（如果 API 支持）。"""
    return gc.collect()
