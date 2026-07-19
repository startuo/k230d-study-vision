"""
Stage 1 主入口 — CanMV K230D AI 视觉识别
基于 YOLOv8n 目标检测的桌面学习状态感知。

运行方式:
    1. 将 device_canmv/ 下所有 .py 文件拷贝到 CanMV IDE K230 工程目录
    2. 确保 kmodel 已放在 /sdcard/models/study_yolov8n_320.kmodel
    3. 在 CanMV IDE K230 中打开 main.py，连接 COM4，点击运行

严格参考 CanMV K230 官方 YOLOv8 video 示例 API 写法。
如果你的固件版本和 API 有差异，请查看代码中的 "注意" 注释进行调整。
"""

import os
import sys
import gc
import time

from libs.PipeLine import PipeLine
from libs.Utils import *
import image

from config import (
    RGB888P_SIZE,
    DISPLAY_MODE,
    DEBUG_MODE,
    DEBUG_PRINT_RES,
)
from yolo_stage1 import YOLOStage1
from vision_state import StudyStateMachine, parse_detections
from peripherals import PeripheralsController
from utils import FPSCounter, InferTimer, debug_print, force_gc


# ============================================================
# OSD 文字叠加
# ============================================================
def draw_status_text(img, state, counts, fps, infer_ms):
    """
    在图像上绘制状态信息。

    注意：
        CanMV 的 image 模块文字绘制 API 取决于固件版本。
        常见写法：
            img.draw_string(x, y, text, color=..., scale=...)
        如果当前固件不支持 draw_string，请改用：
            img.draw_text(text, x, y, ...)  # 旧版 API
        或先保留串口 print 输出，注释掉本函数调用。
    """
    # 尝试两种常见 API
    color_white = (255, 255, 255)
    color_green = (0, 255, 0)
    color_red = (255, 0, 0)
    color_yellow = (255, 255, 0)
    color_cyan = (0, 255, 255)

    state_color = {
        "FOCUS": color_green,
        "PHONE": color_red,
        "AWAY": color_yellow,
        "UNKNOWN": color_cyan,
    }
    sc = state_color.get(state, color_white)

    lines = [
        ("STATE: %s" % state, sc),
        ("person:%d phone:%d book:%d laptop:%d" % (
            counts["person"], counts["phone"],
            counts["book"], counts["laptop"]
        ), color_white),
        ("FPS:%.1f  infer:%dms" % (fps, infer_ms), color_white),
    ]

    y = 5
    for text, color in lines:
        try:
            # 官方 CanMV API: img.draw_string(x, y, text, color=color, scale=2)
            img.draw_string(5, y, text, color=color, scale=2)
        except Exception:
            # 旧版 API fallback: img.draw_text(text, x, y, ...)
            try:
                img.draw_text(text, 5, y, color=color, scale=2)
            except Exception:
                pass
        y += 25


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 50)
    print("Stage 1 AI Vision — CanMV K230D Study State Detection")
    print("Board:", "CanMV K230D ATK-DNK230D - 128M")
    print("Firmware: 0.4.0")
    print("=" * 50)

    # ---- 1. 创建 PipeLine ----
    print("[INIT] Creating PipeLine...")
    pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE)
    pl.create()

    display_size = pl.get_display_size()
    print("[INIT] Display size:", display_size)

    # ---- 2. 创建 YOLO 检测器 ----
    print("[INIT] Loading YOLOv8 model...")
    detector = YOLOStage1(display_size)
    detector.init()
    print("[INIT] Model loaded OK.")

    # ---- 3. 创建状态机 ----
    state_machine = StudyStateMachine()

    # ---- 3.5 初始化外设 (Stage 2) ----
    peripherals = PeripheralsController()
    peripherals.init()

    # ---- 4. 工具 ----
    fps_counter = FPSCounter(window=10)
    infer_timer = InferTimer()

    # ---- 5. 主循环 ----
    print("[RUN] Entering main loop...")
    last_gc = time.ticks_ms()

    # 给 sensor / OSD 通道一点时间稳定
    time.sleep_ms(800)

    # 暖通 sensor: 强制激活 chn(2) (OSD 合成通道)
    print("[WARMUP] forcing chn(2) activation...")
    try:
        # 1. 先尝试用 chn(0) 验证 sensor 本身没问题
        sensor0_img = pl.sensor.snapshot(chn=0)
        print("[WARMUP] chn(0) OK, size={}".format(sensor0_img.size() if sensor0_img else None))
    except Exception as e:
        print("[WARMUP] chn(0) failed: {}".format(e))

    # 2. 在 OSD 上画点东西，强制初始化 chn(2)
    try:
        pl.osd_img.clear()
        pl.osd_img.draw_rectangle(0, 0, 100, 50, color=(255, 0, 0), thickness=2)
        pl.show_image()
        print("[WARMUP] OSD drawn, chn(2) should be ready")
        time.sleep_ms(200)
    except Exception as e:
        print("[WARMUP] OSD init failed: {}".format(e))

    # 3. 多次尝试 get_frame
    warmup_ok = False
    for warmup_i in range(5):
        try:
            warmup_img = pl.get_frame()
            if warmup_img is not None:
                pl.show_image()
                print("[RUN] sensor warmup OK (try {})".format(warmup_i + 1))
                warmup_ok = True
                break
        except Exception as e:
            print("[WARMUP] get_frame failed (try {}): {}".format(warmup_i + 1, e))
        time.sleep_ms(200)

    if not warmup_ok:
        print("[WARN] sensor warmup 失败，主循环可能异常")

    try:
        while True:
            # 获取摄像头帧
            try:
                img = pl.get_frame()
            except Exception as e:
                # chn(2) 偶发失败，丢这一帧不退出
                debug_print(1, "[FRAME_ERR] get_frame failed:", e)
                time.sleep_ms(100)
                # 尝试刷新 OSD 看是否能恢复
                try:
                    pl.show_image()
                except Exception:
                    pass
                continue
            if img is None:
                time.sleep_ms(10)
                continue

            # YOLO 推理
            infer_timer.start()
            res = detector.run(img)
            infer_ms = infer_timer.stop()

            # FPS
            fps = fps_counter.tick()

            # 解析检测结果
            counts = parse_detections(res)
            if DEBUG_PRINT_RES and res is not None:
                print("[RAW_RES]", res)

            # 状态机更新
            state = state_machine.update(
                counts["person"], counts["phone"],
                counts["book"], counts["laptop"]
            )

            # 外设响应 (Stage 2)
            peripherals.set_state(state)

            # 绘制检测框 (由 YOLO 库完成)
            detector.draw_result(res, pl.osd_img)

            # 绘制状态文字叠加
            draw_status_text(pl.osd_img, state, counts, fps, infer_ms)

            # 显示
            pl.show_image()

            # 串口日志 (DEBUG_MODE>=1)
            debug_print(1,
                "STATE:", state,
                "| person:", counts["person"],
                "phone:", counts["phone"],
                "book:", counts["book"],
                "laptop:", counts["laptop"],
                "| FPS:", round(fps, 1),
                "infer:", infer_ms, "ms"
            )

            # 定期 GC，防止内存碎片
            if time.ticks_diff(time.ticks_ms(), last_gc) > 5000:
                force_gc()
                last_gc = time.ticks_ms()

    except KeyboardInterrupt:
        print("[EXIT] KeyboardInterrupt")

    except Exception as e:
        print("[ERROR]", e)
        import sys
        sys.print_exception(e)

    finally:
        print("[CLEANUP] Deinitializing...")
        peripherals.deinit()
        detector.deinit()
        pl.destroy()
        force_gc()
        print("[CLEANUP] Done.")


# ============================================================
if __name__ == "__main__":
    main()
