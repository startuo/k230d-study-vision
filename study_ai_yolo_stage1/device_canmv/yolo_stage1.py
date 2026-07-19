"""
Stage 1 YOLOv8 推理封装 — CanMV K230D
封装模型加载、推理、结果绘制，与 main.py 解耦。

严格参考 CanMV K230 官方 YOLOv8 video 示例 API 写法。
如果官方 libs.YOLO 在固件 0.4.0 中的 API 和以下写法有差异，
请查看官方最新的 YOLO 示例代码进行调整。
"""

from libs.YOLO import YOLOv8
from config import (
    MODEL_PATH,
    LABELS,
    MODEL_INPUT_SIZE,
    RGB888P_SIZE,
    DISPLAY_MODE,
    CONF_THRESH,
    NMS_THRESH,
    MAX_BOXES_NUM,
    DEBUG_MODE,
)


class YOLOStage1:
    """
    封装 YOLOv8 检测器，统一管理初始化和推理流程。

    用法:
        detector = YOLOStage1(display_size)
        detector.init()
        while True:
            img = pl.get_frame()
            res = detector.run(img)
            detector.draw_result(res, pl.osd_img)
    """

    def __init__(self, display_size):
        """
        参数:
            display_size: tuple (W, H)，由 PipeLine.get_display_size() 获得，
                          用于将检测坐标映射到显示分辨率。
        """
        self._display_size = display_size
        self._yolo = None

    # ---- 初始化 ----
    def init(self):
        """
        创建 YOLOv8 实例并执行预处理配置。
        必须在 PipeLine.create() 之后调用。
        """
        self._yolo = YOLOv8(
            task_type="detect",
            mode="video",
            kmodel_path=MODEL_PATH,
            labels=LABELS,
            rgb888p_size=RGB888P_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            display_size=self._display_size,
            conf_thresh=CONF_THRESH,
            nms_thresh=NMS_THRESH,
            max_boxes_num=MAX_BOXES_NUM,
            debug_mode=DEBUG_MODE,
        )
        self._yolo.config_preprocess()

    # ---- 推理 ----
    def run(self, img):
        """
        对一帧图像执行目标检测。

        参数:
            img: PipeLine.get_frame() 返回的图像对象

        返回:
            res: YOLOv8 检测结果，具体格式见 vision_state.parse_detections()
        """
        if self._yolo is None:
            raise RuntimeError("YOLOStage1 not initialized. Call init() first.")
        return self._yolo.run(img)

    # ---- 绘制 ----
    def draw_result(self, res, osd_img):
        """
        在 OSD 图层上绘制检测框和标签。

        参数:
            res: YOLOv8.run() 返回值
            osd_img: PipeLine.osd_img 对象

        注意：
            如果固件 0.4.0 中 draw_result 的签名或 osd_img 的操作方式不同，
            请参考官方 YOLO video 示例中的实际用法进行调整。
            yolo.draw_result(res, osd_img) 是官方标准调用方式。
        """
        if self._yolo is None:
            return
        self._yolo.draw_result(res, osd_img)

    # ---- 销毁 ----
    def deinit(self):
        """释放 YOLO 模型资源。"""
        if self._yolo is not None:
            self._yolo.deinit()
            self._yolo = None
