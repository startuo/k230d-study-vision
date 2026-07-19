# 源码说明

> 《基于 K230D 的桌面学习状态 AI 视觉识别》配套源码

## 目录结构

```
源码/
├── README.md                   # 本文件
├── prepare_dataset.py          # 数据集准备脚本 (RGBA→RGB, 修复错误, 划分)
├── convert_kmodel.py           # ONNX → KModel 转换脚本
├── device_canmv/               # K230D 端 MicroPython 代码 (部署到 /sdcard/)
│   ├── main.py                 # 主入口
│   ├── config.py               # 所有配置 (模型路径、GPIO、阈值)
│   ├── yolo_stage1.py          # YOLOv8 推理封装
│   ├── vision_state.py         # 检测结果解析 + 状态机
│   ├── peripherals.py          # 3 色 LED 控制
│   ├── utils.py                # FPS 计时器、调试工具
│   └── test_gpio.py            # GPIO 引脚扫描测试脚本
├── training_docs/              # 训练与转换文档
│   ├── dataset_guide.md        # 数据集采集与标注指南
│   ├── train_yolov8.md         # YOLOv8 训练命令
│   ├── export_onnx.md          # ONNX 导出
│   ├── convert_to_kmodel.md    # KModel 转换详细说明
│   └── evaluate.md             # 模型评估指标
├── dataset_tools/              # 数据集处理工具
│   ├── dataset.yaml            # 训练用数据集配置
│   ├── split_dataset.py        # 按 70/20/10 划分 train/val/test
│   └── check_yolo_labels.py    # 标注文件格式检查
└── models/                     # 模型与标签
    ├── best.onnx               # 导出的 ONNX (11.6 MB)
    └── labels.txt              # 类别标签 (person / phone / book)
```

## 完整使用流程

### 1. 准备数据集 (PC 端)

```bash
# 收集数据，按以下结构放置:
#   raw_dataset/images/  ← 原始图片
#   raw_dataset/labels/  ← YOLO 格式标注

# 转换 + 划分 + 检查
python prepare_dataset.py --input_dir ./raw_dataset/ --output_dir ./dataset_study_yolo/
python dataset_tools/check_yolo_labels.py --labels_dir dataset_study_yolo/labels/train/
```

### 2. 训练模型 (PC 端)

```bash
yolo detect train \
    data=dataset_study_yolo/dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=320 \
    batch=4 \
    workers=0 \
    project=runs_study \
    name=yolov8n_study_320
```

训练完成后 `runs_study/yolov8n_study_320/weights/best.pt` 是最佳权重。

### 3. 导出 ONNX

```bash
yolo export \
    model=runs_study/yolov8n_study_320/weights/best.pt \
    format=onnx \
    imgsz=320 \
    opset=12 \
    simplify=True
```

输出 `best.onnx`（约 11.6 MB）。

### 4. 转 KModel (PC 端, 需安装 nncase 2.10)

```bash
# 准备校准集 (从训练集抽 100~150 张)
mkdir calib_set
cp dataset_study_yolo/images/train/*.jpg calib_set/ | head -150

# 安装
pip install nncase==2.10.0
# 还需要从 https://gitee.com/kendryte/nncase/releases 下载
#   nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
#   pip install nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl

# 转换
python convert_kmodel.py
```

输出 `models/study_yolov8n_320.kmodel`（约 3.2 MB）。

### 5. 部署到 K230D

把以下文件复制到 K230D 的 SD 卡 `/sdcard/` 目录：

```
/sdcard/
├── main.py
├── config.py
├── yolo_stage1.py
├── vision_state.py
├── peripherals.py
├── utils.py
├── models/
│   ├── study_yolov8n_320.kmodel
│   └── study_labels.txt
└── libs/  (CanMV IDE 自带)
```

### 6. 运行

在 CanMV IDE K230 中打开 `main.py`，连接 COM4，点击运行。

## 状态映射

| AI 状态 | 触发条件 | 蓝灯 (Pin 22) | 绿灯 (Pin 24) | 红灯 (Pin 25) |
|---------|---------|--------------|--------------|--------------|
| AWAY | 无人 | 灭 | 亮 | 灭 |
| PERSON | 人在但无书无手机 | 亮 | 灭 | 灭 |
| FOCUS | 人在 + 有书 | 亮 | 灭 | 灭 |
| PHONE | 检测到手机 | 灭 | 灭 | 亮 |
| UNKNOWN | 默认/过渡 | 灭 | 亮 | 灭 |

## 性能指标

- 模型：YOLOv8n 3 类，mAP50 = 0.610
- KModel 大小：3.2 MB (int8 量化)
- 推理速度：~30 ms / 帧 (KPU 加速)
- 帧率：~25 FPS

## 硬件接线

| K230D BOX 引脚 | 外设 | 说明 |
|----------------|------|------|
| MIPI CSI1 | GC2093 摄像头 | 板载直插 |
| MIPI DSI | 2.4" LCD 触摸屏 | 板载直插 |
| 12Pin 排针 IO0 (Pin 24) | 三色 LED 绿 | 无人 / 程序运行 |
| 12Pin 排针 IO1 (Pin 25) | 三色 LED 红 | 检测到手机 |
| 12Pin 排针 IO2 (Pin 22) | 三色 LED 蓝 | 检测到人 |
| 2x4 上方排针 GND | 三色 LED GND | 公共地 |

## 调试参考

如果 K230D 端灯不亮或状态不对，按以下顺序排查：

1. 检查 `config.py` 中 `LED_BLUE_GPIO/LED_GREEN_GPIO/LED_RED_GPIO` 与实际接线是否一致
2. 在 REPL 跑 `from machine import Pin; Pin(22, Pin.OUT).value(1)` 验证蓝灯控制
3. 检查 `DEBUG_PRINT_RES = 1` 时 `res` 是否包含检测结果
4. 检查 `CONFIRM_SECONDS` 是否设得过短
