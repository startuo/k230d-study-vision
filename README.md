# 基于 K230D 的桌面学习状态 AI 视觉识别

> **K230D BOX + YOLOv8n + CanMV MicroPython** — 端侧桌面学习状态感知与 LED 视觉反馈
>
> 嵌入式人工智能基础与应用 课程项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![K230D](https://img.shields.io/badge/Hardware-CanMV%20K230D%20BOX-blue)](https://wiki.alientek.com/docs/Boards/Kendryte/DNK230D/start-guide/k230d-box-introduction)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-brightgreen)](https://github.com/ultralytics/ultralytics)

## 📌 项目简介

本项目以正点原子 **K230D BOX** 开发板为核心，使用板载 **GC2093 摄像头** 实时采集桌面学习场景画面。通过 **YOLOv8n** 目标检测模型识别画面中的人、手机和书本，结合带时间窗口去抖的有限状态机，自动判定学习者的学习状态，并通过板载 **12Pin GPIO 排针** 驱动的三色 LED 模块给出实时视觉反馈。

**检测类别：** `person` / `phone` / `book`

**输出状态：** `FOCUS` (专注) / `PHONE` (分心) / `PERSON` (人在) / `AWAY` (离开)

系统以 **25 FPS** 的速度在端侧独立运行，KPU 单帧推理约 **30 ms**，模型经 INT8 量化后仅 **3.2 MB**。

---

## 🖥️ 硬件平台

| 组件 | 型号 | 说明 |
|------|------|------|
| 核心板 | CNK230DF (K230D 128MB) | RISC-V 双核 + KPU 6TOPS |
| 底板 | DNK230D | LCD + 摄像头 + 排针扩展 |
| 摄像头 | GC2093 | MIPI CSI1，1920x1080 |
| 屏幕 | 2.4" LCD 触摸屏 | 640×480 |
| 三色 LED 模块 | R/G/B | 12Pin 排针驱动 (蓝/绿/红) |

### 引脚映射

| 物理引脚 | Pin 编号 | 功能 |
|---------|---------|------|
| IO0 | Pin(24) | 绿灯 — 无人 / 程序运行 |
| IO1 | Pin(25) | 红灯 — 检测到手机 |
| IO2 | Pin(22) | 蓝灯 — 检测到人 |

---

## 🧠 系统架构

```
┌──────────────────────────────────────────────────────┐
│                    K230D BOX                           │
│                                                        │
│  Camera → PipeLine → YOLOv8n(KPU) → parse_detections  │
│                        │                              │
│                  ┌─────┴─────┐                         │
│                  │ 状态机     │  FOCUS/PHONE/PERSON/AWAY│
│                  └─────┬─────┘                         │
│                        ▼                               │
│                  peripherals.set_state(state)          │
│                        │                               │
│                  ┌─────┴─────┐                         │
│                  │ Pin(24)=绿 │  ← 无人                 │
│                  │ Pin(22)=蓝 │  ← 人在                 │
│                  │ Pin(25)=红 │  ← 手机                 │
│                  └───────────┘                         │
└──────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
k230d-study-vision/
├── README.md               # 项目介绍
├── .gitignore
├── prepare_dataset.py      # 数据集预处理 (RGBA→RGB, 修复错误)
├── convert_kmodel.py       # ONNX → KModel 转换脚本
├── device_canmv/           # K230D 端 7 个 .py 文件
│   ├── main.py             # 主入口
│   ├── config.py           # 集中配置
│   ├── yolo_stage1.py      # YOLOv8 推理封装
│   ├── vision_state.py     # 检测结果解析 + 状态机
│   ├── peripherals.py      # 三色 LED 控制
│   ├── utils.py            # FPS 计时器、调试工具
│   └── test_gpio.py        # GPIO 引脚扫测
├── models/                 # 模型与标签 (大模型文件通过 Releases 下载)
│   ├── labels.txt          # 类别标签
│   └── put_kmodel_here.txt # 模型放置说明
├── training/               # 训练/转换/评估文档
│   ├── train_yolov8.md     # YOLO训练方案
│   ├── export_onnx.md      # ONNX导出方案
│   ├── convert_to_kmodel.md# nncase转换方案
│   ├── evaluate.md         # 评估标准
│   └── dataset_guide.md    # 数据集采集指南
└── dataset_tools/          # 数据集处理工具
    ├── dataset.yaml        # 训练配置文件
    ├── split_dataset.py    # 70/20/10 划分
    └── check_yolo_labels.py# 标注格式检查
```

---

## 🚀 快速开始

### 1. 数据集准备

```bash
# 原始数据目录结构:
#   raw_dataset/images/ ← 图片
#   raw_dataset/labels/ ← YOLO 格式标注

python prepare_dataset.py --input_dir ./raw_dataset/ --output_dir ./dataset_study_yolo/
python study_ai_yolo_stage1/dataset_tools/check_yolo_labels.py --labels_dir dataset_study_yolo/labels/train/
```

### 2. 模型训练 (PC)

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

### 3. ONNX 导出 + KModel 转换

```bash
# ONNX
yolo export model=runs_study/yolov8n_study_320/weights/best.pt format=onnx imgsz=320 opset=12 simplify=True

# KModel (需 nncase 2.10.0)
python study_ai_yolo_stage1/convert_kmodel.py
```

### 4. 部署到 K230D

将 `device_canmv/` 下 7 个 `.py` 文件 + `models/` 下的 kmodel 和 labels.txt 拷贝到 SD 卡 `/sdcard/`。在 CanMV IDE K230 中打开 `main.py`，连接 COM4，点击运行。

---

## 📊 训练结果

| Epoch | mAP50 | 备注 |
|-------|-------|------|
| 1 | 0.265 | 初始 |
| 40 | 0.542 | 中期 |
| **90 (最佳)** | **0.610** | — |
| 98 | 0.608 | 最终 |

| 类别 | 标注框数 | 占比 |
|------|---------|------|
| person | 11,207 | 43.4% |
| phone | 10,841 | 42.0% |
| book | 3,762 | 14.6% |

---

## 💡 状态映射

| AI 状态 | 触发条件 | 蓝灯 | 绿灯 | 红灯 |
|---------|---------|------|------|------|
| AWAY / UNKNOWN | 无人或过渡 | 灭 | 亮 | 灭 |
| FOCUS / PERSON | 人在 (有书/无书) | 亮 | 灭 | 灭 |
| PHONE | 检测到手机 | 灭 | 灭 | 亮 |

---

## 🐛 常见问题

**Q: 摄像头打不开 / sensor snapshot 失败**
A: 拔插电源硬复位开发板，清理 SD 卡旧文件后重新上传。

**Q: GPIO 报错 `can't import name GPIO`**
A: K230D CanMV 使用 `machine.Pin` 而非 `machine.GPIO`。确认 `peripherals.py` 中使用 `from machine import Pin`。

**Q: 灯不亮**
A: 先用 `test_gpio.py` 扫描 Pin 编号，确认接线对应的实际 Pin 号，再更新 `config.py`。

**Q: 状态一直 AWAY 不切换**
A: CONF_THRESH 过高或检测置信度太低。降低至 0.15 或 0.10 再试。

---

## 📄 License

MIT License

## 🙏 致谢

- [嘉楠科技 - K230](https://www.kendryte.com/k230/) — 芯片 & SDK
- [正点原子 - K230D BOX](https://wiki.alientek.com/docs/Boards/Kendryte/DNK230D/start-guide/k230d-box-introduction) — 开发板
- [Ultralytics - YOLOv8](https://github.com/ultralytics/ultralytics) — 目标检测框架
- [nncase](https://github.com/kendryte/nncase) — 模型编译工具链
