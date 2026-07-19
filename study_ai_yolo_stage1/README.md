# Stage 1: AI 视觉识别闭环 — 桌面学习状态感知

## 阶段定位

**本阶段只做 AI 视觉识别**，不包含以下功能（后续阶段再做）：
- 蜂鸣器 / 光照传感器 / 温湿度传感器
- Web 页面 / 云端推理
- OCR / TTS / 姿态大模型
- 复杂 UI / 复杂日志系统

## 技术栈

| 组件 | 选型 |
|------|------|
| 开发板 | CanMV K230D ATK-DNK230D 128M |
| 固件版本 | 0.4.0 |
| IDE | CanMV IDE K230 |
| 串口 | COM4 |
| 模型 | YOLOv8n 目标检测 (detect) |
| 输入尺寸 | 320×320 |
| 检测类别 | person, phone, book, laptop (4类) |
| 学习状态 | FOCUS / PHONE / AWAY / UNKNOWN |

## 目录结构

```
study_ai_yolo_stage1/
├── README.md                          # 本文件
├── device_canmv/                      # K230D 设备端代码 (MicroPython)
│   ├── main.py                        # 主入口
│   ├── config.py                      # 所有配置
│   ├── yolo_stage1.py                 # YOLO 推理封装
│   ├── vision_state.py                # 状态机
│   └── utils.py                       # 工具函数
├── training/                          # PC 端训练文档
│   ├── dataset_guide.md               # 数据集采集指南
│   ├── train_yolov8.md                # 训练方案
│   ├── export_onnx.md                 # ONNX 导出
│   ├── convert_to_kmodel.md           # KModel 转换
│   └── evaluate.md                    # 模型评估
├── dataset_tools/                     # 数据集工具 (PC端)
│   ├── dataset.yaml                   # 训练配置
│   ├── split_dataset.py               # 数据集划分
│   └── check_yolo_labels.py           # 标注检查
└── models/                            # 模型文件
    ├── labels.txt                     # 类别标签
    └── put_kmodel_here.txt            # kmodel 放置说明
```

## 文件放置路径 (K230D)

```
/sdcard/models/
├── study_yolov8n_320.kmodel    # 转换好的 YOLOv8n 模型
└── study_labels.txt            # 类别标签文件
```

## CanMV IDE K230 运行步骤

1. 将 `device_canmv/` 下的 5 个 `.py` 文件拷贝到 CanMV IDE K230 工程中
2. 确保 K230D 的 SD 卡 `models/` 目录下有 `study_yolov8n_320.kmodel` 和 `study_labels.txt`
3. 打开 CanMV IDE K230，连接 COM4
4. 打开 `main.py`，点击运行（或按 F5）

## 串口输出

在 CanMV IDE K230 的串口终端中可以看到实时日志：
```
[STAGE1] STATE: FOCUS | person: 1 phone: 0 book: 1 laptop: 0 | FPS: 15.2 infer: 45 ms
```

## 阶段一成功标准

- [ ] CanMV IDE K230 能运行 main.py，不报错
- [ ] LCD 显示实时摄像头画面
- [ ] 成功加载 study_yolov8n_320.kmodel
- [ ] 画面上绘制出 YOLO 检测框
- [ ] 人在画面中 → 检测到 person
- [ ] 手机在画面中 → 检测到 phone
- [ ] 书或电脑在画面中 → 检测到 book/laptop
- [ ] 串口或画面显示 STATE (FOCUS/PHONE/AWAY/UNKNOWN)
- [ ] 手机持续出现 3 秒 → 状态切到 PHONE
- [ ] 人离开 5 秒 → 状态切到 AWAY
- [ ] 状态不会频繁跳变（去抖有效）
- [ ] 不接任何外设也能跑完整 AI 流程

## 验收测试流程

| 测试 | 操作 | 期望结果 |
|------|------|----------|
| 1. 模型加载 | 运行 main.py | 不报错，画面显示 |
| 2. person 检测 | 人坐在桌前 | 检测到 person，状态→FOCUS/UNKNOWN |
| 3. phone 检测 | 拿起手机保持 3s+ | 检测到 phone，状态→PHONE |
| 4. away 检测 | 人离开画面 5s+ | 状态→AWAY |
| 5. 学习物品 | 放入书本/电脑 | 检测到 book/laptop |
| 6. 抗抖动 | 手机快速划过 | 状态不切换 |

## 开发流程总览

```
数据集采集 → 标注 → 划分
    ↓
PC 端训练 (YOLOv8n, 320×320)
    ↓
ONNX 导出 (opset=12, simplify)
    ↓
nncase 转换 → study_yolov8n_320.kmodel
    ↓
拷贝到 K230D /sdcard/models/
    ↓
CanMV IDE K230 运行 device_canmv/main.py
    ↓
验收测试 → 阶段一完成
```

## 常见问题

**Q: main.py 运行报 ImportError**
A: 确认 `libs/` 目录在 CanMV IDE 工程路径下，这是 K230D 固件自带的库。

**Q: 模型加载失败**
A: 检查 kmodel 路径是否正确，SD 卡是否插入，文件名是否一致。

**Q: 检测框显示但状态不切换**
A: 检查 vision_state.py 中的去抖时间设置，确认 DEBUG_MODE>=1 查看串口日志。

**Q: res 格式解析错误**
A: 在 config.py 中设 `DEBUG_PRINT_RES = 1`，运行后查看串口打印的实际格式，修改 vision_state.py 的 parse_detections()。

**Q: 画面卡顿**
A: 降低 RGB888P_SIZE 或 MODEL_INPUT_SIZE，关闭 DEBUG_MODE。
