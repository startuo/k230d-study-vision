# YOLOv8n 训练方案 — Stage 1

## 环境要求

- Python >= 3.9
- PyTorch >= 2.0
- Ultralytics >= 8.1.0
- GPU (推荐 NVIDIA GTX 1060 6G 以上)

### 安装
```bash
pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 数据集准备

1. 按 `dataset_guide.md` 完成数据采集和标注
2. 使用 `dataset_tools/split_dataset.py` 划分数据集
3. 将 `dataset_tools/dataset.yaml` 放到数据集根目录
4. 确认目录结构：
```
dataset_study_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

## 训练命令

```bash
yolo detect train \
    data=dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=320 \
    batch=16 \
    workers=4 \
    project=runs_study \
    name=yolov8n_study_320 \
    patience=20 \
    cos_lr=True \
    close_mosaic=10
```

### 参数说明
| 参数 | 值 | 说明 |
|------|-----|------|
| data | dataset.yaml | 数据集配置文件路径 |
| model | yolov8n.pt | 预训练权重（自动下载） |
| epochs | 100 | 训练轮数 |
| imgsz | 320 | 输入尺寸，与 K230D 部署一致 |
| batch | 16 | 批次大小，显存不足降到 8 或 4 |
| workers | 4 | 数据加载线程数 |
| project | runs_study | 输出目录 |
| name | yolov8n_study_320 | 实验名称 |
| patience | 20 | 早停轮数，20 轮无提升则停止 |
| cos_lr | True | 余弦学习率衰减 |
| close_mosaic | 10 | 最后 10 轮关闭 mosaic 增强 |

### 显存不足时
```bash
# 8G 显存
batch=8

# 4-6G 显存
batch=4

# CPU 训练 (慢，仅应急)
batch=1 workers=0 device=cpu
```

## 训练输出

训练完成后在 `runs_study/yolov8n_study_320/` 下查看：
- `weights/best.pt` — 最佳模型权重
- `weights/last.pt` — 最后一轮权重
- `results.csv` — 每轮指标记录
- `confusion_matrix.png` — 混淆矩阵
- `results.png` — 训练曲线图
- `val_batch*.jpg` — 验证集推理结果样例

## 验收指标

| 指标 | 目标 | 说明 |
|------|------|------|
| 整体 mAP50 | >= 0.80 | 所有类别均值 |
| person mAP50 | >= 0.85 | 人检测必须可靠 |
| phone mAP50 | >= 0.80 | 手机检测关键指标 |
| book mAP50 | >= 0.75 | 书本检测 |
| laptop mAP50 | >= 0.75 | 电脑检测 |
| phone recall | 优先于 precision | 漏检手机的影响大于误检 |

## 如何判断训练好坏

1. 打开 `results.png`，观察 loss 曲线：
   - box_loss 和 cls_loss 持续下降并趋于平稳 → 正常
   - loss 震荡剧烈 → 数据量不足或标注质量差
2. 打开 `confusion_matrix.png`：
   - 对角线颜色深 → 分类准确
   - phone 被误判为其他类别 → 需要增加 phone 样本
3. 查看 `val_batch*.jpg`：
   - 目测检测框是否贴合
   - 是否有漏检（目标存在但未画框）

## 训练结果不达标时的优化建议

1. **phone 召回率低**：
   - 增加手持手机场景样本
   - 增加弱光下手机样本
   - 增加不同角度手机样本
   - 降低 conf_thresh 到 0.35 或 0.30

2. **person 召回率低**：
   - 增加不同姿势样本
   - 增加不同距离样本
   - 检查标注是否有漏标

3. **book/laptop 召回率低**：
   - 增加不同种类书本/电脑样本
   - 增加不同角度和遮挡情况

4. **误检多**：
   - 增加背景多样性（不同桌面、不同杂物）
   - 适度提高 conf_thresh

5. **通用**：
   - 增加总数据量到 2000+
   - 检查标注质量（框是否贴合、是否有漏标）
   - 尝试 yolov8s（更大的模型，但可能超出 K230D 128M 的内存限制）
