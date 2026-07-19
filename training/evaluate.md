# 模型验收评估 — Stage 1

## 评估命令

```bash
# 在测试集上评估模型
yolo detect val \
    data=dataset.yaml \
    model=runs_study/yolov8n_study_320/weights/best.pt \
    imgsz=320 \
    batch=16 \
    save_json=True \
    save_hybrid=True
```

## 重点查看指标

### 1. 整体指标
- **mAP50** — 目标 >= 0.80
- **mAP50-95** — 目标 >= 0.50（320x320 小尺寸下合理目标）
- **Precision** — 精确率
- **Recall** — 召回率

### 2. 各类别指标
| 类别 | mAP50 目标 | 优先级 |
|------|-----------|--------|
| person | >= 0.85 | 高 — 判断是否在座的基础 |
| phone | >= 0.80 | 最高 — 分心判断关键，召回率优先 |
| book | >= 0.75 | 中 — 判断学习状态 |
| laptop | >= 0.75 | 中 — 判断学习状态 |

### 3. phone 类别专项检查
- phone 召回率应 >= 0.85
- 检查 phone 被误判为 book/其他类别的比例
- 手持手机、桌面手机、亮屏、黑屏四个子场景分别检查

## 混淆矩阵解读

打开 `confusion_matrix.png`：
- 对角线: 正确分类
- 非对角线: 分类错误
- phone 行: phone 被误判为什么
- phone 列: 什么被误判为 phone

## 实际图片推理验证

```bash
# 对单张图片推理
yolo detect predict \
    model=runs_study/yolov8n_study_320/weights/best.pt \
    source=dataset_study_yolo/images/test/ \
    imgsz=320 \
    conf=0.45 \
    save=True
```

在 `runs/detect/predict/` 下查看推理结果图片，目测：
- 框是否贴合目标
- 有没有漏检
- 有没有重复框
- phone 在各种场景下是否能被检测到

## 如果指标不达标

1. **phone 召回率低 (< 0.80)**
   - 这是最严重的问题
   - 增加手持手机样本 50-100 张
   - 增加弱光手机样本 30-50 张
   - 增加手机放在桌面上的样本 30-50 张
   - 降低 conf_thresh 到 0.35

2. **person 召回率低**
   - 增加不同姿势、距离样本
   - 检查标注是否有 person 漏标

3. **book 和 laptop 互相混淆**
   - 两者形态可能相似（都是矩形）
   - 增加更多样化的 book 和 laptop 样本
   - 考虑 context: laptop 通常在更远的位置，book 更近

4. **误检过多**
   - 增加纯背景样本（空桌面、无人物场景）
   - 适当提高 conf_thresh

## 评估通过标准

- [ ] 整体 mAP50 >= 0.80
- [ ] person mAP50 >= 0.85
- [ ] phone mAP50 >= 0.80
- [ ] phone 召回率 >= 0.85
- [ ] 测试集推理结果目测无明显问题
- [ ] 没有某个类别完全失效
