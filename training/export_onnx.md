# ONNX 导出方案 — Stage 1

## 前提

已完成 YOLOv8n 训练，获得 `best.pt`。

## 导出命令

```bash
yolo export \
    model=runs_study/yolov8n_study_320/weights/best.pt \
    format=onnx \
    imgsz=320 \
    opset=12 \
    simplify=True
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| model | best.pt 路径 | 训练得到的最佳权重 |
| format | onnx | 导出格式 |
| imgsz | 320 | 输入尺寸，与训练一致 |
| opset | 12 | ONNX opset 版本（K230 nncase 一般支持 11-12） |
| simplify | True | 使用 onnxsim 简化模型图 |

### 注意

- 如果 nncase 工具链文档明确要求 opset=11，请改为 `opset=11`
- 如果 nncase 要求 opset=13+，请相应调整
- **opset 版本以 nncase 工具链文档为准**

## 导出后检查

```bash
# 确认文件存在
ls -la runs_study/yolov8n_study_320/weights/best.onnx

# 安装 netron 查看模型结构
pip install netron
netron runs_study/yolov8n_study_320/weights/best.onnx
```

### 检查清单

1. `best.onnx` 文件是否生成
2. 输入节点：
   - 名称通常是 `images`
   - shape 应为 `[1, 3, 320, 320]` (NCHW)
   - 数据类型 float32
3. 输出节点：
   - 通常有 1 个或多个输出
   - 类别数应与训练一致 (4 类, 但 YOLOv8 输出通道包含 box 坐标)
   - 具体输出格式取决于 YOLOv8 版本和导出参数
4. 使用 `onnx.checker.check_model(best.onnx)` 验证 ONNX 文件完整性

### 验证脚本 (PC 端)

```python
import onnx

# 检查 ONNX 文件
model = onnx.load("runs_study/yolov8n_study_320/weights/best.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid!")

# 打印输入输出信息
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")

print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}")
```

## 常见问题

### 问题 1: onnxsim 失败
- 确保安装了 `onnxsim`: `pip install onnxsim`
- 如果不支持 simplify，去掉 `simplify=True` 参数

### 问题 2: opset 版本报错
- nncase 转换时如果报 "unsupported opset version"，调整 opset 值
- 尝试 `opset=11` 或 `opset=13`

### 问题 3: 输入尺寸不匹配
- 确认训练时 imgsz=320，导出时也用 320
- 不要用 640 导出再转 320，那样可能丢失信息
