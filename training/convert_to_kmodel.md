# nncase 转 KModel 方案 — Stage 1

## 重要说明

**nncase 工具链的命令和参数可能随版本变化。**
以下命令为模板，请以 **Canaan 官方 nncase 文档** 为准进行调整。

官方参考文档：
- https://www.kendryte.com/k230_canmv/zh/main/ai_dev_doc.html
- nncase GitHub: https://github.com/kendryte/nncase

## 环境要求

- Linux (Ubuntu 20.04/22.04) 或 WSL2
- Python 3.9-3.11
- nncase >= 2.x（具体版本以官方 K230 SDK 配套版本为准）
- dotnet runtime (nncase 依赖)

### 安装 nncase

```bash
# 方式 1: pip 安装（如果官方提供了 pip 包）
pip install nncase
pip install nncase-k230

# 方式 2: 从 Canaan 官方下载预编译包
# 参考 https://github.com/kendryte/nncase/releases
# 解压后设置环境变量
```

**请以官方文档为准。** 不同版本安装方式可能不同。

## 校准集准备

从训练集的 train 中抽取 100-300 张代表性图片：

```bash
# 创建校准集目录
mkdir -p calib_set

# 从 train 目录随机复制 150 张图片
# Linux/macOS:
find dataset_study_yolo/images/train -name "*.jpg" | shuf -n 150 | xargs -I {} cp {} calib_set/

# Windows PowerShell:
# Get-ChildItem dataset_study_yolo/images/train/*.jpg | Get-Random -Count 150 | Copy-Item -Destination calib_set/
```

校准集要求：
- 覆盖所有 4 个类别
- 覆盖不同光照和角度
- 图片需要预处理为模型输入尺寸 320x320

## 转换命令 (模板)

```python
# convert_kmodel.py — 转换脚本模板
# 注意：以下 API 仅供参考，请以 nncase 实际版本的 API 为准

import nncase
import numpy as np
from PIL import Image
import os

# ---- 配置 ----
ONNX_PATH = "runs_study/yolov8n_study_320/weights/best.onnx"
KMODEL_PATH = "study_yolov8n_320.kmodel"
CALIB_DIR = "calib_set/"
INPUT_SIZE = (320, 320)
TARGET = "k230"

# ---- 1. 导入 ONNX ----
print("[1/5] Importing ONNX...")
model = nncase.Importer().import_onnx(ONNX_PATH)
# 或: compile_options = nncase.CompileOptions()
#     compile_options.target = TARGET
#     compiler = nncase.Compiler(compile_options)
#     compiler.import_onnx(ONNX_PATH)

# ---- 2. 设置输入形状 ----
print("[2/5] Setting input shape...")
# 设置输入 shape 为 (1, 3, 320, 320)
# shape_list = [nncase.Shape(1, 3, 320, 320)]
# compiler.set_input_shapes(shape_list)

# ---- 3. 准备校准集数据 ----
print("[3/5] Preparing calibration dataset...")
def calib_data_generator():
    for fname in os.listdir(CALIB_DIR):
        if fname.endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(CALIB_DIR, fname)).convert('RGB')
            img = img.resize(INPUT_SIZE)
            arr = np.array(img, dtype=np.float32)
            # 归一化 (根据模型预处理方式调整)
            arr = arr / 255.0
            # 转换为 NCHW
            arr = arr.transpose(2, 0, 1)
            arr = np.expand_dims(arr, axis=0)
            yield [arr]

# ---- 4. 编译 ----
print("[4/5] Compiling...")
# compiler.use_ptq()  # 使用后训练量化
# calib_dataset = nncase.CalibrationDataset(calib_data_generator)
# compiler.set_calibration_dataset(calib_dataset)
# compiler.compile()
# compiler.gencode(KMODEL_PATH)

# ---- 5. 验证 ----
print("[5/5] Done! kmodel saved to:", KMODEL_PATH)
```

**警告：以上代码是模板。** nncase API 在不同版本中差异较大。
请以 nncase 当前版本的官方示例代码为准。

关键参数说明：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| target | 目标芯片 | "k230" |
| input_shape | 模型输入尺寸 | (1, 3, 320, 320) |
| calibration_images | 校准图片数量 | 100-300 |
| quantization | 量化方式 | int8 后训练量化 (PTQ) |
| preprocess | 是否在模型中包含预处理 | True (推荐) |
| input_mean | 输入均值 | [0, 0, 0] 或 [127.5, 127.5, 127.5] |
| input_std | 输入标准差 | [255, 255, 255] 或 [255, 255, 255] |
| input_layout | 输入数据排布 | "NCHW" |
| output_layout | 输出数据排布 | 根据模型 |
| inference_type | 推理数据类型 | "uint8" 或 "float" |

## 输出验证

转换成功后应得到：
- `study_yolov8n_320.kmodel` — 用于部署到 K230D

验证：
```bash
ls -la study_yolov8n_320.kmodel
# 文件大小通常在 1-5 MB 之间
```

## 拷贝到 K230D

### 方式 1: 使用 CanMV IDE K230 的文件管理
在 IDE 中连接设备后，通过文件管理器将 kmodel 上传到 `/sdcard/models/`

### 方式 2: 使用 SD 卡
将 kmodel 拷贝到 SD 卡的 `models/` 目录，插入 K230D

### 方式 3: 使用串口文件传输
通过 K230D 的 MicroPython REPL 或专用工具传输

### 目标路径
```
/sdcard/models/study_yolov8n_320.kmodel
/sdcard/models/study_labels.txt
```

如果 `/sdcard/` 不可用，尝试 `/data/models/`

## labels 文件

`/sdcard/models/study_labels.txt` 内容：
```
person
phone
book
laptop
```

一行一个类别，顺序与训练时的 `dataset.yaml` 的 `names` 一致。

## 常见错误排查

| 错误 | 可能原因 | 解决方法 |
|------|----------|----------|
| "Unsupported ONNX op: xxx" | nncase 不支持某个 ONNX 算子 | 降低 opset，或使用 supported ops |
| "Shape mismatch" | 输入 shape 配置错误 | 确认是 (1,3,320,320) NCHW |
| "Out of memory" | 校准集图片太大 | 检查预处理 resize 是否正确 |
| "Quantization error" | 校准数据量不足或分布不均匀 | 增加校准集数量，确保覆盖所有类别 |
| kmodel 加载失败 | 转换参数与固件不匹配 | 检查 nncase 版本与固件版本对应关系 |
| 推理结果为全 0 | 预处理参数错误 | 检查 input_mean/input_std 是否与训练时一致 |

## 参考资源

- K230 AI 开发文档: https://www.kendryte.com/k230_canmv/zh/main/ai_dev_doc.html
- nncase GitHub: https://github.com/kendryte/nncase
- Canaan 开发者社区: https://developer.canaan-creative.com/
