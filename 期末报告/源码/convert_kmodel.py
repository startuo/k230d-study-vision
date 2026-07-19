"""
nncase 转换脚本 — Stage 1
将 best.onnx 转换为 study_yolov8n_320.kmodel (int8 量化)

用法:
    python convert_kmodel.py

前提:
    1. pip install nncase==2.10.0
    2. 从 https://gitee.com/kendryte/nncase/releases 下载
       nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
    3. pip install nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
    4. 校准集已准备在 calib_set/ 目录下 (150张)
"""

import os
import sys
import time
import numpy as np
from PIL import Image

# ============================================================
# 配置
# ============================================================
ONNX_PATH = r"F:\嵌入式开发\嵌入式ai学习\结课大作业\study_ai_yolo_stage1\models\best.onnx"
KMODEL_PATH = r"F:\嵌入式开发\嵌入式ai学习\结课大作业\study_ai_yolo_stage1\models\study_yolov8n_320.kmodel"
CALIB_DIR = r"F:\嵌入式开发\嵌入式ai学习\结课大作业\calib_set"

INPUT_SIZE = (320, 320)       # (W, H) — 与训练一致
INPUT_SHAPE = [1, 3, 320, 320]  # NCHW
CALIB_SAMPLES = 100             # 校准图片数量 (≤150)
TARGET = "k230"

# ============================================================
# 校准数据生成器
# ============================================================
def generate_calibration_data(calib_dir, input_shape, num_samples):
    """
    从校准集目录加载图片，预处理后返回 nncase 需要的格式。
    返回: list of [np.ndarray] — 每个元素是一个 batch 的输入数据
    """
    img_files = [
        f for f in os.listdir(calib_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    img_files = sorted(img_files)[:num_samples]

    if len(img_files) < num_samples:
        print(f"[WARN] 校准集只有 {len(img_files)} 张图片，少于请求的 {num_samples} 张")

    _, C, H, W = input_shape
    data = []

    print(f"[CALIB] 加载 {len(img_files)} 张校准图片...")
    for i, fname in enumerate(img_files):
        filepath = os.path.join(calib_dir, fname)
        try:
            img = Image.open(filepath).convert('RGB')
            img = img.resize((W, H), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)       # uint8, HWC
            arr = np.transpose(arr, (2, 0, 1))           # HWC -> CHW
            arr = arr[np.newaxis, ...]                   # CHW -> NCHW (N=1)
            data.append([arr])
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(img_files)}")

    print(f"[CALIB] 加载完成, 共 {len(data)} 张")
    return data


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("nncase ONNX → KModel 转换 — Stage 1")
    print("=" * 60)
    print(f"  ONNX:     {ONNX_PATH}")
    print(f"  KModel:   {KMODEL_PATH}")
    print(f"  校准集:   {CALIB_DIR}")
    print(f"  目标:     {TARGET}")
    print(f"  输入尺寸: {INPUT_SIZE}")
    print("=" * 60)

    # 检查 ONNX 文件
    if not os.path.exists(ONNX_PATH):
        print(f"[ERROR] ONNX 文件不存在: {ONNX_PATH}")
        sys.exit(1)

    # 检查校准集
    if not os.path.isdir(CALIB_DIR):
        print(f"[ERROR] 校准集目录不存在: {CALIB_DIR}")
        sys.exit(1)

    # ---- 导入 nncase ----
    print("\n[1/5] 导入 nncase...")
    try:
        import nncase
        print(f"  nncase imported successfully")
    except ImportError:
        print("[ERROR] nncase 未安装。请先执行:")
        print("  pip install nncase==2.10.0")
        print("  然后从 https://gitee.com/kendryte/nncase/releases 下载并安装 nncase_kpu wheel")
        sys.exit(1)

    # ---- 编译选项 ----
    print("\n[2/5] 配置编译选项...")
    compile_options = nncase.CompileOptions()
    compile_options.target = TARGET
    compile_options.preprocess = True
    compile_options.input_type = 'uint8'
    compile_options.input_shape = INPUT_SHAPE
    compile_options.input_range = [0, 255]
    compile_options.input_layout = 'NCHW'
    compile_options.swapRB = False
    compile_options.mean = [0, 0, 0]
    compile_options.std = [255, 255, 255]
    compile_options.letterbox_value = 0
    compile_options.output_layout = 'NCHW'

    # ---- 导入 ONNX ----
    print("\n[3/5] 导入 ONNX 模型...")
    compiler = nncase.Compiler(compile_options)

    import_options = nncase.ImportOptions()
    with open(ONNX_PATH, 'rb') as f:
        onnx_content = f.read()
    compiler.import_onnx(onnx_content, import_options)
    print("  ONNX 导入成功")

    # ---- PTQ 量化校准 ----
    print("\n[4/5] 设置 PTQ 量化校准...")
    calib_data = generate_calibration_data(
        CALIB_DIR, INPUT_SHAPE, CALIB_SAMPLES
    )

    if len(calib_data) == 0:
        print("[ERROR] 校准数据集为空，无法进行量化")
        sys.exit(1)

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = len(calib_data)
    ptq_options.calibrate_method = 'Kld'
    ptq_options.quant_type = 'uint8'
    ptq_options.w_quant_type = 'uint8'
    ptq_options.finetune_weights_method = 'NoFineTuneWeights'
    ptq_options.set_tensor_data(calib_data)
    compiler.use_ptq(ptq_options)

    # ---- 编译 ----
    print("  开始编译 (可能需要几分钟)...")
    t0 = time.time()
    compiler.compile()
    elapsed = time.time() - t0
    print(f"  编译完成, 耗时 {elapsed:.1f}s")

    # ---- 生成 kmodel ----
    print("\n[5/5] 生成 kmodel...")
    kmodel_bytes = compiler.gencode_tobytes()

    kmodel_dir = os.path.dirname(KMODEL_PATH)
    os.makedirs(kmodel_dir, exist_ok=True)

    with open(KMODEL_PATH, 'wb') as f:
        f.write(kmodel_bytes)

    size_mb = len(kmodel_bytes) / (1024 * 1024)
    print(f"  kmodel 已保存: {KMODEL_PATH} ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("转换完成!")
    print(f"  {KMODEL_PATH}")
    print("")
    print("下一步:")
    print(f"  1. 拷贝 kmodel 到 K230D: /sdcard/models/study_yolov8n_320.kmodel")
    print(f"  2. 拷贝 labels.txt 到 K230D: /sdcard/models/study_labels.txt")
    print(f"  3. 在 CanMV IDE K230 中运行 device_canmv/main.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
