"""
数据集准备脚本 — Stage 1 (3类: person/phone/book)
一步完成：
  1. RGBA → RGB 转换 (全部转 JPG)
  2. 合并 train1 + train2 → images/train/
  3. 拷贝 labels → labels/train/ + labels/val/
  4. 修复 4 个错误标注文件
  5. 跳过孤立标注（有 label 无 image）
  6. 生成 dataset.yaml (nc=3)
"""

import os
import sys
import shutil
from PIL import Image

# ---- 路径配置 ----
BASE = "F:/嵌入式开发/嵌入式ai学习/结课大作业"
SRC_IMAGES = os.path.join(BASE, "images")
SRC_LABELS = os.path.join(BASE, "labels")
OUT_DIR = os.path.join(BASE, "dataset_study_yolo")

# 错误标注修复
FIXES = {
    "1276033.txt": [
        # 第7行 x_center 越界，修正为 1.0
        (6, lambda parts: ["0", "1.0", parts[2], parts[3], parts[4]]),
    ],
    "3002021.txt": [
        # 第7行 width=0.0，删除该行
        (6, None),
    ],
    "3005076.txt": [
        # 第9行 width=0.0，删除该行
        (8, None),
    ],
    "3005163.txt": [
        # 第7行 width=0.0，删除该行
        (6, None),
    ],
}


def fix_label(filepath, fname):
    """修复已知错误标注。返回修复后的行列表。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if fname not in FIXES:
        return lines

    fixes = FIXES[fname]
    for line_idx, action in fixes:
        if line_idx < len(lines):
            if action is None:
                # 删除该行
                lines[line_idx] = None
            else:
                parts = lines[line_idx].strip().split()
                new_parts = action(parts)
                lines[line_idx] = ' '.join(new_parts) + '\n'

    # 过滤 None（已删除的行）
    lines = [l for l in lines if l is not None]
    return lines


def convert_and_copy_images(src_dirs, dst_dir, prefix=""):
    """
    复制并转换图片: RGBA→RGB, PNG→JPG。
    返回: {original_stem: new_filename}
    """
    os.makedirs(dst_dir, exist_ok=True)
    mapping = {}
    count = 0
    rgba_count = 0

    for src_dir in src_dirs:
        if not os.path.isdir(src_dir):
            continue
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith('.png'):
                continue
            stem = os.path.splitext(fname)[0]
            src_path = os.path.join(src_dir, fname)

            try:
                img = Image.open(src_path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                    rgba_count += 1
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                    rgba_count += 1

                # 保存为 JPG (质量 95)
                new_name = stem + '.jpg'
                dst_path = os.path.join(dst_dir, new_name)
                img.save(dst_path, 'JPEG', quality=95)
                mapping[stem] = new_name
                count += 1

                if count % 500 == 0:
                    print(f"  ... {count} 张已处理")

            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")

    print(f"  转换完成: {count} 张 ({rgba_count} 张从 RGBA 转 RGB)")
    return mapping


def copy_labels(src_dir, dst_dir, image_mapping, is_train=True):
    """
    复制标注文件。
    跳过孤立标注（无对应图片）。
    修复已知错误标注。
    """
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    skip = 0

    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith('.txt'):
            continue
        stem = os.path.splitext(fname)[0]

        # 检查是否有对应图片
        if stem not in image_mapping:
            print(f"  [SKIP] 孤立标注: {fname}")
            skip += 1
            continue

        src_path = os.path.join(src_dir, fname)

        # 如果需要修复
        if fname in FIXES:
            lines = fix_label(src_path, fname)
            dst_path = os.path.join(dst_dir, fname)
            with open(dst_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"  [FIXED] {fname}")
        else:
            shutil.copy2(src_path, dst_dir)

        count += 1

    return count, skip


def generate_dataset_yaml(out_dir):
    """生成 dataset.yaml (nc=3)"""
    content = """# YOLOv8 数据集配置 — Stage 1 (3类)
# 类别: person, phone, book (暂时不含 laptop)

path: .
train: images/train
val: images/val
nc: 3
names:
  0: person
  1: phone
  2: book
"""
    path = os.path.join(out_dir, 'dataset.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"dataset.yaml 已写入: {path}")


def main():
    print("=" * 60)
    print("Stage 1 数据集准备")
    print("=" * 60)

    # 清理旧输出
    if os.path.isdir(OUT_DIR):
        print(f"清理旧目录: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    # ---- 1. 转换并复制 Training 图片 ----
    print("\n[1/4] 处理 Training 图片 (train1 + train2)...")
    train_src = [
        os.path.join(SRC_IMAGES, "train1"),
        os.path.join(SRC_IMAGES, "train2"),
    ]
    train_map = convert_and_copy_images(
        train_src,
        os.path.join(OUT_DIR, "images", "train"),
    )

    # ---- 2. 转换并复制 Validation 图片 ----
    print("\n[2/4] 处理 Validation 图片...")
    val_src = [os.path.join(SRC_IMAGES, "val")]
    val_map = convert_and_copy_images(
        val_src,
        os.path.join(OUT_DIR, "images", "val"),
    )

    # ---- 3. 复制并修复 Training 标注 ----
    print("\n[3/4] 处理 Training 标注...")
    train_labels_src = os.path.join(SRC_LABELS, "train")
    train_cnt, train_skip = copy_labels(
        train_labels_src,
        os.path.join(OUT_DIR, "labels", "train"),
        train_map,
        is_train=True,
    )
    print(f"  Train 标注: {train_cnt} 个, 跳过孤立: {train_skip} 个")

    # ---- 4. 复制 Validation 标注 ----
    val_labels_src = os.path.join(SRC_LABELS, "val")
    val_cnt, val_skip = copy_labels(
        val_labels_src,
        os.path.join(OUT_DIR, "labels", "val"),
        val_map,
        is_train=False,
    )
    print(f"  Val 标注: {val_cnt} 个, 跳过孤立: {val_skip} 个")

    # ---- 5. 生成 dataset.yaml ----
    print("\n[4/4] 生成 dataset.yaml...")
    generate_dataset_yaml(OUT_DIR)

    # ---- 汇总 ----
    train_imgs = len(os.listdir(os.path.join(OUT_DIR, "images", "train")))
    val_imgs = len(os.listdir(os.path.join(OUT_DIR, "images", "val")))
    train_lbls = len(os.listdir(os.path.join(OUT_DIR, "labels", "train")))
    val_lbls = len(os.listdir(os.path.join(OUT_DIR, "labels", "val")))

    print("\n" + "=" * 60)
    print("数据集准备完成!")
    print(f"  Train: {train_imgs} 图片, {train_lbls} 标注")
    print(f"  Val:   {val_imgs} 图片, {val_lbls} 标注")
    print(f"  输出目录: {OUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
