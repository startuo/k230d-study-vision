"""
YOLO 标注文件检查脚本 — Stage 1
检查标注文件的格式正确性和类别合法性。

用法 (PC 端):
    python check_yolo_labels.py --labels_dir ./dataset_study_yolo/labels/train/

检查内容:
    1. 标注文件是否存在
    2. 每行格式是否正确: <class_id> <x_center> <y_center> <width> <height>
    3. class_id 是否在 0-3 范围内（本阶段 4 类）
    4. 坐标值是否在 0-1 范围内
    5. 宽高是否 > 0
    6. 统计各类别标注数量
"""

import os
import sys
import argparse
from collections import Counter


def check_label_file(filepath, allowed_classes=None):
    """
    检查单个标注文件。
    返回: (is_ok, errors_list)
    """
    errors = []

    if not os.path.isfile(filepath):
        return False, ["文件不存在"]

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) == 0:
        return True, []  # 空标注（无目标图片）是合法的

    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"第{line_no}行: 需要5个字段，实际{len(parts)}个: '{line}'")
            continue

        try:
            cls_id = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            errors.append(f"第{line_no}行: 字段无法解析为数字: '{line}'")
            continue

        # 检查 class_id
        if allowed_classes is not None and cls_id not in allowed_classes:
            errors.append(f"第{line_no}行: class_id={cls_id} 不在允许范围 {allowed_classes}")

        # 检查坐标范围 [0, 1]
        for name, val in [('x_center', x_c), ('y_center', y_c),
                          ('width', w), ('height', h)]:
            if val < 0.0 or val > 1.0:
                errors.append(
                    f"第{line_no}行: {name}={val} 超出 [0,1] 范围"
                )

        # 检查宽高为正
        if w <= 0:
            errors.append(f"第{line_no}行: width={w} 必须 > 0")
        if h <= 0:
            errors.append(f"第{line_no}行: height={h} 必须 > 0")

    return len(errors) == 0, errors


def count_classes(labels_dir):
    """统计所有标注文件中各类别的出现次数。"""
    class_counter = Counter()
    total_boxes = 0
    files_with_boxes = 0

    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith('.txt'):
            continue
        filepath = os.path.join(labels_dir, fname)
        with open(filepath, 'r', encoding='utf-8') as f:
            has_box = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    class_counter[cls_id] += 1
                    total_boxes += 1
                    has_box = True
            if has_box:
                files_with_boxes += 1

    return class_counter, total_boxes, files_with_boxes


def main():
    parser = argparse.ArgumentParser(description="YOLO 标注文件检查工具")
    parser.add_argument('--labels_dir', required=True,
                        help='标注文件目录')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='类别总数 (默认 4)')
    parser.add_argument('--verbose', action='store_true',
                        help='打印每个文件的错误详情')
    args = parser.parse_args()

    if not os.path.isdir(args.labels_dir):
        print(f"[ERROR] 目录不存在: {args.labels_dir}")
        sys.exit(1)

    allowed = set(range(args.num_classes))

    label_files = [f for f in os.listdir(args.labels_dir) if f.endswith('.txt')]
    print(f"检查目录: {args.labels_dir}")
    print(f"标注文件数: {len(label_files)}")
    print(f"允许类别: {allowed}")
    print("-" * 50)

    ok_count = 0
    error_count = 0
    total_errors = 0

    for fname in sorted(label_files):
        filepath = os.path.join(args.labels_dir, fname)
        is_ok, errors = check_label_file(filepath, allowed)

        if is_ok:
            ok_count += 1
        else:
            error_count += 1
            total_errors += len(errors)
            if args.verbose:
                print(f"[FAIL] {fname}")
                for e in errors:
                    print(f"       {e}")
            else:
                if error_count <= 20:
                    print(f"[FAIL] {fname}: {len(errors)} 个错误")
                    for e in errors[:3]:  # 只显示前 3 个
                        print(f"       {e}")

    print("-" * 50)
    print(f"结果: OK={ok_count}  FAIL={error_count}  总错误数={total_errors}")

    # 类别统计
    print("\n类别统计:")
    class_counter, total_boxes, files_with_boxes = count_classes(args.labels_dir)
    class_names = {0: 'person', 1: 'phone', 2: 'book', 3: 'laptop'}
    for cls_id in sorted(class_counter.keys()):
        name = class_names.get(cls_id, f'class_{cls_id}')
        count = class_counter[cls_id]
        pct = count / total_boxes * 100 if total_boxes > 0 else 0
        print(f"  {cls_id} ({name}): {count} 个框 ({pct:.1f}%)")
    print(f"  总计: {total_boxes} 个标注框 ({files_with_boxes} 个文件有目标)")

    # 检查类别缺失
    missing = allowed - set(class_counter.keys())
    if missing:
        mids = [class_names.get(c, f'class_{c}') for c in missing]
        print(f"\n[WARN] 以下类别没有任何标注: {mids}")

    if error_count > 0:
        print("\n[FAIL] 存在标注错误，请修正后再训练。")
        sys.exit(1)
    else:
        print("\n[PASS] 所有标注文件检查通过!")


if __name__ == '__main__':
    main()
