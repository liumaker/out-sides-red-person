"""
YOLOv8 验证脚本 - 在测试集上评估模型精度
用法：
    python val.py                                         # 默认只评估 Person
    python val.py --model runs/detect/train/weights/best.pt
    python val.py --classes 0 1                           # 评估多个类别
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Person Detection Validation")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt",
                        help="模型权重路径（YOLO 默认路径）")
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                        help="数据集配置文件路径")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="评估数据集划分")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图片尺寸")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (cuda / cpu)")
    parser.add_argument("--classes", type=int, nargs="+", default=[0],
                        help="只评估指定类别的目标，默认只评估 Person (0)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {args.model}")
        print("请先训练模型或指定正确的模型路径")
        return

    print(f"[INFO] 加载模型: {args.model}")
    model = YOLO(str(model_path))

    class_names = {
        0: "Person",
        1: "Car",
        2: "Bicycle",
        3: "OtherVehicle",
        4: "DontCare",
    }
    print(f"[INFO] 评估类别: {[class_names.get(c, f'未知({c})') for c in args.classes]}")
    print(f"[INFO] 数据集: {args.data}")
    print(f"[INFO] 评估划分: {args.split}")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        classes=args.classes,
    )

    print("\n===== 评估结果 =====")
    print(f"mAP50:   {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"精度 (P): {metrics.box.p:.4f}")
    print(f"召回 (R): {metrics.box.r:.4f}")
    print(f"结果保存在 runs/detect/val/")


if __name__ == "__main__":
    main()
