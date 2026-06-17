"""
YOLOv8 验证脚本 - 在测试集上评估模型精度
用法：
    python val.py
    python val.py --model runs/train/person_detection/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Person Detection Validation")
    parser.add_argument("--model", type=str, default="runs/train/person_detection/weights/best.pt",
                        help="模型权重路径")
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
    args = parser.parse_args()

    # 检查模型是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {args.model}")
        print("请先训练模型或指定正确的模型路径")
        return

    # 加载模型
    print(f"[INFO] 加载模型: {args.model}")
    model = YOLO(str(model_path))

    # 验证
    print(f"[INFO] 数据集: {args.data}")
    print(f"[INFO] 评估划分: {args.split}")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/val",
        name="exp",
        exist_ok=True,
    )

    # 打印结果
    print("\n===== 评估结果 =====")
    print(f"mAP50:   {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"精度 (P): {metrics.box.p:.4f}")
    print(f"召回 (R): {metrics.box.r:.4f}")
    print(f"结果保存在 runs/val/exp/")


if __name__ == "__main__":
    main()
