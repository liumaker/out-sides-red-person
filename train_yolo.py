"""
YOLOv8 训练脚本 - Person Detection

用法：
    # 从头训练（使用预训练权重）
    python train_yolo.py

    # 断点续传（从上次中断的 checkpoint 继续训练）
    python train_yolo.py --resume

    # 用已有模型在新数据集上继续训练（微调）
    python train_yolo.py --weights runs/detect/train/weights/best.pt --data 新数据集/data.yaml --epochs 50
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Person Detection Training")

    # 模型与数据
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                        help="初始权重路径（从头训练用预训练权重，微调用已有模型）")
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                        help="数据集配置文件路径")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备 (cuda / cpu)")

    # 断点续传
    parser.add_argument("--resume", action="store_true",
                        help="断点续传：从上次中断的 checkpoint 继续训练")

    args = parser.parse_args()

    # ==================== 加载模型 ====================
    # --resume 模式下 weights 必须是上次训练的 last.pt 路径
    # YOLOv8 的 resume=True 会自动从 runs/detect/train/weights/last.pt 恢复
    model = YOLO(args.weights)

    # ==================== 训练 ====================
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,              # 断点续传
        patience=20,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        augment=True,
        val=True,
        save=True,
        save_period=10,
        workers=4,
        amp=True,
    )

    # ==================== 评估 ====================
    print("\n===== 在测试集上评估最佳模型 =====")
    best_model_path = str(results.save_dir / "weights" / "best.pt")
    model = YOLO(best_model_path)
    metrics = model.val(data=args.data, split="test")
    print(f"mAP50:   {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\n===== 训练完成 =====")
    print(f"最佳模型: {best_model_path}")
    print(f"训练输出: {results.save_dir}")


if __name__ == "__main__":
    main()
