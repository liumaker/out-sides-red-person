"""
YOLOv8 推理脚本 - 支持图片、视频、目录推理
用法：
    python infer.py --source path/to/image.jpg          # 单张图片
    python infer.py --source path/to/images/            # 图片目录
    python infer.py --source path/to/video.mp4          # 视频文件
    python infer.py --source video.mp4 --show           # 视频推理并实时显示
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Person Detection Inference")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt",
                        help="模型权重路径（YOLO 默认路径）")
    parser.add_argument("--source", type=str, default="dataset/test/images",
                        help="输入源：图片路径、图片目录、或视频路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图片尺寸")
    parser.add_argument("--save_txt", action="store_true",
                        help="是否保存标注结果为 txt 文件")
    parser.add_argument("--save_img", action="store_true", default=True,
                        help="是否保存标注后的图片/视频")
    parser.add_argument("--show", action="store_true",
                        help="是否实时显示推理画面")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (cuda / cpu)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {args.model}")
        print("请先训练模型或指定正确的模型路径")
        return

    print(f"[INFO] 加载模型: {args.model}")
    model = YOLO(str(args.model))

    print(f"[INFO] 推理源: {args.source}")
    print(f"[INFO] 置信度阈值: {args.conf}, IoU阈值: {args.iou}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save_img,
        save_txt=args.save_txt,
        show=args.show,
    )

    print(f"[INFO] 推理完成，共处理 {len(results)} 张图片")
    print(f"[INFO] 结果保存在 runs/detect/predict/")


if __name__ == "__main__":
    main()
