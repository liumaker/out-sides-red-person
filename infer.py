"""
YOLOv8 推理脚本 - 支持图片、视频、目录推理
用法：
    python infer.py --source path/to/image.jpg          # 单张图片，只检测 Person
    python infer.py --source path/to/images/            # 图片目录
    python infer.py --source path/to/video.mp4          # 视频文件
    python infer.py --source video.mp4 --show           # 视频推理并实时显示
    python infer.py --source image.jpg --classes 0 1    # 检测多个类别（Person 和 Car）
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
    parser.add_argument("--classes", type=int, nargs="+", default=[0],
                        help="只检测指定类别的目标，默认只检测 Person (0)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {args.model}")
        print("请先训练模型或指定正确的模型路径")
        return

    print(f"[INFO] 加载模型: {args.model}")
    model = YOLO(str(args.model))

    class_names = {
        0: "Person",
        1: "Car",
        2: "Bicycle",
        3: "OtherVehicle",
        4: "DontCare",
    }
    print(f"[INFO] 检测类别: {[class_names.get(c, f'未知({c})') for c in args.classes]}")
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
        classes=args.classes,
    )

    print(f"[INFO] 推理完成，共处理 {len(results)} 张图片")
    print(f"[INFO] 结果保存在 runs/detect/predict/")


if __name__ == "__main__":
    main()
