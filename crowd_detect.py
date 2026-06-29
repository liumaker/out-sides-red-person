"""
人员聚集检测脚本 - 检测视频/图片中的人员聚集行为
用法：
    python crowd_detect.py --source path/to/video.mp4                # 视频聚集检测
    python crowd_detect.py --source path/to/video.mp4 --show        # 实时显示
    python crowd_detect.py --source path/to/image.jpg               # 图片聚集检测
    python crowd_detect.py --source path/to/images/                 # 图片目录

参数：
    --distance N     聚集距离阈值，相邻N个人体宽度内视为靠近（默认1.5）
    --min-people N   最少多少人算聚集（默认3）
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def find_groups(centers, distance_threshold):
    """
    基于空间距离的连通分量法，将人员分组。
    
    centers: np.array (N, 2)，每个人的中心点坐标 (cx, cy)
    distance_threshold: 距离阈值（像素），小于此距离视为"靠近"
    
    返回: list[list[int]]，每个子列表是一组人员的索引
    """
    n = len(centers)
    if n == 0:
        return []

    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i]:
            continue
        group = []
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            group.append(node)
            for j in range(n):
                if not visited[j]:
                    dist = np.linalg.norm(centers[node] - centers[j])
                    if dist < distance_threshold:
                        stack.append(j)
        groups.append(group)

    return groups


def draw_results(frame, boxes, is_gathering):
    """
    在帧上绘制检测结果，聚集人员用红色框，非聚集用绿色框。
    
    frame: np.array, BGR 图像
    boxes: list[tuple], 每个为 (x1, y1, x2, y2)
    is_gathering: list[bool]，是否属于聚集
    """
    for i, (box, gathering) in enumerate(zip(boxes, is_gathering)):
        x1, y1, x2, y2 = box
        color = (0, 0, 255) if gathering else (0, 255, 0)  # 红=聚集, 绿=正常
        label = "Gathering" if gathering else "Person"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 左上角统计信息
    gathering_count = sum(is_gathering)
    total = len(boxes)
    cv2.putText(frame, f"Total: {total}  Gathering: {gathering_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame


def process_frame(frame, model, conf, iou, imgsz, device,
                  distance_factor, min_people):
    """
    处理单帧：检测人员 -> 判断聚集 -> 绘制结果。
    
    返回: (annotated_frame, gathering_count, total_count)
    """
    results = model(frame, conf=conf, iou=iou, imgsz=imgsz,
                    device=device, classes=[0], verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        # 无检测结果
        h, w = frame.shape[:2]
        cv2.putText(frame, "Total: 0  Gathering: 0",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame, 0, 0

    # 提取边界框和中心点
    boxes = []
    centers = []
    for box in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        boxes.append((x1, y1, x2, y2))
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx, cy])

    centers = np.array(centers)
    n = len(centers)

    # 自适应距离阈值：基于人体平均宽度
    widths = np.array([b[2] - b[0] for b in boxes])
    avg_person_width = np.mean(widths) if len(widths) > 0 else 100.0
    distance_threshold = distance_factor * avg_person_width

    # 检测聚集组
    groups = find_groups(centers, distance_threshold)

    # 标记每个人是否属于聚集
    is_gathering = [False] * n
    gathering_count = 0
    for group in groups:
        if len(group) >= min_people:
            for idx in group:
                is_gathering[idx] = True
                gathering_count += 1

    # 对每个聚集组绘制凸包轮廓
    for group in groups:
        if len(group) >= min_people:
            pts = np.array([centers[i] for i in group], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.polylines(frame, [hull], True, (0, 255, 255), 2)
            # 在聚集组上方标注人数
            mx, my = np.mean(pts, axis=0).astype(int)
            cv2.putText(frame, f"x{len(group)}", (mx - 15, my - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    frame = draw_results(frame, boxes, is_gathering)
    return frame, gathering_count, n


def main():
    parser = argparse.ArgumentParser(description="人员聚集检测")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--source", type=str, default="dataset/test/images",
                        help="输入源：视频路径、图片路径、图片目录")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图片尺寸")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备 (cuda / cpu)")
    parser.add_argument("--distance", type=float, default=1.5,
                        help="聚集距离阈值（人体宽度的倍数，默认1.5）")
    parser.add_argument("--min-people", type=int, default=3,
                        help="最少多少人算聚集（默认3）")
    parser.add_argument("--show", action="store_true",
                        help="是否实时显示推理画面")
    parser.add_argument("--output", type=str, default=None,
                        help="输出视频/图片路径（默认保存到 runs/detect/crowd/）")
    args = parser.parse_args()

    # 检查模型
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {args.model}")
        return

    print(f"[INFO] 加载模型: {args.model}")
    print(f"[INFO] 聚集参数: 距离={args.distance}x人体宽度, 最少{args.min_people}人")
    model = YOLO(str(args.model))

    source_path = Path(args.source)

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("runs/detect/crowd")
    output_path.mkdir(parents=True, exist_ok=True)

    # ==================== 视频处理 ====================
    if str(args.source).endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        cap = cv2.VideoCapture(str(args.source))
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {args.source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_video_path = output_path / f"{source_path.stem}_crowd.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        print(f"[INFO] 处理视频: {args.source}")
        print(f"[INFO] 帧率: {fps:.1f}, 总帧数: {total_frames}")
        print(f"[INFO] 输出: {out_video_path}")

        frame_idx = 0
        gathering_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, g_count, t_count = process_frame(
                frame, model, args.conf, args.iou, args.imgsz, args.device,
                args.distance, args.min_people
            )

            if g_count > 0:
                gathering_frames += 1

            out.write(annotated)

            if args.show:
                cv2.imshow("Crowd Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  进度: {frame_idx}/{total_frames}")

        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()

        print(f"\n===== 视频处理完成 =====")
        print(f"总帧数: {total_frames}")
        print(f"出现聚集的帧数: {gathering_frames}")
        print(f"聚集占比: {gathering_frames / max(total_frames, 1) * 100:.1f}%")
        print(f"输出: {out_video_path}")

    # ==================== 图片处理 ====================
    elif str(args.source).endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        frame = cv2.imread(str(args.source))
        if frame is None:
            print(f"[ERROR] 无法读取图片: {args.source}")
            return

        print(f"[INFO] 处理图片: {args.source}")
        annotated, g_count, t_count = process_frame(
            frame, model, args.conf, args.iou, args.imgsz, args.device,
            args.distance, args.min_people
        )

        out_img_path = output_path / f"{source_path.stem}_crowd.jpg"
        cv2.imwrite(str(out_img_path), annotated)
        print(f"\n===== 图片处理完成 =====")
        print(f"检测人数: {t_count}, 聚集人数: {g_count}")
        print(f"输出: {out_img_path}")

        if args.show:
            cv2.imshow("Crowd Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # ==================== 目录处理 ====================
    elif source_path.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = sorted([f for f in source_path.iterdir()
                             if f.suffix.lower() in image_exts])
        print(f"[INFO] 图片目录: {args.source} ({len(image_files)} 张)")

        for i, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  [WARN] 跳过: {img_path}")
                continue

            annotated, g_count, t_count = process_frame(
                frame, model, args.conf, args.iou, args.imgsz, args.device,
                args.distance, args.min_people
            )

            out_img_path = output_path / f"{img_path.stem}_crowd.jpg"
            cv2.imwrite(str(out_img_path), annotated)

            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{len(image_files)}")

        print(f"\n===== 目录处理完成 =====")
        print(f"共处理 {len(image_files)} 张图片")
        print(f"输出目录: {output_path}")

    else:
        print(f"[ERROR] 不支持的输入源: {args.source}")
        print("支持: .mp4/.avi/.mov/.mkv/.webm / .jpg/.png/.bmp / 图片目录")


if __name__ == "__main__":
    main()
