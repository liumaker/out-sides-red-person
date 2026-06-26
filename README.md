# out-sides-red-person

YOLOv8 行人检测（Person Detection）项目。

## 训练

### 1. 从头训练（使用预训练权重）

```bash
python train_yolo.py
```

使用 `yolov8n.pt` 预训练权重，在默认数据集 `dataset/data.yaml` 上训练 100 个 epoch。

### 2. 断点续传

训练中途中断后，从上次保存的 checkpoint 继续训练：

```bash
python train_yolo.py --resume
```

会自动从 `runs/detect/train/weights/last.pt` 恢复优化器状态和已完成的 epoch 数。

### 3. 用已有模型在新数据集上微调

```bash
python train_yolo.py --weights runs/detect/train/weights/best.pt --data path/to/data.yaml --epochs 50
```

加载已训练好的 `best.pt` 权重，在新数据集上继续训练。适用于迁移学习场景。

### 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `yolov8n.pt` | 初始权重路径 |
| `--data` | `dataset/data.yaml` | 数据集配置 |
| `--epochs` | `100` | 训练轮数 |
| `--batch` | `16` | 批次大小 |
| `--imgsz` | `640` | 输入图片尺寸 |
| `--device` | `cuda` | 训练设备 |
| `--resume` | `False` | 断点续传 |

## 推理

```bash
# 单张图片
python infer.py --source path/to/image.jpg

# 图片目录
python infer.py --source path/to/images/

# 视频文件
python infer.py --source path/to/video.mp4

# 视频推理 + 实时显示
python infer.py --source video.mp4 --show
```

## 验证

```bash
python val.py
```
