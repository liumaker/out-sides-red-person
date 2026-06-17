"""
YOLOv8 训练脚本 - Person Detection
数据集格式：YOLO 格式 (单类别: Person)
"""

from ultralytics import YOLO

# ==================== 配置 ====================
DATA_YAML = "dataset/data.yaml"          # 数据集配置文件（相对路径）
MODEL_NAME = "yolov8n.pt"                # 预训练模型 (n/s/m/l/x)
EPOCHS = 100                             # 训练轮数
BATCH_SIZE = 16                          # 批次大小
IMGSZ = 640                              # 输入图片尺寸
DEVICE = "cuda"                          # 使用 GPU 训练
PROJECT = "runs/train"                   # 输出目录（相对路径）
EXPERIMENT_NAME = "person_detection"     # 实验名称

# ==================== 初始化模型 ====================
model = YOLO(MODEL_NAME)

# ==================== 训练 ====================
results = model.train(
    data=DATA_YAML,                      # 数据集配置文件
    epochs=EPOCHS,                       # 训练轮数
    batch=BATCH_SIZE,                    # 批次大小
    imgsz=IMGSZ,                         # 输入图片尺寸
    device=DEVICE,                       # 训练设备
    project=PROJECT,                     # 输出项目目录
    name=EXPERIMENT_NAME,                # 实验名称
    patience=20,                         # Early stopping 轮数
    lr0=0.01,                            # 初始学习率
    lrf=0.01,                            # 最终学习率 (lr0 * lrf)
    warmup_epochs=3,                     # 预热轮数
    augment=True,                        # 是否使用数据增强
    val=True,                            # 是否在训练中验证
    save=True,                           # 保存模型
    save_period=10,                      # 每 N 轮保存一次模型
    workers=4,                           # 数据加载线程数
    amp=True,                            # 混合精度训练
)

# ==================== 评估 ====================
print("\n===== 在测试集上评估最佳模型 =====")
best_model_path = f"{PROJECT}/{EXPERIMENT_NAME}/weights/best.pt"
model = YOLO(best_model_path)
metrics = model.val(data=DATA_YAML, split="test")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# ==================== 导出模型 ====================
print("\n===== 导出模型 =====")
model.export(format="onnx")  # 导出为 ONNX 格式

print("\n===== 训练完成 =====")
print(f"最佳模型: {best_model_path}")
