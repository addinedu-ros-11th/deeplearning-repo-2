from ultralytics import YOLO
import torch
from pathlib import Path

DATASET_YAML = "/home/clyde/dev_ws/deeplearning-repo-2/data/CCTV Fall and Lying Down Detection/laying_dataset/dataset.yaml"
PROJECT_DIR = "/home/clyde/dev_ws/deeplearning-repo-2/runs/train"
RUN_NAME = "cctv_fall_laying_pose_v8n"


def main():
    if not Path(DATASET_YAML).exists():
        raise FileNotFoundError(f"dataset.yaml not found: {DATASET_YAML}")

    model_name = "yolov8s-pose.pt" if torch.cuda.is_available() else "yolov8n-pose.pt"
    device = 0 if torch.cuda.is_available() else "cpu"
    batch = 16 if torch.cuda.is_available() else 4

    print(f"Using model={model_name}, device={device}, batch={batch}")

    model = YOLO(model_name)
    model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=640,
        batch=batch,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        resume=False,
        seed=42,
        patience=20,
        cache=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
