from ultralytics import YOLO
import torch
from pathlib import Path
import random
import shutil

DATASET_ROOT = (
    "/home/jm/dev_ws/mind_reading/deeplearning-repo-2/data/"
    "Fall & Lying Down Detection/laying_dataset"
)
DATASET_YAML = f"{DATASET_ROOT}/dataset.yaml"
PROJECT_DIR = "/home/jm/dev_ws/mind_reading/deeplearning-repo-2/runs/train"
RUN_NAME = "cctv_fall_laying_pose_v8n"


def update_dataset_yaml(dataset_yaml: Path, root_dir: Path, train_rel: str, val_rel: str) -> None:
    lines = dataset_yaml.read_text().splitlines()
    out_lines = []
    seen = {"path": False, "train": False, "val": False}

    for line in lines:
        if line.startswith("path:"):
            out_lines.append(f"path: {root_dir}")
            seen["path"] = True
        elif line.startswith("train:"):
            out_lines.append(f"train: {train_rel}")
            seen["train"] = True
        elif line.startswith("val:"):
            out_lines.append(f"val: {val_rel}")
            seen["val"] = True
        else:
            out_lines.append(line)

    if not seen["path"]:
        out_lines.insert(0, f"path: {root_dir}")
    if not seen["train"]:
        out_lines.insert(1, f"train: {train_rel}")
    if not seen["val"]:
        out_lines.insert(2, f"val: {val_rel}")

    dataset_yaml.write_text("\n".join(out_lines) + "\n")


def split_train_val(root_dir: Path, val_ratio: float = 0.2, seed: int = 42) -> None:
    images_dir = root_dir / "images"
    labels_dir = root_dir / "labels"
    train_images_dir = images_dir / "train"
    val_images_dir = images_dir / "val"
    train_labels_dir = labels_dir / "train"
    val_labels_dir = labels_dir / "val"

    if train_images_dir.exists() and val_images_dir.exists():
        return

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    rng = random.Random(seed)
    rng.shuffle(image_paths)

    val_count = max(1, int(len(image_paths) * val_ratio)) if len(image_paths) > 1 else 0
    val_set = set(image_paths[:val_count])

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if img_path in val_set:
            shutil.move(str(img_path), str(val_images_dir / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(val_labels_dir / label_path.name))
        else:
            shutil.move(str(img_path), str(train_images_dir / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(train_labels_dir / label_path.name))


def main():
    dataset_yaml = Path(DATASET_YAML)
    dataset_root = Path(DATASET_ROOT)

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {DATASET_YAML}")

    split_train_val(dataset_root, val_ratio=0.2, seed=42)
    update_dataset_yaml(
        dataset_yaml,
        dataset_root,
        train_rel="images/train",
        val_rel="images/val",
    )

    model_name = "yolov8n-pose.pt"
    device = 0 if torch.cuda.is_available() else "cpu"
    batch = 4 if torch.cuda.is_available() else 2

    print(f"Using model={model_name}, device={device}, batch={batch}")

    model = YOLO(model_name)
    model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=640,
        batch=batch,
        device=device,
        workers=2,
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
