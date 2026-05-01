from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# =========================
# PATHS
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "configs" / "data.yaml"
MODIFIED_YAML = ROOT / "models" / "cadiyolo_modified.yaml"

RUNS_DIR = ROOT / "runs" / "detect"

# =========================
# TRAINING SETTINGS
# =========================
EPOCHS = 50
IMGSZ = 640
BATCH = 16

# Force CUDA GPU
DEVICE = 0

BASELINE_NAME = "baseline_yolov8s3"

# Latest modified run folder you already trained
MODIFIED_NAME = "cadiyolo_modified223"
MODIFIED_START_CKPT = RUNS_DIR / "cadiyolo_modified223" / "weights" / "best.pt"


# =========================
# HELPERS
# =========================
def best_weights(run_name: str) -> Path:
    return RUNS_DIR / run_name / "weights" / "best.pt"


def last_weights(run_name: str) -> Path:
    return RUNS_DIR / run_name / "weights" / "last.pt"


# =========================
# STEP 1: BASELINE TRAINING
# =========================
def train_baseline():
    print("\n=== STEP 1: TRAIN BASELINE YOLOv8 ===")
    model = YOLO("yolov8s.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(RUNS_DIR),
        name=BASELINE_NAME,
        plots=True,
        save=True,
    )

    print(f"Baseline training finished: {best_weights(BASELINE_NAME)}")


# =========================
# STEP 2 + 3 + 4 + 5: MODIFIED MODEL
# =========================
def train_modified():
    print("\n=== STEP 2 TO 5: TRAIN MODIFIED MODEL FROM CHECKPOINT ===")

    modified_best = best_weights(MODIFIED_NAME)
    modified_last = last_weights(MODIFIED_NAME)

    # Prefer the checkpoint from the previous modified run
    if MODIFIED_START_CKPT.exists():
        ckpt_path = MODIFIED_START_CKPT
    elif modified_best.exists():
        ckpt_path = modified_best
    elif modified_last.exists():
        ckpt_path = modified_last
    else:
        ckpt_path = None

    if ckpt_path is not None:
        print(f"Loading checkpoint: {ckpt_path}")
        model = YOLO(str(ckpt_path))
    else:
        print("No checkpoint found. Starting from YAML + pretrained yolov8s.pt")
        model = YOLO(str(MODIFIED_YAML))
        try:
            model.load("yolov8s.pt")
            print("Loaded pretrained yolov8s.pt weights into modified model.")
        except Exception as e:
            print(f"Pretrained weight loading skipped: {e}")

    # Fine-tuning on the balanced dataset
    model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=768,
        batch=BATCH,
        device=DEVICE,
        project=str(RUNS_DIR),
        name=MODIFIED_NAME,
        exist_ok=True,

        # Fine-tuning settings
        lr0=0.0001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,

        # Class focus after balancing
        cls=1.6,
        box=7.5,
        dfl=1.5,

        # Controlled augmentation
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5,
        translate=0.05,
        scale=0.3,
        shear=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.0,
        close_mosaic=10,

        # Stability
        patience=30,
        workers=4,
        cache="disk",
        amp=True,
    )

    print(f"Modified training finished: {best_weights(MODIFIED_NAME)}")


# =========================
# ONE-CALL PIPELINE
# =========================
def run_all():
    train_baseline()
    train_modified()

    print("\n=== TRAINING COMPLETE ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "baseline_train",
            "modified_train",
        ],
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_all()
    elif args.mode == "baseline_train":
        train_baseline()
    elif args.mode == "modified_train":
        train_modified()


if __name__ == "__main__":
    main()