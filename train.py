"""Train YOLOv8n-seg on the human+vehicle dataset, on GPU. Run after prepare_dataset.py."""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--model", default="yolov8n-seg.pt", help="use yolov8n.pt for plain detection")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,          # RTX GPU
        project="runs",
        name="human_vehicle",
    )


if __name__ == "__main__":
    main()  # Windows spawns worker processes; must guard top-level code
