# src/detect_regions.py
from typing import Dict, List, Tuple

import numpy as np

# (x1, y1, x2, y2, confidence)
Box = Tuple[int, int, int, int, float]


def detect_regions(
    image: np.ndarray,
    model_path: str = "runs/detect/models/yolo_regions/phieu_gui_v1/weights/best.pt",
    conf_threshold: float = 0.4,
    imgsz: int = 1024,
) -> Dict[str, List[Box]]:
    import ultralytics  # noqa: PLC0415 — lazy import keeps patch("ultralytics.YOLO") effective
    model = ultralytics.YOLO(model_path)
    results = model(image, conf=conf_threshold, imgsz=imgsz, verbose=False)

    regions: Dict[str, List[Box]] = {}
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls.item())]
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            regions.setdefault(cls_name, []).append((x1, y1, x2, y2, conf))

    return regions


def train_yolo(
    data_yaml: str = "data/yolo_dataset/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 1024,
    batch: int = 8,
    project: str = "models/yolo_regions",
    name: str = "phieu_gui_v1",
) -> str:
    import ultralytics  # noqa: PLC0415
    model = ultralytics.YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
    )
    best_path = f"runs/detect/{project}/{name}/weights/best.pt"
    print(f"Training done. Best weights: {best_path}")
    return best_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    train_p = subparsers.add_parser("train", help="Fine-tune YOLOv8 trên dataset phiếu")
    train_p.add_argument("--data", default="data/yolo_dataset/data.yaml")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch", type=int, default=8)
    train_p.add_argument("--imgsz", type=int, default=1024)

    args = parser.parse_args()
    if args.cmd == "train":
        train_yolo(data_yaml=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
    else:
        parser.print_help()
