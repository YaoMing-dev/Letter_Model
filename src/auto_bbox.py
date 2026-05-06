# src/auto_bbox.py
"""
auto_bbox.py — Tu dong tao YOLO bounding box labels bang Grounding DINO (Option B).
Detect 7 vung tren phieu gui hang, output .txt YOLO format + phan chia train/val.

Chay:
    python src/auto_bbox.py --img_dir data/raw_images --out_dir data/yolo_dataset
"""
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pillow_heif
import torch
from PIL import Image

pillow_heif.register_heif_opener()
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

YOLO_CLASSES = [
    "sender_block",
    "receiver_block",
    "tracking_number",
    "address_block",
    "content_block",
    "fee_block",
    "datetime_block",
]

CLASS_ID = {name: i for i, name in enumerate(YOLO_CLASSES)}

_GDINO_MODEL = "IDEA-Research/grounding-dino-base"

_PROMPTS: Dict[str, str] = {
    "sender_block":    "sender name phone address information.",
    "receiver_block":  "receiver name phone address information.",
    "tracking_number": "tracking number barcode waybill code.",
    "address_block":   "delivery address province district ward.",
    "content_block":   "package content goods description items.",
    "fee_block":       "shipping fee cod amount collection charge.",
    "datetime_block":  "date time sent timestamp.",
}


def _load_model(device: str):
    processor = AutoProcessor.from_pretrained(_GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(_GDINO_MODEL).to(device)
    model.eval()
    return processor, model


def _detect_one_class(
    processor,
    model,
    image: Image.Image,
    cls_name: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> List[Tuple[float, float, float, float]]:
    text = _PROMPTS[cls_name]
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    w_img, h_img = image.size
    boxes = []
    for x1, y1, x2, y2 in results["boxes"].tolist():
        cx = (x1 + x2) / 2 / w_img
        cy = (y1 + y2) / 2 / h_img
        bw = (x2 - x1) / w_img
        bh = (y2 - y1) / h_img
        boxes.append((cx, cy, bw, bh))
    return boxes


def label_image(
    img_path: str,
    processor,
    model,
    device: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> Dict[str, List[Tuple[float, float, float, float]]]:
    image = Image.open(img_path).convert("RGB")
    results = {}
    for cls_name in YOLO_CLASSES:
        boxes = _detect_one_class(
            processor, model, image, cls_name, device, box_threshold, text_threshold
        )
        if boxes:
            results[cls_name] = boxes
    return results


def _write_yolo_label(
    out_path: Path,
    detections: Dict[str, List[Tuple[float, float, float, float]]],
) -> None:
    lines = []
    for cls_name, boxes in detections.items():
        cls_id = CLASS_ID[cls_name]
        for cx, cy, bw, bh in boxes:
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def batch_label_and_split(
    img_dir: str,
    out_dir: str,
    val_ratio: float = 0.2,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    seed: int = 42,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Grounding DINO on {device}...")
    processor, model = _load_model(device)

    exts = {".jpg", ".jpeg", ".png", ".heic"}
    images = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts)
    if not images:
        print(f"Khong tim thay anh nao trong {img_dir}")
        return

    random.seed(seed)
    shuffled = list(images)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_set = set(shuffled[:n_val])

    for split in ("train", "val"):
        (Path(out_dir) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "labels" / split).mkdir(parents=True, exist_ok=True)

    ok = 0
    for img_path in images:
        split = "val" if img_path in val_set else "train"
        dst_lbl = Path(out_dir) / "labels" / split / (img_path.stem + ".txt")
        existing = [
            Path(out_dir) / "labels" / s / (img_path.stem + ".txt")
            for s in ("train", "val")
        ]
        if any(p.exists() for p in existing):
            print(f"Skip (exists): {img_path.name}")
            continue

        print(f"[{split}] {img_path.name} ...", end=" ", flush=True)
        try:
            detections = label_image(
                str(img_path), processor, model, device, box_threshold, text_threshold
            )
            shutil.copy2(img_path, Path(out_dir) / "images" / split / img_path.name)
            _write_yolo_label(dst_lbl, detections)
            n_boxes = sum(len(v) for v in detections.values())
            print(f"OK — {n_boxes} boxes ({len(detections)} classes)")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone: {ok} anh labeled")
    print(f"Train: {len(images) - n_val} | Val: {n_val}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="data/raw_images")
    parser.add_argument("--out_dir", default="data/yolo_dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--box_threshold", type=float, default=0.35,
                        help="Nguong confidence cho box detection")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="Nguong confidence cho text matching")
    args = parser.parse_args()
    batch_label_and_split(
        args.img_dir, args.out_dir, args.val_ratio,
        args.box_threshold, args.text_threshold,
    )
