"""
test_model.py — Kiểm tra nhanh YOLO model + full pipeline
Usage:
    python test_model.py <image_path> [--model <pt_path>] [--out <output_image>]
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import pillow_heif
from PIL import Image
from dotenv import load_dotenv

pillow_heif.register_heif_opener()
load_dotenv()

COLORS = {
    "sender_block":    (255, 100,  50),
    "receiver_block":  ( 50, 200,  50),
    "tracking_number": (255, 200,   0),
    "address_block":   ( 50, 150, 255),
    "content_block":   (200,  50, 255),
    "fee_block":       (  0, 220, 220),
    "datetime_block":  (255, 120, 180),
}


def load_image_cv2(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > 2048:
        scale = 2048 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_regions(img: np.ndarray, regions: dict) -> np.ndarray:
    vis = img.copy()
    for cls_name, boxes in regions.items():
        color = COLORS.get(cls_name, (200, 200, 200))
        for x1, y1, x2, y2, conf in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Đường dẫn ảnh phiếu")
    parser.add_argument("--model", default="models/yolo_regions/phieu_gui_v1/weights/best.pt")
    parser.add_argument("--out", default="test_output.jpg", help="Lưu ảnh kết quả")
    parser.add_argument("--no-llm", action="store_true", help="Chỉ chạy YOLO+OCR, bỏ qua GPT")
    args = parser.parse_args()

    img_path = args.image
    print(f"\n{'='*55}")
    print(f"  Ảnh: {img_path}")
    print(f"  Model: {args.model}")
    print(f"{'='*55}\n")

    # ── 1. Load ảnh
    print("[1/4] Load ảnh...")
    img_bgr = load_image_cv2(img_path)
    h, w = img_bgr.shape[:2]
    print(f"      Kích thước: {w}x{h}")

    # ── 2. YOLO detect
    print("[2/4] YOLO phát hiện vùng...")
    from src.detect_regions import detect_regions
    regions = detect_regions(img_bgr, model_path=args.model)
    total_boxes = sum(len(v) for v in regions.values())
    print(f"      Phát hiện {total_boxes} vùng:")
    for cls, boxes in regions.items():
        print(f"        • {cls}: {len(boxes)} box")

    # ── 3. Vẽ bounding boxes → lưu ảnh
    vis = draw_regions(img_bgr, regions)
    cv2.imwrite(args.out, vis)
    print(f"\n      ✓ Ảnh kết quả lưu tại: {args.out}")

    # ── 4. OCR + LLM extract
    print("\n[3/4] OCR từng vùng (EasyOCR)...")
    from src.ocr_regions import ocr_image_regions
    ocr_texts = ocr_image_regions(img_bgr, regions)
    for cls, text in ocr_texts.items():
        preview = text.replace("\n", " | ")[:80]
        print(f"        • {cls}: {preview}")

    if not args.no_llm:
        print("\n[4/4] GPT-4o mini trích xuất trường...")
        from src.extract_info import extract_info
        result = extract_info(ocr_texts)

        # Lưu JSON
        json_out = str(Path(args.out).with_suffix(".json"))
        Path(json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("\n" + "="*55)
        print("  KẾT QUẢ TRÍCH XUẤT")
        print("="*55)
        for k, v in result.items():
            status = "✓" if v else "✗"
            print(f"  {status} {k:<25} {v or '(không tìm thấy)'}")
        filled = sum(1 for v in result.values() if v)
        total = len(result)
        print("="*55)
        print(f"  Trích xuất được {filled}/{total} trường")
        print(f"\n  JSON lưu tại : {json_out}")
        print(f"  Ảnh lưu tại  : {args.out}")
    else:
        print("\n[4/4] Bỏ qua GPT (--no-llm)")
        print(f"\n  Ảnh lưu tại  : {args.out}")

    print(f"\n✓ Xong!\n")


if __name__ == "__main__":
    main()
