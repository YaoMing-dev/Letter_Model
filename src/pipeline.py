# src/pipeline.py
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from src.preprocessing import load_and_prepare
from src.detect_regions import detect_regions
from src.ocr_regions import ocr_image_regions
from src.extract_info import extract_info
from src.export import push_to_sheet, send_notification_email


def process_image(
    img_path: str,
    config: Dict,
) -> Dict[str, Optional[str]]:
    path = Path(img_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    print(f"[1/5] Preprocessing: {path.name}")
    img_rgb = load_and_prepare(path)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    print("[2/5] Detecting regions (YOLO)...")
    regions = detect_regions(img_bgr, model_path=config["yolo_model"])

    print("[3/5] OCR per region (PaddleOCR)...")
    ocr_texts = ocr_image_regions(img_bgr, regions)

    print("[4/5] Extracting fields (GPT-4o mini)...")
    extracted = extract_info(ocr_texts)

    print("[5/5] Exporting to Google Sheets + Email...")
    push_to_sheet(extracted, sheet_id=config["sheet_id"])
    send_notification_email(
        data=extracted,
        confirm_base_url=config["confirm_base_url"],
        smtp_host=config["smtp_host"],
        smtp_port=config["smtp_port"],
        smtp_user=config["smtp_user"],
        smtp_password=config["smtp_password"],
        notify_email=config["notify_email"],
    )

    print(f"Done: {extracted.get('ma_van_don', 'unknown')}")
    return extracted


if __name__ == "__main__":
    import argparse
    import json
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Xử lý phiếu gửi hàng end-to-end")
    parser.add_argument("image", help="Đường dẫn tới ảnh phiếu")
    parser.add_argument(
        "--yolo_model",
        default="runs/detect/models/yolo_regions/phieu_gui_v1/weights/best.pt",
    )
    args = parser.parse_args()

    cfg = {
        "yolo_model": args.yolo_model,
        "sheet_id": os.environ["GOOGLE_SHEET_ID"],
        "confirm_base_url": os.environ["CONFIRM_BASE_URL"],
        "smtp_host": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": int(os.environ.get("SMTP_PORT", "587")),
        "smtp_user": os.environ["SMTP_USER"],
        "smtp_password": os.environ["SMTP_PASSWORD"],
        "notify_email": os.environ["NOTIFY_EMAIL"],
    }

    result = process_image(args.image, cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
