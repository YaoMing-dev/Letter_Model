"""
label_existing_crops.py
-----------------------
Chay VietOCR tren cac crop da co trong data/ocr_crops/ va ghi label file.
Dung khi crop da duoc don sach thu cong, chi can label lai.

Chay:
  python src/label_existing_crops.py
  python src/label_existing_crops.py --no_ocr   # ghi path, de label sau
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_ocr(img_paths: list[Path]) -> dict[str, str]:
    import torch
    from PIL import Image
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"VietOCR device: {device}")
    cfg = Cfg.load_config_from_name("vgg_transformer")
    cfg["device"] = device
    cfg["cnn"]["pretrained"] = False  # VietOCR .pth da co weights, khong load them ImageNet
    predictor = Predictor(cfg)

    results = {}
    total = len(img_paths)
    for i, p in enumerate(img_paths, 1):
        try:
            img = Image.open(p).convert("RGB")
            results[str(p)] = predictor.predict(img)
        except Exception as e:
            results[str(p)] = ""
            print(f"  [err] {p.name}: {e}")
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crop_dir",  default="data/ocr_crops")
    p.add_argument("--label_dir", default="data/ocr_labels")
    p.add_argument("--no_ocr", action="store_true",
                   help="Chi ghi path, de label trong de tu dien sau")
    args = p.parse_args()

    crop_dir  = Path(args.crop_dir)
    label_dir = Path(args.label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        crops = sorted((crop_dir / split).glob("*.png"))
        if not crops:
            print(f"[{split}] Khong co crop nao trong {crop_dir / split}")
            continue

        print(f"\n[{split}] {len(crops)} crops...")
        ocr = run_ocr(crops) if not args.no_ocr else {str(p): "" for p in crops}

        label_file = label_dir / f"{split}.txt"
        written = 0
        with open(label_file, "w", encoding="utf-8") as f:
            for cp in crops:
                label = ocr.get(str(cp), "").strip()
                if not label and not args.no_ocr:
                    continue          # bo qua dong khong co label khi chay OCR that
                rel = cp.resolve().relative_to(ROOT.resolve())
                f.write(f"{rel}\t{label}\n")
                written += 1

        print(f"  -> {label_file}: {written} dong")

    print("\nXong. Kiem tra label roi chay:")
    print("  python src/train_vietocr.py --iters 10000")


if __name__ == "__main__":
    main()
