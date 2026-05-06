"""
VietOCR Fine-tuning Script
--------------------------
- Train lần đầu hoặc retrain khi có thêm data: chạy lại script này là xong.
- Output: models/vietocr_finetuned.pth  (load được bằng VietOCR hoặc torch.load)
- Web backend: load bằng Predictor(config) với weights trỏ tới file .pth

Data format (train/val):
  data/ocr_labels/train.txt  — mỗi dòng: <tên_ảnh>\t<text>
  data/ocr_labels/val.txt
  ảnh tương ứng nằm trong: data/ocr_crops/train/ và data/ocr_crops/val/

Chạy:
  python src/train_vietocr.py
  python src/train_vietocr.py --iters 10000 --batch_size 32
"""

import argparse
import os
import shutil
from pathlib import Path

# ── Cấu hình mặc định ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
WEIGHTS_OUT = MODEL_DIR / "vietocr_finetuned.pth"
CONFIG_OUT = MODEL_DIR / "vietocr_config.yml"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune VietOCR trên dữ liệu phiếu thu")
    p.add_argument("--base_model", default="vgg_transformer",
                   choices=["vgg_transformer", "vgg_seq2seq"],
                   help="Backbone VietOCR base (default: vgg_transformer)")
    p.add_argument("--iters", type=int, default=5000,
                   help="Số iteration train (default: 5000)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--valid_every", type=int, default=500)
    p.add_argument("--train_annotation",
                   default=str(DATA_DIR / "ocr_labels" / "train.txt"))
    p.add_argument("--val_annotation",
                   default=str(DATA_DIR / "ocr_labels" / "val.txt"))
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="Load pretrained VietOCR weights (default: True)")
    p.add_argument("--resume", default=None,
                   help="Path tới checkpoint để resume training")
    p.add_argument("--device", default="cuda",
                   choices=["cuda", "cpu"],
                   help="Device train (default: cuda)")
    return p.parse_args()


def check_data(train_ann: str, val_ann: str):
    for path in [train_ann, val_ann]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy annotation file: {path}\n"
                "Tạo file theo format: <tên_ảnh>\\t<text> mỗi dòng"
            )
    with open(train_ann, encoding="utf-8") as f:
        n_train = sum(1 for _ in f)
    with open(val_ann, encoding="utf-8") as f:
        n_val = sum(1 for _ in f)
    print(f"[Data] Train: {n_train} samples | Val: {n_val} samples")
    if n_train < 50:
        print("[Warning] Data train < 50 samples — kết quả có thể không tốt")


def train(args):
    try:
        from vietocr.tool.config import Cfg
        from vietocr.model.trainer import Trainer
    except ImportError:
        raise ImportError("Chưa cài vietocr: pip install vietocr")

    check_data(args.train_annotation, args.val_annotation)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = Cfg.load_config_from_name(args.base_model)

    config['trainer']['pretrained'] = args.pretrained
    config['trainer']['batch_size'] = args.batch_size
    config['trainer']['print_every'] = args.print_every
    config['trainer']['valid_every'] = args.valid_every
    config['trainer']['iters'] = args.iters
    config['trainer']['export'] = str(WEIGHTS_OUT)

    config['dataset']['train_annotation'] = args.train_annotation
    config['dataset']['valid_annotation'] = args.val_annotation
    config['dataset']['data_root'] = ''

    config['device'] = args.device
    config['weights'] = str(WEIGHTS_OUT)

    # LMDB Environment không pickle được → tắt multiprocessing workers
    config['dataloader']['num_workers'] = 0

    # Resume từ checkpoint nếu có
    if args.resume and os.path.exists(args.resume):
        config['weights'] = args.resume
        print(f"[Resume] Load checkpoint từ: {args.resume}")
    elif WEIGHTS_OUT.exists():
        print(f"[Resume] Tìm thấy checkpoint cũ tại {WEIGHTS_OUT}, tiếp tục train...")
        config['weights'] = str(WEIGHTS_OUT)

    print(f"\n[Config] Base model  : {args.base_model}")
    print(f"[Config] Iterations  : {args.iters}")
    print(f"[Config] Batch size  : {args.batch_size}")
    print(f"[Config] Device      : {args.device}")
    print(f"[Config] Output      : {WEIGHTS_OUT}\n")

    trainer = Trainer(config, pretrained=args.pretrained)
    trainer.train()

    # Lưu config đi kèm để inference sau không cần guess lại
    config.save(str(CONFIG_OUT))
    print(f"\n[Done] Model saved : {WEIGHTS_OUT}")
    print(f"[Done] Config saved: {CONFIG_OUT}")
    print("\nLoad để inference:")
    print("  from vietocr.tool.predictor import Predictor")
    print("  from vietocr.tool.config import Cfg")
    print(f"  cfg = Cfg.load_config_from_file('{CONFIG_OUT}')")
    print("  predictor = Predictor(cfg)")


if __name__ == "__main__":
    args = parse_args()
    train(args)
