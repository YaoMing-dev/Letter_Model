"""
PhoBERT NER Fine-tuning Script
-------------------------------
- Train lần đầu hoặc retrain khi có thêm data: chạy lại script này là xong.
- Output:
    models/phobert_ner/          → full HuggingFace format (tokenizer + config + safetensors)
    models/phobert_ner.pt        → state_dict thuần PyTorch để web backend load nhẹ hơn
    models/label_map.json        → mapping id→label cần kèm theo khi load .pt

Data format (CoNLL):
  data/ner_data/train.conll  — mỗi token một dòng: <token>\t<tag>, câu cách nhau bởi dòng trống
  data/ner_data/val.conll

Chạy:
  python src/train_phobert_ner.py
  python src/train_phobert_ner.py --epochs 15 --batch_size 8
"""

import argparse
import json
import os
from pathlib import Path

import torch

# ── Cấu hình mặc định ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "ner_data"
MODEL_DIR = ROOT / "models"
NER_MODEL_DIR = MODEL_DIR / "phobert_ner"
NER_PT_OUT = MODEL_DIR / "phobert_ner.pt"
LABEL_MAP_OUT = MODEL_DIR / "label_map.json"

# Toàn bộ NER tags cho bài toán phiếu giao nhận
ALL_LABELS = [
    "O",
    "B-SENDER", "I-SENDER",
    "B-SENDER_DEPT", "I-SENDER_DEPT",
    "B-RECEIVER", "I-RECEIVER",
    "B-RECEIVER_DEPT", "I-RECEIVER_DEPT",
    "B-ADDRESS", "I-ADDRESS",
    "B-TIME", "I-TIME",
    "B-CONTENT", "I-CONTENT",
    "B-STATUS", "I-STATUS",
]


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune PhoBERT NER cho phiếu giao nhận")
    p.add_argument("--base_model", default="vinai/phobert-base",
                   help="HuggingFace model ID (default: vinai/phobert-base)")
    p.add_argument("--train_file", default=str(DATA_DIR / "train.conll"))
    p.add_argument("--val_file", default=str(DATA_DIR / "val.conll"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--resume", default=None,
                   help="Path thư mục checkpoint để resume (HuggingFace format)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(),
                   help="Dùng mixed precision (chỉ khi có GPU)")
    return p.parse_args()


# ── Đọc CoNLL format ─────────────────────────────────────────────────────────

def read_conll(file_path: str) -> tuple[list[list[str]], list[list[str]]]:
    """Đọc file CoNLL, trả về (list_of_token_lists, list_of_label_lists)."""
    sentences, labels = [], []
    cur_tokens, cur_labels = [], []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if cur_tokens:
                    sentences.append(cur_tokens)
                    labels.append(cur_labels)
                    cur_tokens, cur_labels = [], []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    cur_tokens.append(parts[0])
                    cur_labels.append(parts[1])
                else:
                    # Bỏ qua dòng lỗi format
                    print(f"[Warning] Bỏ qua dòng không đúng format: {repr(line)}")
        if cur_tokens:
            sentences.append(cur_tokens)
            labels.append(cur_labels)

    print(f"[Data] {file_path}: {len(sentences)} câu")
    return sentences, labels


def check_labels(all_label_lists: list[list[str]], label2id: dict):
    """Kiểm tra không có label lạ ngoài danh sách."""
    unknown = set()
    for labels in all_label_lists:
        for lbl in labels:
            if lbl not in label2id:
                unknown.add(lbl)
    if unknown:
        raise ValueError(
            f"Có {len(unknown)} label không có trong ALL_LABELS: {unknown}\n"
            "Thêm vào ALL_LABELS trong script hoặc kiểm tra lại data."
        )


# ── Dataset ───────────────────────────────────────────────────────────────────

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, token_lists, label_lists, tokenizer, label2id, max_length):
        self.encodings = []
        skipped = 0
        for tokens, labels in zip(token_lists, label_lists):
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            word_ids = enc.word_ids(batch_index=0)
            label_ids = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)  # special tokens → ignore trong loss
                elif word_id != prev_word_id:
                    label_ids.append(label2id[labels[word_id]])
                else:
                    # Sub-token của cùng word → dùng I- tag hoặc -100
                    lbl = labels[word_id]
                    # Nếu là B- tag, sub-token sau dùng I- tương ứng
                    if lbl.startswith("B-"):
                        i_tag = "I-" + lbl[2:]
                        label_ids.append(label2id.get(i_tag, label2id[lbl]))
                    else:
                        label_ids.append(label2id[lbl])
                prev_word_id = word_id

            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(label_ids, dtype=torch.long),
            })

        if skipped:
            print(f"[Warning] Bỏ qua {skipped} samples lỗi")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


# ── Compute metrics ───────────────────────────────────────────────────────────

def make_compute_metrics(id2label):
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
        use_seqeval = True
    except ImportError:
        print("[Warning] seqeval chưa cài — chỉ dùng accuracy. pip install seqeval")
        use_seqeval = False

    import numpy as np

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            true_label_seq, true_pred_seq = [], []
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    true_label_seq.append(id2label[label])
                    true_pred_seq.append(id2label[pred])
            true_labels.append(true_label_seq)
            true_preds.append(true_pred_seq)

        if use_seqeval:
            return {
                "f1": f1_score(true_labels, true_preds),
                "precision": precision_score(true_labels, true_preds),
                "recall": recall_score(true_labels, true_preds),
            }
        # fallback: token-level accuracy
        correct = sum(
            p == l
            for seq_p, seq_l in zip(true_preds, true_labels)
            for p, l in zip(seq_p, seq_l)
        )
        total = sum(len(s) for s in true_labels)
        return {"accuracy": correct / total if total else 0}

    return compute_metrics


# ── Export .pt ────────────────────────────────────────────────────────────────

def export_pt(model, label2id: dict, id2label: dict, out_path: Path):
    """
    Xuất model state_dict + metadata ra file .pt duy nhất.
    Web backend load bằng:
        ckpt = torch.load('phobert_ner.pt', map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
    """
    torch.save({
        "state_dict": model.state_dict(),
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": len(label2id),
        "base_model": "vinai/phobert-base",
    }, out_path)
    print(f"[Export] .pt saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
    except ImportError:
        raise ImportError("Chưa cài transformers: pip install transformers")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    NER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    label2id = {lbl: i for i, lbl in enumerate(ALL_LABELS)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    # Lưu label map ra file riêng (web cần để decode output)
    with open(LABEL_MAP_OUT, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f,
                  ensure_ascii=False, indent=2)

    # ── Load data ──
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(
            f"Không tìm thấy: {args.train_file}\n"
            "Tạo file CoNLL: mỗi dòng <token>\\t<tag>, câu cách nhau bởi dòng trống"
        )

    train_tokens, train_labels = read_conll(args.train_file)
    val_tokens, val_labels = read_conll(args.val_file)

    check_labels(train_labels + val_labels, label2id)

    # ── Tokenizer + Model ──
    model_source = args.resume if args.resume else args.base_model
    print(f"\n[Model] Load từ: {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)  # tokenizer luôn load từ base
    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=len(ALL_LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # khi resume từ checkpoint có num_labels khác
    )

    # ── Dataset ──
    train_dataset = NERDataset(train_tokens, train_labels, tokenizer, label2id, args.max_length)
    val_dataset = NERDataset(val_tokens, val_labels, tokenizer, label2id, args.max_length)

    print(f"[Config] Epochs      : {args.epochs}")
    print(f"[Config] Batch size  : {args.batch_size}")
    print(f"[Config] LR          : {args.lr}")
    print(f"[Config] Max length  : {args.max_length}")
    print(f"[Config] Device      : {args.device}")
    print(f"[Config] FP16        : {args.fp16}")
    print(f"[Config] Output      : {NER_MODEL_DIR}\n")

    # ── Training args ──
    training_args = TrainingArguments(
        output_dir=str(NER_MODEL_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        logging_dir=str(MODEL_DIR / "logs"),
        logging_steps=50,
        report_to="none",           # không gửi lên wandb/tensorboard mặc định
        save_total_limit=2,         # chỉ giữ 2 checkpoint gần nhất
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # ── Lưu HuggingFace format (tokenizer + config + weights) ──
    trainer.save_model(str(NER_MODEL_DIR))
    tokenizer.save_pretrained(str(NER_MODEL_DIR))
    print(f"[Done] HuggingFace model saved: {NER_MODEL_DIR}")

    # ── Xuất .pt thuần PyTorch cho web ──
    export_pt(trainer.model, label2id, id2label, NER_PT_OUT)

    print(f"\n[Done] Label map saved: {LABEL_MAP_OUT}")
    print("\n── Cách load trong web backend ──────────────────────────────────────")
    print("  from transformers import AutoTokenizer, AutoModelForTokenClassification")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{NER_MODEL_DIR}')")
    print(f"  model = AutoModelForTokenClassification.from_pretrained('{NER_MODEL_DIR}')")
    print("")
    print("  # Hoặc load .pt trực tiếp:")
    print(f"  ckpt = torch.load('{NER_PT_OUT}', map_location='cpu', weights_only=True)")
    print("  model.load_state_dict(ckpt['state_dict'])")
    print("────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    args = parse_args()
    train(args)
