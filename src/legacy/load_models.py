"""
load_models.py — Helper để web backend load model đã train
-----------------------------------------------------------
Import module này trong Flask / FastAPI rồi dùng luôn.

Ví dụ:
    from src.load_models import load_vietocr, load_phobert_ner
    ocr = load_vietocr()
    ner = load_phobert_ner()
    text = ocr.predict(image)
    entities = ner.predict(text)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


# ── VietOCR ──────────────────────────────────────────────────────────────────

class VietOCRPredictor:
    def __init__(self, config_path: str, weights_path: str, device: str = "cpu"):
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        cfg = Cfg.load_config_from_file(config_path)
        cfg["weights"] = weights_path
        cfg["device"] = device
        cfg["cnn"]["pretrained"] = False
        self._predictor = Predictor(cfg)

    def predict(self, image) -> str:
        """image: PIL.Image hoặc numpy array (BGR)"""
        from PIL import Image
        import numpy as np
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[..., ::-1])  # BGR → RGB
        return self._predictor.predict(image)

    def predict_batch(self, images: list) -> list[str]:
        return [self.predict(img) for img in images]


def load_vietocr(
    config_path: str | None = None,
    weights_path: str | None = None,
    device: str = "cpu",
) -> VietOCRPredictor:
    config_path = config_path or str(MODEL_DIR / "vietocr_config.yml")
    weights_path = weights_path or str(MODEL_DIR / "vietocr_finetuned.pth")
    print(f"[VietOCR] Loading from {weights_path}")
    return VietOCRPredictor(config_path, weights_path, device)


# ── PhoBERT NER ──────────────────────────────────────────────────────────────

class PhoBERTNERPredictor:
    def __init__(
        self,
        model_dir: str,
        label_map_path: str,
        pt_path: str | None = None,
        device: str = "cpu",
    ):
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        with open(label_map_path, encoding="utf-8") as f:
            lm = json.load(f)
        self.label2id: dict = lm["label2id"]
        self.id2label: dict = {int(k): v for k, v in lm["id2label"].items()}
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if pt_path and Path(pt_path).exists():
            # Load .pt state_dict — nhẹ hơn, không cần kết nối HuggingFace Hub
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_dir,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
            )
            ckpt = torch.load(pt_path, map_location=device, weights_only=True)
            self.model.load_state_dict(ckpt["state_dict"])
            print(f"[NER] Loaded .pt weights from {pt_path}")
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            print(f"[NER] Loaded HuggingFace model from {model_dir}")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, max_length: int = 256) -> list[dict]:
        """
        Trả về list of dicts: [{"token": str, "label": str, "score": float}, ...]
        """
        words = text.split()
        if not words:
            return []

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits  # (1, seq_len, num_labels)

        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (seq_len, num_labels)
        predictions = probs.argmax(dim=-1).tolist()
        scores = probs.max(dim=-1).values.tolist()

        word_ids = enc.word_ids(batch_index=0)
        results: list[dict] = []
        seen_word_ids: set = set()

        for i, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            results.append({
                "token": words[word_id],
                "label": self.id2label[predictions[i]],
                "score": round(scores[i], 4),
            })

        return results

    def predict_entities(self, text: str) -> dict[str, str]:
        """
        Gộp các token cùng entity thành chuỗi, trả về dict field → value.
        Dùng trực tiếp làm đầu vào cho postprocess.py
        """
        tokens = self.predict(text)
        field_map = {
            "SENDER": "nguoi_gui",
            "SENDER_DEPT": "phong_gui",
            "RECEIVER": "nguoi_nhan",
            "RECEIVER_DEPT": "phong_nhan",
            "ADDRESS": "dia_chi",
            "TIME": "thoi_gian_nhan",
            "CONTENT": "noi_dung",
            "STATUS": "trang_thai",
        }
        result = {v: "" for v in field_map.values()}
        buf: list[str] = []
        cur_entity: str | None = None

        def flush():
            if cur_entity and buf:
                field = field_map.get(cur_entity, "")
                if field:
                    existing = result[field]
                    chunk = " ".join(buf)
                    result[field] = (existing + " " + chunk).strip() if existing else chunk

        for item in tokens:
            label = item["label"]
            token = item["token"]
            if label == "O":
                flush()
                buf, cur_entity = [], None
            elif label.startswith("B-"):
                flush()
                cur_entity = label[2:]
                buf = [token]
            elif label.startswith("I-"):
                entity = label[2:]
                if entity == cur_entity:
                    buf.append(token)
                else:
                    flush()
                    cur_entity = entity
                    buf = [token]
        flush()

        # Fallback: field trống → "N/A"
        for k, v in result.items():
            if not v:
                result[k] = "N/A"

        return result


def load_phobert_ner(
    model_dir: str | None = None,
    label_map_path: str | None = None,
    pt_path: str | None = None,
    device: str = "cpu",
) -> PhoBERTNERPredictor:
    model_dir = model_dir or str(MODEL_DIR / "phobert_ner")
    label_map_path = label_map_path or str(MODEL_DIR / "label_map.json")
    pt_path = pt_path or str(MODEL_DIR / "phobert_ner.pt")
    print(f"[NER] Loading from {model_dir}")
    return PhoBERTNERPredictor(model_dir, label_map_path, pt_path, device)
