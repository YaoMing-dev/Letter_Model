# src/ocr_regions.py
from typing import Dict, List, Tuple

import numpy as np

Box = Tuple[int, int, int, int, float]

_readers: Dict[str, object] = {}


def _get_reader(lang: str = "vi"):
    """Lazy-load EasyOCR reader, cached per language."""
    if lang not in _readers:
        import easyocr
        _readers[lang] = easyocr.Reader([lang, "en"], gpu=False, verbose=False)
    return _readers[lang]


def ocr_region(
    img_crop: np.ndarray,
    lang: str = "vi",
    conf_threshold: float = 0.5,
) -> str:
    if img_crop.size == 0:
        return ""
    reader = _get_reader(lang)
    result = reader.readtext(img_crop)
    lines = [text for (_, text, conf) in result if conf >= conf_threshold]
    return "\n".join(lines)


def ocr_image_regions(
    image: np.ndarray,
    regions: Dict[str, List[Box]],
    lang: str = "vi",
    conf_threshold: float = 0.5,
) -> Dict[str, str]:
    """OCR each detected region, choosing the highest-confidence bounding box."""
    texts: Dict[str, str] = {}
    for cls_name, boxes in regions.items():
        if not boxes:
            texts[cls_name] = ""
            continue
        x1, y1, x2, y2, _ = max(boxes, key=lambda b: b[4])
        crop = image[y1:y2, x1:x2]
        texts[cls_name] = ocr_region(crop, lang=lang, conf_threshold=conf_threshold)
    return texts
