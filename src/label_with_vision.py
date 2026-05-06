# src/label_with_vision.py
import base64
import io
import json
import time
from pathlib import Path
from typing import Dict, Optional

import openai
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

REQUIRED_FIELDS = [
    "don_vi_van_chuyen", "ma_van_don", "nguoi_gui", "sdt_gui",
    "dia_chi_gui", "nguoi_nhan", "sdt_nhan", "dia_chi_nhan",
    "tinh_tp_nhan", "quan_huyen_nhan", "phuong_xa_nhan",
    "noi_dung_hang_hoa", "trong_luong", "tien_thu_ho",
    "ngay_gio_gui", "loai_dich_vu",
]

_PROMPT = (
    "Đọc phiếu gửi hàng Việt Nam này và trả về JSON với đúng các trường: "
    + ", ".join(REQUIRED_FIELDS)
    + ". Nếu không tìm thấy trường nào → null. Chỉ trả JSON thuần, không giải thích."
)


def _to_jpeg_b64(img_path: str, max_px: int = 2048, quality: int = 85) -> str:
    """Đọc ảnh (kể cả HEIC), resize về max_px nếu cần, trả về base64 JPEG."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def label_image(img_path: str, max_retries: int = 2) -> Dict[str, Optional[str]]:
    client = openai.OpenAI()
    # Thử với quality cao, nếu fail thử lại với ảnh nhỏ hơn
    for attempt, (max_px, quality) in enumerate([(2048, 85), (1280, 75), (960, 65)]):
        if attempt >= max_retries + 1:
            break
        b64 = _to_jpeg_b64(img_path, max_px=max_px, quality=quality)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _PROMPT},
                ],
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if content:
            return json.loads(content)
        print(f"  (retry {attempt+1}, giam chat luong anh...)", end=" ", flush=True)
    raise ValueError(f"GPT-4o khong tra ve noi dung sau {max_retries+1} lan thu")


def batch_label(img_dir: str, out_dir: str, delay: float = 1.0) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".heic"}
    images = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts)

    for img_path in images:
        out_path = Path(out_dir) / (img_path.stem + ".json")
        if out_path.exists():
            print(f"Skip (exists): {img_path.name}")
            continue

        print(f"Labeling: {img_path.name} ...", end=" ", flush=True)
        try:
            result = label_image(str(img_path))
            result["_source_image"] = img_path.name
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(delay)


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="data/raw_images")
    parser.add_argument("--out_dir", default="data/labels")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Giây chờ giữa mỗi ảnh (tránh rate limit)")
    args = parser.parse_args()

    batch_label(args.img_dir, args.out_dir, args.delay)
