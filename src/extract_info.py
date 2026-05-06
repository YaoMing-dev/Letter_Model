# src/extract_info.py
import json
from typing import Dict, Optional

import openai

_SYSTEM_PROMPT = """
Bạn nhận dữ liệu OCR thô từ phiếu gửi hàng Việt Nam (Viettel Post / EMS / DHL).
Dữ liệu được chia theo vùng đã detect. Nhiệm vụ:
1. Làm sạch lỗi OCR (sai dấu, thiếu ký tự, ghép từ sai)
2. Tách đúng từng trường thông tin
3. Trả về JSON thuần (không giải thích) với đúng các trường:
   don_vi_van_chuyen, ma_van_don, nguoi_gui, sdt_gui, dia_chi_gui,
   nguoi_nhan, sdt_nhan, dia_chi_nhan, tinh_tp_nhan, quan_huyen_nhan,
   phuong_xa_nhan, noi_dung_hang_hoa, trong_luong, tien_thu_ho,
   ngay_gio_gui, loai_dich_vu
Nếu không tìm thấy trường nào → null
""".strip()

_OUTPUT_FIELDS = [
    "don_vi_van_chuyen", "ma_van_don", "nguoi_gui", "sdt_gui",
    "dia_chi_gui", "nguoi_nhan", "sdt_nhan", "dia_chi_nhan",
    "tinh_tp_nhan", "quan_huyen_nhan", "phuong_xa_nhan",
    "noi_dung_hang_hoa", "trong_luong", "tien_thu_ho",
    "ngay_gio_gui", "loai_dich_vu",
]


def extract_info(regions: Dict[str, str]) -> Dict[str, Optional[str]]:
    ocr_text = "\n\n".join(
        f"[{k}]\n{v}"
        for k, v in regions.items()
        if v and v.strip()
    )

    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": ocr_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = json.loads(response.choices[0].message.content)
    result = {field: raw.get(field) for field in _OUTPUT_FIELDS}
    result["trang_thai"] = "Pending"
    return result
