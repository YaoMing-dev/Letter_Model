# tests/test_extract_info.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_FIELDS = [
    "don_vi_van_chuyen", "ma_van_don", "nguoi_gui", "sdt_gui",
    "dia_chi_gui", "nguoi_nhan", "sdt_nhan", "dia_chi_nhan",
    "tinh_tp_nhan", "quan_huyen_nhan", "phuong_xa_nhan",
    "noi_dung_hang_hoa", "trong_luong", "tien_thu_ho",
    "ngay_gio_gui", "loai_dich_vu", "trang_thai",
]

SAMPLE_OCR = {
    "sender_block": "CÔNG TY ABC\n0901234567\nHà Nội",
    "receiver_block": "Nguyễn Văn A\n0987654321\nTP.HCM",
    "tracking_number": "138963567225",
}

SAMPLE_LLM_RESPONSE = {
    "don_vi_van_chuyen": "Viettel Post",
    "ma_van_don": "138963567225",
    "nguoi_gui": "CÔNG TY ABC",
    "sdt_gui": "0901234567",
    "dia_chi_gui": "Hà Nội",
    "nguoi_nhan": "Nguyễn Văn A",
    "sdt_nhan": "0987654321",
    "dia_chi_nhan": "TP.HCM",
    "tinh_tp_nhan": "Hồ Chí Minh",
    "quan_huyen_nhan": None,
    "phuong_xa_nhan": None,
    "noi_dung_hang_hoa": None,
    "trong_luong": None,
    "tien_thu_ho": None,
    "ngay_gio_gui": None,
    "loai_dich_vu": None,
}


def _mock_openai(response_dict: dict):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(response_dict)
    return mock_response


def test_extract_info_returns_all_required_fields():
    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_openai(SAMPLE_LLM_RESPONSE)
        from src.extract_info import extract_info
        result = extract_info(SAMPLE_OCR)

    for field in OUTPUT_FIELDS:
        assert field in result, f"Missing field: {field}"


def test_extract_info_always_injects_pending_status():
    response_with_wrong_status = dict(SAMPLE_LLM_RESPONSE)
    response_with_wrong_status["trang_thai"] = "Received"

    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = _mock_openai(response_with_wrong_status)
        from src.extract_info import extract_info
        result = extract_info(SAMPLE_OCR)

    assert result["trang_thai"] == "Pending"


def test_extract_info_uses_gpt4o_mini():
    mock_create = MagicMock(return_value=_mock_openai(SAMPLE_LLM_RESPONSE))
    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create = mock_create
        from src.extract_info import extract_info
        extract_info(SAMPLE_OCR)

    call_kwargs = mock_create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0


def test_extract_info_skips_empty_regions():
    ocr_with_empties = {
        "sender_block": "text",
        "receiver_block": "",
        "tracking_number": "   ",
    }
    mock_create = MagicMock(return_value=_mock_openai(SAMPLE_LLM_RESPONSE))
    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create = mock_create
        from src.extract_info import extract_info
        extract_info(ocr_with_empties)

    prompt_text = mock_create.call_args[1]["messages"][1]["content"]
    assert "receiver_block" not in prompt_text
    assert "tracking_number" not in prompt_text
    assert "sender_block" in prompt_text
