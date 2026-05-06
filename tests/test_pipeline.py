# tests/test_pipeline.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_CONFIG = {
    "yolo_model": "models/best.pt",
    "sheet_id": "fake_sheet",
    "confirm_base_url": "https://example.com",
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "u@gmail.com",
    "smtp_password": "pass",
    "notify_email": "n@gmail.com",
}

_SAMPLE_EXTRACTED = {
    "ma_van_don": "123456",
    "don_vi_van_chuyen": "Viettel Post",
    "nguoi_gui": "ABC Corp",
    "sdt_gui": "0901234567",
    "nguoi_nhan": "Nguyễn Văn A",
    "sdt_nhan": "0987654321",
    "dia_chi_nhan": "TP.HCM",
    "noi_dung_hang_hoa": "1 x giấy tờ",
    "tien_thu_ho": "55000",
    "ngay_gio_gui": "01/01/2026",
    "trang_thai": "Pending",
    "dia_chi_gui": None,
    "tinh_tp_nhan": None,
    "quan_huyen_nhan": None,
    "phuong_xa_nhan": None,
    "trong_luong": None,
    "loai_dich_vu": None,
}


def test_process_image_calls_all_five_stages(tmp_path):
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

    fake_img = np.zeros((800, 600, 3), dtype=np.uint8)

    with patch("src.pipeline.load_and_prepare", return_value=fake_img) as mock_load, \
         patch("src.pipeline.detect_regions", return_value={"sender_block": [(10, 10, 200, 100, 0.9)]}) as mock_det, \
         patch("src.pipeline.ocr_image_regions", return_value={"sender_block": "ABC Corp"}) as mock_ocr, \
         patch("src.pipeline.extract_info", return_value=_SAMPLE_EXTRACTED) as mock_ext, \
         patch("src.pipeline.push_to_sheet") as mock_sheet, \
         patch("src.pipeline.send_notification_email") as mock_email:

        from src.pipeline import process_image
        result = process_image(str(img_path), _CONFIG)

    mock_load.assert_called_once()
    mock_det.assert_called_once()
    mock_ocr.assert_called_once()
    mock_ext.assert_called_once()
    mock_sheet.assert_called_once()
    mock_email.assert_called_once()
    assert result["ma_van_don"] == "123456"


def test_process_image_passes_config_to_export(tmp_path):
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    fake_img = np.zeros((800, 600, 3), dtype=np.uint8)

    with patch("src.pipeline.load_and_prepare", return_value=fake_img), \
         patch("src.pipeline.detect_regions", return_value={}), \
         patch("src.pipeline.ocr_image_regions", return_value={}), \
         patch("src.pipeline.extract_info", return_value=_SAMPLE_EXTRACTED), \
         patch("src.pipeline.push_to_sheet") as mock_sheet, \
         patch("src.pipeline.send_notification_email") as mock_email:

        from src.pipeline import process_image
        process_image(str(img_path), _CONFIG)

    mock_sheet.assert_called_once_with(_SAMPLE_EXTRACTED, sheet_id="fake_sheet")
    _, email_kwargs = mock_email.call_args
    assert email_kwargs["smtp_host"] == "smtp.gmail.com"
    assert email_kwargs["notify_email"] == "n@gmail.com"


def test_process_image_raises_if_file_not_found():
    import pytest
    with pytest.raises(FileNotFoundError):
        from src.pipeline import process_image
        process_image("nonexistent_file.jpg", _CONFIG)
