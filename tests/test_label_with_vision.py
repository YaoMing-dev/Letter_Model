# tests/test_label_with_vision.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_jpg(path):
    Image.new("RGB", (64, 64), color=(200, 150, 100)).save(str(path), format="JPEG")

REQUIRED_FIELDS = [
    "don_vi_van_chuyen", "ma_van_don", "nguoi_gui", "sdt_gui",
    "dia_chi_gui", "nguoi_nhan", "sdt_nhan", "dia_chi_nhan",
    "tinh_tp_nhan", "quan_huyen_nhan", "phuong_xa_nhan",
    "noi_dung_hang_hoa", "trong_luong", "tien_thu_ho",
    "ngay_gio_gui", "loai_dich_vu",
]

SAMPLE_RESULT = {f: "test_value" for f in REQUIRED_FIELDS}


def test_label_image_returns_all_required_fields(tmp_path):
    img_path = tmp_path / "test.jpg"
    _make_jpg(img_path)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(SAMPLE_RESULT)

    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = mock_response
        from src.label_with_vision import label_image
        result = label_image(str(img_path))

    for field in REQUIRED_FIELDS:
        assert field in result, f"Missing field: {field}"


def test_label_image_model_is_gpt4o(tmp_path):
    img_path = tmp_path / "test.jpg"
    _make_jpg(img_path)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(SAMPLE_RESULT)
    mock_create = MagicMock(return_value=mock_response)

    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create = mock_create
        from src.label_with_vision import label_image
        label_image(str(img_path))

    call_kwargs = mock_create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["temperature"] == 0


def test_batch_label_creates_json_files(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    out_dir = tmp_path / "labels"

    for i in range(2):
        _make_jpg(img_dir / f"img_{i}.jpg")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(SAMPLE_RESULT)

    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = mock_response
        from src.label_with_vision import batch_label
        batch_label(str(img_dir), str(out_dir), delay=0.0)

    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) == 2


def test_batch_label_skips_existing(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    out_dir = tmp_path / "labels"
    out_dir.mkdir()

    _make_jpg(img_dir / "img_0.jpg")
    (out_dir / "img_0.json").write_text(json.dumps(SAMPLE_RESULT), encoding="utf-8")

    mock_create = MagicMock()
    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create = mock_create
        from src.label_with_vision import batch_label
        batch_label(str(img_dir), str(out_dir), delay=0.0)

    mock_create.assert_not_called()
