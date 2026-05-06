# tests/test_ocr_regions.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_mock_reader(lines: list):
    """lines = list of (text, conf). Returns EasyOCR-style [(bbox, text, conf), ...]"""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[0, i*25], [100, i*25], [100, (i+1)*25], [0, (i+1)*25]], text, conf)
        for i, (text, conf) in enumerate(lines)
    ]
    return mock_reader


def test_ocr_region_returns_string():
    mock_reader = _make_mock_reader([("Xin chào", 0.98), ("Thế giới", 0.85)])
    with patch("easyocr.Reader", return_value=mock_reader):
        import src.ocr_regions as m
        m._readers.clear()
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = m.ocr_region(img)

    assert isinstance(result, str)
    assert "Xin chào" in result
    assert "Thế giới" in result


def test_ocr_region_filters_low_confidence():
    mock_reader = _make_mock_reader([("high conf", 0.9), ("low conf", 0.3)])
    with patch("easyocr.Reader", return_value=mock_reader):
        import src.ocr_regions as m
        m._readers.clear()
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = m.ocr_region(img, conf_threshold=0.5)

    assert "high conf" in result
    assert "low conf" not in result


def test_ocr_region_empty_image_returns_empty_string():
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    from src.ocr_regions import ocr_region
    result = ocr_region(img)
    assert result == ""


def test_ocr_image_regions_returns_dict_per_class():
    regions = {
        "sender_block": [(10, 10, 200, 100, 0.9)],
        "receiver_block": [(10, 110, 200, 200, 0.85)],
    }
    mock_reader = _make_mock_reader([("text line", 0.9)])
    with patch("easyocr.Reader", return_value=mock_reader):
        import src.ocr_regions as m
        m._readers.clear()
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        result = m.ocr_image_regions(img, regions)

    assert "sender_block" in result
    assert "receiver_block" in result
    assert isinstance(result["sender_block"], str)


def test_ocr_image_regions_picks_highest_conf_box():
    regions = {
        "sender_block": [
            (10, 10, 100, 50, 0.6),
            (10, 60, 100, 110, 0.95),  # box này được chọn
        ]
    }
    mock_reader = _make_mock_reader([("selected", 0.9)])
    with patch("easyocr.Reader", return_value=mock_reader):
        import src.ocr_regions as m
        m._readers.clear()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = m.ocr_image_regions(img, regions)

    assert "selected" in result["sender_block"]
