# tests/test_auto_bbox.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

YOLO_CLASSES = [
    "sender_block", "receiver_block", "tracking_number",
    "address_block", "content_block", "fee_block", "datetime_block",
]


def _make_mock_gdino():
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_processor.return_value = MagicMock()
    mock_processor.post_process_grounded_object_detection.return_value = [{
        "boxes": torch.tensor([[50.0, 50.0, 200.0, 150.0]]),
        "labels": ["detected"],
        "scores": torch.tensor([0.8]),
    }]
    return mock_processor, mock_model


def test_label_image_returns_dict_with_valid_classes(tmp_path):
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (600, 800), color=(200, 200, 200)).save(str(img_path))
    mock_processor, mock_model = _make_mock_gdino()

    from src.auto_bbox import label_image
    result = label_image(str(img_path), mock_processor, mock_model, "cpu")

    for cls_name in result:
        assert cls_name in YOLO_CLASSES
    for boxes in result.values():
        for cx, cy, bw, bh in boxes:
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < bw <= 1.0
            assert 0.0 < bh <= 1.0


def test_write_yolo_label_creates_valid_format(tmp_path):
    from src.auto_bbox import _write_yolo_label

    detections = {
        "sender_block": [(0.25, 0.10, 0.40, 0.15)],
        "receiver_block": [(0.25, 0.40, 0.40, 0.20)],
    }
    out_file = tmp_path / "test.txt"
    _write_yolo_label(out_file, detections)

    lines = out_file.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        parts = line.split()
        assert len(parts) == 5, f"Expected 5 values per line, got: {line}"
        cls_id = int(parts[0])
        assert 0 <= cls_id <= 6
        for coord in parts[1:]:
            assert 0.0 <= float(coord) <= 1.0


def test_write_yolo_label_correct_class_id(tmp_path):
    from src.auto_bbox import _write_yolo_label, CLASS_ID

    detections = {"tracking_number": [(0.5, 0.1, 0.3, 0.05)]}
    out_file = tmp_path / "test.txt"
    _write_yolo_label(out_file, detections)

    cls_id = int(out_file.read_text().split()[0])
    assert cls_id == CLASS_ID["tracking_number"]


def test_batch_creates_train_val_split(tmp_path):
    from PIL import Image
    img_dir = tmp_path / "raw"
    img_dir.mkdir()
    out_dir = tmp_path / "dataset"
    for i in range(5):
        Image.new("RGB", (300, 400)).save(str(img_dir / f"img_{i}.jpg"))

    mock_processor, mock_model = _make_mock_gdino()
    with patch("src.auto_bbox._load_model", return_value=(mock_processor, mock_model)):
        from src.auto_bbox import batch_label_and_split
        batch_label_and_split(str(img_dir), str(out_dir), val_ratio=0.2)

    assert (out_dir / "images" / "train").exists()
    assert (out_dir / "images" / "val").exists()
    assert (out_dir / "labels" / "train").exists()
    assert (out_dir / "labels" / "val").exists()
    total_labels = (
        len(list((out_dir / "labels" / "train").glob("*.txt")))
        + len(list((out_dir / "labels" / "val").glob("*.txt")))
    )
    assert total_labels == 5


def test_batch_skips_existing_labels(tmp_path):
    from PIL import Image
    img_dir = tmp_path / "raw"
    img_dir.mkdir()
    out_dir = tmp_path / "dataset"
    (out_dir / "images" / "train").mkdir(parents=True)
    (out_dir / "labels" / "train").mkdir(parents=True)
    (out_dir / "images" / "val").mkdir(parents=True)
    (out_dir / "labels" / "val").mkdir(parents=True)

    img = img_dir / "img_0.jpg"
    Image.new("RGB", (300, 400)).save(str(img))
    (out_dir / "labels" / "train" / "img_0.txt").write_text("0 0.5 0.1 0.3 0.05")

    mock_processor, mock_model = _make_mock_gdino()
    mock_model_call = MagicMock()
    mock_model.return_value = mock_model_call

    with patch("src.auto_bbox._load_model", return_value=(mock_processor, mock_model)):
        from src.auto_bbox import batch_label_and_split
        batch_label_and_split(str(img_dir), str(out_dir), val_ratio=0.2)

    mock_model.assert_not_called()
