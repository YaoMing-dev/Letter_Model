# tests/test_export.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_DATA = {
    "ma_van_don": "138963567225",
    "don_vi_van_chuyen": "Viettel Post",
    "nguoi_gui": "CÔNG TY ABC",
    "sdt_gui": "0901234567",
    "nguoi_nhan": "Nguyễn Văn A",
    "sdt_nhan": "0987654321",
    "dia_chi_nhan": "TP.HCM",
    "noi_dung_hang_hoa": "1 x giấy tờ",
    "tien_thu_ho": "55000",
    "ngay_gio_gui": "08/04/2026 15:00:53",
    "trang_thai": "Pending",
}


def test_push_to_sheet_appends_one_row():
    mock_ws = MagicMock()
    mock_gc = MagicMock()
    mock_gc.open_by_key.return_value.sheet1 = mock_ws

    with patch("src.export.get_gspread_client", return_value=mock_gc):
        from src.export import push_to_sheet
        push_to_sheet(SAMPLE_DATA, sheet_id="fake_sheet_id")

    mock_ws.append_row.assert_called_once()


def test_push_to_sheet_row_contains_key_fields():
    mock_ws = MagicMock()
    mock_gc = MagicMock()
    mock_gc.open_by_key.return_value.sheet1 = mock_ws

    with patch("src.export.get_gspread_client", return_value=mock_gc):
        from src.export import push_to_sheet
        push_to_sheet(SAMPLE_DATA, sheet_id="fake_sheet_id")

    row = mock_ws.append_row.call_args[0][0]
    assert "138963567225" in row
    assert "Pending" in row
    assert "" in row  # ngay_nhan phải rỗng


def test_push_to_sheet_ngay_nhan_is_empty():
    mock_ws = MagicMock()
    mock_gc = MagicMock()
    mock_gc.open_by_key.return_value.sheet1 = mock_ws

    with patch("src.export.get_gspread_client", return_value=mock_gc):
        from src.export import push_to_sheet
        push_to_sheet(SAMPLE_DATA, sheet_id="fake_sheet_id")

    row = mock_ws.append_row.call_args[0][0]
    assert row[-1] == ""  # ngay_nhan là cột cuối, phải rỗng


def test_send_notification_email_calls_smtp_starttls():
    with patch("smtplib.SMTP") as MockSMTP:
        mock_server = MagicMock()
        MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_server)
        MockSMTP.return_value.__exit__ = MagicMock(return_value=False)

        from src.export import send_notification_email
        send_notification_email(
            data=SAMPLE_DATA,
            confirm_base_url="https://example.com",
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_user="sender@gmail.com",
            smtp_password="password",
            notify_email="recipient@gmail.com",
        )

    MockSMTP.assert_called_once_with("smtp.gmail.com", 587)
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with("sender@gmail.com", "password")
    mock_server.send_message.assert_called_once()


def test_send_notification_email_contains_confirm_url():
    with patch("smtplib.SMTP") as MockSMTP:
        mock_server = MagicMock()
        MockSMTP.return_value.__enter__ = MagicMock(return_value=mock_server)
        MockSMTP.return_value.__exit__ = MagicMock(return_value=False)

        from src.export import send_notification_email
        send_notification_email(
            data=SAMPLE_DATA,
            confirm_base_url="https://example.com",
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_user="s@gmail.com",
            smtp_password="pass",
            notify_email="r@gmail.com",
        )

    msg_obj = mock_server.send_message.call_args[0][0]
    # HTML part may be base64-encoded; decode payload to get plain text
    html_part = msg_obj.get_payload(0)
    body = html_part.get_payload(decode=True).decode("utf-8")
    assert "138963567225" in body
    assert "https://example.com/api/confirm-received?id=138963567225" in body
