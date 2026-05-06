# src/export.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from src.auth_google import get_gspread_client

_SHEET_COLUMNS: List[str] = [
    "ma_van_don", "don_vi_van_chuyen", "nguoi_gui", "sdt_gui",
    "nguoi_nhan", "sdt_nhan", "dia_chi_nhan",
    "noi_dung_hang_hoa", "tien_thu_ho", "ngay_gio_gui",
    "trang_thai", "ngay_nhan",
]


def push_to_sheet(
    data: Dict[str, Optional[str]],
    sheet_id: str,
    credentials_file: str = "credentials.json",
    token_file: str = "token.json",
) -> None:
    gc = get_gspread_client(credentials_file=credentials_file, token_file=token_file)
    ws = gc.open_by_key(sheet_id).sheet1
    row = [data.get(col) or "" for col in _SHEET_COLUMNS[:-1]]
    row.append("")  # ngay_nhan — điền sau khi nhấn Received
    ws.append_row(row)


def _build_html(data: Dict[str, Optional[str]], confirm_url: str) -> str:
    def v(key: str) -> str:
        return data.get(key) or ""

    return f"""<!DOCTYPE html>
<html><body style="font-family:sans-serif;max-width:600px;margin:auto">
<h3>Bưu phẩm mới: {v('ma_van_don')}</h3>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%">
  <tr><td><b>Đơn vị vận chuyển</b></td><td>{v('don_vi_van_chuyen')}</td></tr>
  <tr><td><b>Người gửi</b></td><td>{v('nguoi_gui')} &mdash; {v('sdt_gui')}</td></tr>
  <tr><td><b>Người nhận</b></td><td>{v('nguoi_nhan')} &mdash; {v('sdt_nhan')}</td></tr>
  <tr><td><b>Địa chỉ giao</b></td><td>{v('dia_chi_nhan')}</td></tr>
  <tr><td><b>Nội dung hàng</b></td><td>{v('noi_dung_hang_hoa')}</td></tr>
  <tr><td><b>Tiền thu hộ</b></td><td>{v('tien_thu_ho')} VNĐ</td></tr>
  <tr><td><b>Ngày gửi</b></td><td>{v('ngay_gio_gui')}</td></tr>
</table>
<br>
<a href="{confirm_url}"
   style="background:#4CAF50;color:white;padding:14px 28px;text-decoration:none;
          border-radius:6px;display:inline-block;font-size:16px;">
  ✓ Xác nhận đã nhận bưu phẩm
</a>
</body></html>"""


def send_notification_email(
    data: Dict[str, Optional[str]],
    confirm_base_url: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    notify_email: str,
) -> None:
    ma_van_don = data.get("ma_van_don") or ""
    confirm_url = f"{confirm_base_url.rstrip('/')}/api/confirm-received?id={ma_van_don}"

    # Build HTML part with 8bit CTE so ASCII content (tracking ID, URL)
    # stays literal in msg.as_string() instead of being base64-encoded.
    html_part = MIMEText(_build_html(data, confirm_url), "html", "utf-8")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Bưu phẩm] Mã vận đơn {ma_van_don}"
    msg["From"] = smtp_user
    msg["To"] = notify_email
    msg.attach(html_part)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
