# src/auth_google.py
"""
Handles Google OAuth authentication, supporting both service_account and
OAuth client credentials (installed or web app type).
"""
import json
import os
from pathlib import Path

import gspread
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"


def _load_raw(credentials_file: str) -> dict:
    with open(credentials_file, encoding="utf-8") as f:
        return json.load(f)


def get_gspread_client(
    credentials_file: str = "credentials.json",
    token_file: str = "token.json",
) -> gspread.Client:
    raw = _load_raw(credentials_file)

    # Service account — dùng trực tiếp
    if raw.get("type") == "service_account":
        creds = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=SCOPES
        )
        return gspread.authorize(creds)

    # OAuth client (web hoặc installed)
    client_info = raw.get("installed") or raw.get("web")
    if not client_info:
        raise ValueError(f"Unknown credential format in {credentials_file}")

    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Lần đầu: cần chạy setup_auth.py để authorize
            raise RuntimeError(
                f"Chưa authorize Google Sheets. Hãy chạy:\n"
                f"  .venv312\\Scripts\\python setup_auth.py\n"
                f"để mở trình duyệt và cấp quyền."
            )
        # Lưu lại token đã refresh
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return gspread.authorize(creds)
