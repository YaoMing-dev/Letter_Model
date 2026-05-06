# setup_auth.py — Chay 1 lan de authorize Google Sheets
# Usage: .venv312\Scripts\python setup_auth.py
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

load_dotenv()

CREDENTIALS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
TOKEN_FILE = os.environ.get("GOOGLE_TOKEN_FILE", "token.json")

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

# Cho phep Google tra ve subset cua scope ma khong raise loi
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"


def main():
    if not Path(CREDENTIALS_FILE).exists():
        print(f"Khong tim thay {CREDENTIALS_FILE}")
        sys.exit(1)

    with open(CREDENTIALS_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    if raw.get("type") == "service_account":
        print("Service account - khong can authorize them.")
        return

    client_info = raw.get("installed") or raw.get("web")
    if not client_info:
        print("Khong nhan ra dinh dang credentials.")
        sys.exit(1)

    app_flow_config = {"installed": client_info}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(app_flow_config, tmp)
        tmp_path = tmp.name

    print("=" * 55)
    print("  Google Sheets Authorization")
    print("=" * 55)
    print()
    print("1. Trinh duyet se mo ra → Dang nhap → Bam 'Allow'")
    print("2. Trinh duyet chuyen sang trang localhost (co the")
    print("   bao loi 'This site can't be reached') — BINH THUONG")
    print("3. Doi terminal hien 'Authorization thanh cong!'")
    print()
    print("!!! KHONG bam Ctrl+C trong luc cho !!!")
    print()

    try:
        flow = InstalledAppFlow.from_client_secrets_file(tmp_path, scopes=SCOPES)
        creds = flow.run_local_server(
            port=8888,
            open_browser=True,
            success_message="<h1>Authorization xong! Dong tab nay va quay lai terminal.</h1>",
        )
    finally:
        os.unlink(tmp_path)

    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    print()
    print("Authorization thanh cong!")
    print(f"Token luu tai: {TOKEN_FILE}")
    print("Bay gio co the chay pipeline binh thuong.")


if __name__ == "__main__":
    main()
