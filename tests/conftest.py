# tests/conftest.py
# Stub out packages that are not installable in this environment
# so that unittest.mock.patch can find them for patching.
import sys
from unittest.mock import MagicMock

# paddleocr stubs kept for any legacy test imports
sys.modules.setdefault("paddleocr", MagicMock())
