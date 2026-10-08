"""Entry point for Hugging Face Spaces deployment.

HF Spaces runs the `app_file` at the repo root (see README.md). The UI itself
lives in application/app.py; this file only puts the project root on sys.path
and launches it. Locally, `python app.py` and `python application/app.py`
both work.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.app import app, _CSS, _THEME  # noqa: E402

if __name__ == "__main__":
    app.launch(show_error=True, theme=_THEME, css=_CSS)
