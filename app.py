"""Thin entrypoint for the Gradio bone suppression demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def main() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from bone_suppression.gradio_app import main as run_demo

    run_demo()


if __name__ == "__main__":
    main()
