from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


class PromptRenderer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.environment = Environment(
            loader=FileSystemLoader(str(root)),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def render(self, relative: str, **context: Any) -> str:
        return self.environment.get_template(relative).render(**context).strip()

    def system(self, relative: str) -> str:
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_text().strip()
