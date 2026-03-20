from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _normalize_prompt_paths(
    prompt_files: str | Path | Iterable[str | Path],
) -> list[Path]:
    if isinstance(prompt_files, (str, Path)):
        return [Path(prompt_files)]
    return [Path(path) for path in prompt_files]


def load_prompt_from(prompt_files: str | Path | Iterable[str | Path]) -> str:
    paths = _normalize_prompt_paths(prompt_files)
    sections: list[str] = []

    for path in paths:
        content = path.read_text(encoding="utf-8").strip()
        sections.append(f"{path.name}\n----\n{content}")

    return "\n\n".join(sections)
