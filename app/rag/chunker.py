from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    start_char: int
    end_char: int


def split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not compact:
        return []

    chunks: list[Chunk] = []
    start = 0
    n = len(compact)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            boundary = compact.rfind("\n\n", start, end)
            if boundary <= start:
                boundary = compact.rfind(". ", start, end)
            if boundary > start + int(chunk_size * 0.55):
                end = boundary + (2 if compact[boundary: boundary + 2] == "\n\n" else 1)

        chunk_text = compact[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, start_char=start, end_char=end))

        if end >= n:
            break

        start = max(0, end - chunk_overlap)

    return chunks
