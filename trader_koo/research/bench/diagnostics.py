"""Counters and item lists for every place the bench substitutes, skips or rejects.

A run reports them under ``data.substitutions`` so nothing is replaced or dropped
without a number. ``count`` adds to a named counter; ``note`` records a distinct
item (a ticker, a filing) once however many reviews meet it again.
"""
from __future__ import annotations

from collections import Counter, defaultdict


class Diagnostics:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._items: dict[str, set] = defaultdict(set)

    def count(self, key: str, n: int = 1) -> None:
        if n:
            self._counts[key] += int(n)

    def note(self, key: str, item: tuple | str) -> None:
        self._items[key].add(item)

    def as_dict(self) -> dict:
        """Counters plus, per noted key, the number of distinct items and the sorted list."""
        out: dict = {k: v for k, v in sorted(self._counts.items())}
        for key, items in sorted(self._items.items()):
            out[key] = {"distinct": len(items), "items": [list(i) if isinstance(i, tuple) else i for i in sorted(items)]}
        return out
