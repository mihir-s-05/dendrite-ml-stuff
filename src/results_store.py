"""Resumable per-cell result store backed by an append-only CSV.

Each sweep cell is keyed by ``(section, d, model, seed)``. Rows are flushed and
``fsync``'d on write, so an interrupted or crashed run resumes simply by skipping
cells that are already recorded. Reusable across sweep scripts.
"""

from __future__ import annotations

import csv
import os

Key = tuple[str, int, str, int]


class ResultStore:
    COLUMNS = ("section", "d", "model", "seed", "acc", "params", "secs")

    def __init__(self, path: str | None):
        self.path = path
        self._acc: dict[Key, float] = {}
        if path and os.path.exists(path):
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    self._acc[self._key(r["section"], r["d"], r["model"], r["seed"])] = float(r["acc"])

    @staticmethod
    def _key(section: str, d, model: str, seed) -> Key:
        return (section, int(d), model, int(seed))

    def __len__(self) -> int:
        return len(self._acc)

    def has(self, section: str, d: int, model: str, seed: int) -> bool:
        return self._key(section, d, model, seed) in self._acc

    def get(self, section: str, d: int, model: str, seed: int) -> float | None:
        return self._acc.get(self._key(section, d, model, seed))

    def record(self, section: str, d: int, model: str, seed: int,
               acc: float, params: int, secs: float) -> None:
        self._acc[self._key(section, d, model, seed)] = acc
        if not self.path:
            return
        is_new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(self.COLUMNS)
            w.writerow([section, int(d), model, int(seed), f"{acc:.6f}", params, f"{secs:.1f}"])
            f.flush()
            os.fsync(f.fileno())

    def values(self, section: str, d: int, model: str, seeds) -> list[float]:
        """Recorded accuracies for the given seeds (missing seeds omitted)."""
        return [self._acc[k] for s in seeds
                if (k := self._key(section, d, model, s)) in self._acc]
