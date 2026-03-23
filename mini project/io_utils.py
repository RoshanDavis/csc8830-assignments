"""I/O helpers for CSV and output directory management."""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List


def ensure_output_dirs(base_output_dir: str) -> Dict[str, str]:
    """Create and return all output directories used by the pipeline."""
    paths = {
        "base": base_output_dir,
        "per_video": os.path.join(base_output_dir, "per_video"),
        "plots": os.path.join(base_output_dir, "plots"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def write_csv(path: str, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    """Write rows to CSV with explicit header order."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
