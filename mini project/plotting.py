"""Plot generation for blink-rate and dimension summaries."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np


def plot_blink_rate_timeline(global_per_second_rows: List[Dict], output_path: str) -> None:
    """Generate blink-rate timeline (blinks per second over cumulative study time)."""
    import matplotlib.pyplot as plt

    if not global_per_second_rows:
        return

    cumulative_seconds = [float(row["cumulative_second"]) for row in global_per_second_rows]
    blink_per_second = [float(row["blinks_this_second"]) for row in global_per_second_rows]

    x_hours = np.array(cumulative_seconds, dtype=float) / 3600.0
    y = np.array(blink_per_second, dtype=float)

    plt.figure(figsize=(12, 4.5))
    plt.plot(x_hours, y, linewidth=1.0, color="#1f77b4", label="Blinks per second")
    plt.xlabel("Cumulative Study Time (hours)")
    plt.ylabel("Blinks per Second")
    plt.title("Estimated Blink Rate Across Study Duration")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_dimension_summary(global_means_mm: Dict[str, float], output_path: str) -> None:
    """Generate bar chart for average estimated facial dimensions."""
    import matplotlib.pyplot as plt

    if not global_means_mm:
        return

    labels = list(global_means_mm.keys())
    values = [float(global_means_mm[key]) for key in labels]

    pretty_labels = [label.replace("_", " ").title() for label in labels]

    plt.figure(figsize=(11, 5))
    bars = plt.bar(pretty_labels, values, color="#2ca02c", edgecolor="black")
    plt.ylabel("Estimated Size (mm)")
    plt.title("Average Estimated Facial Dimensions")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
