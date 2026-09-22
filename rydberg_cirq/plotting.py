"""Shared matplotlib styling so every generated figure looks consistent."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["apply_style", "PALETTE", "save_figure", "figure_path"]

#: Colour-blind-safe qualitative palette (Okabe-Ito), reused across all figures.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#808080",
}

SERIES_COLORS = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["red"],
    PALETTE["purple"],
    PALETTE["cyan"],
]


def apply_style() -> None:
    """Apply the shared publication-grade rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "lines.linewidth": 2.2,
            "mathtext.fontset": "dejavusans",
        }
    )


def figure_path(script_file: str, name: str) -> str:
    """Absolute path to ``name`` next to the script at ``script_file``."""
    return os.path.join(os.path.dirname(os.path.abspath(script_file)), name)


def save_figure(fig, script_file: str, name: str, dpi: int = 240) -> str:
    """Save ``fig`` beside the calling script and return the path."""
    path = figure_path(script_file, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved figure: {path}")
    return path
