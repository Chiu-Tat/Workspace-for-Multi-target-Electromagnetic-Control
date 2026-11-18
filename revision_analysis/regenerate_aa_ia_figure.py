#!/usr/bin/env python3
# Run from the repository root: python -m revision_analysis.regenerate_aa_ia_figure
"""Regenerate the affine/interval comparison as a legible vector figure.

This reuses the archived workspace implementation without modifying that code.
The labels deliberately say "TFW-in/TFW-out" because machine-level
certification additionally requires the enclosure conditions stated in the paper.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
from analysis.affine.ten_coil import Create2DBox, Ws2DDetermination  # noqa: E402
from emns.numerics.interval import interval  # noqa: E402


def box_geometry(box):
    x_interval = box[0][0]
    y_interval = box[1][0]
    x0, x1 = x_interval[0][0], x_interval[0][1]
    y0, y1 = y_interval[0][0], y_interval[0][1]
    return x0, y0, x1 - x0, y1 - y0


def draw_boxes(axis, inside, outside, unresolved, title, runtime):
    styles = [
        (outside, "#f4b6b6", "xx", 0.45),
        (unresolved, "#d9d9d9", "....", 0.80),
        (inside, "#8fd19e", "", 0.75),
    ]
    for collection, colour, hatch, alpha in styles:
        for box in collection:
            x0, y0, width, height = box_geometry(box)
            axis.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    width,
                    height,
                    linewidth=0.25,
                    edgecolor="#222222",
                    facecolor=colour,
                    hatch=hatch,
                    alpha=alpha,
                )
            )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.05, 0.05)
    axis.set_ylim(-0.05, 0.05)
    axis.set_xlabel("$x$ (m)")
    axis.set_ylabel("$y$ (m)")
    axis.set_title(f"{title}\n{len(inside)} TFW-in; {len(unresolved)} unresolved", fontsize=9)
    axis.grid(True, linestyle=":", linewidth=0.45, color="#777777", alpha=0.55)


def run(use_affine):
    pmin, pmax = -0.05, 0.05
    initial_box = Create2DBox(interval[pmin, pmax], interval[pmin, pmax])
    start = time.perf_counter()
    _, inside, outside, unresolved, iterations = Ws2DDetermination(
        initial_box,
        -0.08,
        0.08,
        -15,
        15,
        0.002,
        -0.01 + 0.085,
        use_affine=use_affine,
        use_tight_certificates=False,
    )
    return {
        "inside": inside,
        "outside": outside,
        "unresolved": unresolved,
        "iterations": iterations,
        "runtime_s": time.perf_counter() - start,
    }


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    affine = run(True)
    standard_interval = run(False)

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), constrained_layout=False)
    draw_boxes(axes[0], affine["inside"], affine["outside"], affine["unresolved"], "(a) Affine enclosure", affine["runtime_s"])
    draw_boxes(
        axes[1],
        standard_interval["inside"],
        standard_interval["outside"],
        standard_interval["unresolved"],
        "(b) Standard interval enclosure",
        standard_interval["runtime_s"],
    )
    handles = [
        patches.Patch(facecolor="#8fd19e", edgecolor="#222222", label="TFW-in"),
        patches.Patch(facecolor="#f4b6b6", edgecolor="#222222", hatch="xx", label="TFW-out"),
        patches.Patch(facecolor="#d9d9d9", edgecolor="#222222", hatch="....", label="Unresolved"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=8)
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.88, wspace=0.24)

    image_root = SCRIPT_DIR
    pdf_path = image_root / "AA IA comparison revised.pdf"
    png_path = image_root / "AA IA comparison revised.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    metrics = {}
    for name, result in [("affine", affine), ("interval", standard_interval)]:
        metrics[name] = {
            "inside_boxes": len(result["inside"]),
            "outside_boxes": len(result["outside"]),
            "unresolved_boxes": len(result["unresolved"]),
            "iterations": result["iterations"],
            "runtime_s": result["runtime_s"],
        }
    (SCRIPT_DIR / "aa_ia_figure_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
