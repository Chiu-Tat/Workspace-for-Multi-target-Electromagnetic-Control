#!/usr/bin/env python3
"""Generate the Supplementary direct dipole/finite-body comparison figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main():
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    example = metrics.get("worked_boundary_example")
    if not example:
        raise RuntimeError("Run run_comparison.py first to create worked_boundary_example.")

    data = np.load(RESULTS / "masks_and_errors.npz")
    xs, ys = data["xs"] * 1e3, data["ys"] * 1e3
    valid, discrepancy = data["valid"], data["absolute_discrepancy"]
    force_discrepancy = np.max(np.where(np.isnan(discrepancy[:, :, 2:]), -np.inf,
                                        discrepancy[:, :, 2:]), axis=2) * 1e3
    force_discrepancy[~valid] = np.nan
    positive = force_discrepancy[np.isfinite(force_discrepancy) & (force_discrepancy > 0)]

    fig = plt.figure(figsize=(10.5, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=(0.9, 2.4))
    flow = fig.add_subplot(gs[0, 0])
    flow.set_axis_off()

    boxes = [
        (0.015, "Point dipole\n$\\mathbf{c}_{\\mathrm{dip}}(\\mathbf{q})$", "#f4a261"),
        (0.21, "Finite-body quadrature\n$\\mathbf{c}_{\\mathrm{FB}}(\\mathbf{q})$", "#2a9d8f"),
        (0.405, "Signed correction\n$\\mathbf{d}_{\\mathrm{FB}}=\\mathbf{c}_{\\mathrm{FB}}-\\mathbf{c}_{\\mathrm{dip}}$", "#8ecae6"),
        (0.60, "Dipole test\n$\\mathcal{C}\\subseteq\\mathbf{G}\\mathcal{I}+\\mathbf{c}_{\\mathrm{dip}}$", "#cdb4db"),
        (0.795, "Finite-body test\n$\\mathcal{C}\\subseteq\\mathbf{G}\\mathcal{I}+\\mathbf{c}_{\\mathrm{FB}}$", "#b7e4c7"),
    ]
    width = 0.175
    for x, label, color in boxes:
        flow.text(x + width / 2, 0.52, label, ha="center", va="center", fontsize=8,
                  transform=flow.transAxes,
                  bbox=dict(boxstyle="round,pad=.45", facecolor=color,
                            edgecolor="#333333", linewidth=.9))
    for x in (0.195, 0.39, 0.585, 0.78):
        flow.annotate("", xy=(x + .012, .52), xytext=(x - .012, .52),
                      xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#333333"))
    flow.text(0.0, 0.98, "(a) Matched pointwise model comparison",
              ha="left", va="top", fontsize=11, transform=flow.transAxes)
    flow.text(0.5, 0.01,
              "The comparison diagnoses model sensitivity; it does not establish a physical error envelope.",
              ha="center", va="bottom", fontsize=7.5, color="#8b1e1e",
              transform=flow.transAxes)

    lower = gs[1, 0].subgridspec(1, 2, width_ratios=(1.25, 1.0))
    table_ax = fig.add_subplot(lower[0, 0])
    table_ax.set_axis_off()
    table_ax.set_title("(b) Worked boundary sample", loc="left", fontsize=11)
    labels = example["component_order"]
    units = ["mT", "mT", "mN", "mN"]
    scale = np.array([1e3, 1e3, 1e3, 1e3])
    values = np.column_stack((example["c_dip"], example["c_finite"],
                              example["signed_discrepancy"], example["absolute_discrepancy"])) * scale
    cell_text = [[labels[i], units[i], *[f"{v:.3f}" for v in values[i]]] for i in range(4)]
    table = table_ax.table(cellText=cell_text,
                           colLabels=["component", "unit", "$c_{\\mathrm{dip}}$", "$c_{\\mathrm{FB}}$",
                                      "$d_{\\mathrm{FB}}$", "$|d_{\\mathrm{FB}}|$"],
                           cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.55)
    x_mm = example["pose"]["x_m"] * 1e3
    y_mm = example["pose"]["y_m"] * 1e3
    sep_mm = example["pose"]["center_separation_m"] * 1e3
    table_ax.text(0.5, 0.10,
                  f"$q^*=({x_mm:.2f},{y_mm:.2f})$ mm, $\\beta=90^\\circ$, separation $={sep_mm:.2f}$ mm",
                  ha="center", va="center", fontsize=8, transform=table_ax.transAxes)
    table_ax.text(0.5, 0.01,
                  "Point-dipole test: feasible    |    finite-body test: infeasible",
                  ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                  color="#8b1e1e", transform=table_ax.transAxes)

    map_ax = fig.add_subplot(lower[0, 1])
    im = map_ax.imshow(force_discrepancy, origin="lower",
                       extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       cmap="magma", interpolation="nearest",
                       norm=LogNorm(vmin=max(float(np.nanpercentile(positive, 1)), 1e-4),
                                    vmax=float(np.nanmax(positive))))
    map_ax.plot(x_mm, y_mm, marker="*", ms=11, color="#00ffff",
                markeredgecolor="black", markeredgewidth=.6, label="$q^*$")
    map_ax.set_title("(c) Maximum force-model discrepancy", fontsize=11)
    map_ax.set_xlabel("moving-magnet $x$ (mm)")
    map_ax.set_ylabel("$y$ (mm)")
    map_ax.set_aspect("equal")
    map_ax.legend(loc="upper left", fontsize=8)
    cb = fig.colorbar(im, ax=map_ax, shrink=.86)
    cb.set_label("$\\max(|d_{f_y}|,|d_{f_z}|)$ (mN)")

    fig.savefig(RESULTS / "dipole_error_worked_process.pdf")
    fig.savefig(RESULTS / "dipole_error_worked_process.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
