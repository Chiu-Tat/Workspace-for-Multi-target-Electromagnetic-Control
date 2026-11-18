"""Shared protocol, paths, and plotting helpers for Sections 6 and 7."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from emns.workspace import affine as coil_model
from hybrid_region._backend import COIL_MODEL_ORIGIN_OFFSET_M


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = Path(os.environ["EMNS_MANUSCRIPT_ROOT"]) if "EMNS_MANUSCRIPT_ROOT" in os.environ else None
OUTPUT_ROOT = PROJECT_ROOT / "sections_6_7_results"
MODEL_ROOT = OUTPUT_ROOT / "inputs"
ARCHIVED_REVISION = PROJECT_ROOT / "revision_analysis"
DIPOLE_RESULTS = PROJECT_ROOT / "dipole_error_comparison" / "results"
TASK_SWEEP_CSV = (
    PROJECT_ROOT / "Results" / "actuation_requirement_comparison" / "actuation_requirement_metrics.csv"
)

PAPER_Z_OFFSET_M = float(COIL_MODEL_ORIGIN_OFFSET_M)
SEARCH_BOX_M = ((-0.05, 0.05), (-0.05, 0.05))
SEARCH_AREA_M2 = 0.01
ANISOTROPIC_TASK_T = ((-0.12, 0.12), (-0.04, 0.04))
CURRENT_LIMIT_A = 15.0
COIL_MODEL_Z_M = 0.075
PAPER_PLANE_Z_M = COIL_MODEL_Z_M - PAPER_Z_OFFSET_M

CLASS_COLORS = {
    "inside": "#8fd19e",
    "outside": "#f4b6b6",
    "undetermined": "#d9d9d9",
}
CLASS_HATCHES = {"inside": "", "outside": "", "undetermined": ""}
CLASS_LABELS = {
    "inside": "TFW-in",
    "outside": "TFW-out",
    "undetermined": "Unresolved",
}


def configure_matplotlib(cache_dir: Path | None = None) -> None:
    cache = cache_dir or (OUTPUT_ROOT / ".mplconfig")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def output_dirs(root: Path = OUTPUT_ROOT) -> dict[str, Path]:
    paths = {
        "root": root,
        "configs": root / "configs",
        "data": root / "data",
        "figures": root / "figures",
        "validation": root / "validation",
        "writing": root / "writing",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


def paper_z_to_coil(z_paper_m: float) -> float:
    return float(z_paper_m + PAPER_Z_OFFSET_M)


def coil_positions_m() -> np.ndarray:
    params = np.asarray(coil_model.params_list, dtype=float)
    return params[:, 3:6]


def coil_moments() -> np.ndarray:
    params = np.asarray(coil_model.params_list, dtype=float)
    return params[:, :3]


def report_to_metrics(report, extra: dict | None = None) -> dict:
    widths = []
    if report.boxes:
        from hybrid_region import box_widths
        from hybrid_region.boxes import box_bounds

        for boxes in report.boxes.values():
            widths.extend(max(box_widths(box)) for box in boxes)
    payload = {
        **report.summary(),
        "terminal_boxes_m": {
            label: [{"lower": box_bounds(box)[0].tolist(),
                     "upper": box_bounds(box)[1].tolist()} for box in boxes]
            for label, boxes in report.boxes.items()
        },
        "terminal_width_m": {
            "n": len(widths),
            "min": min(widths) if widths else None,
            "median": float(np.median(widths)) if widths else None,
            "max": max(widths) if widths else None,
        },
    }
    if extra:
        payload.update(extra)
    return payload


def copy_to_paper(figure_path: Path, paper_name: str | None = None) -> Path | None:
    if PAPER_ROOT is None or not figure_path.exists():
        return None
    destination = PAPER_ROOT / "images" / (paper_name or figure_path.name)
    destination.write_bytes(figure_path.read_bytes())
    return destination


def add_region_legend(fig):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=CLASS_COLORS[k], label=CLASS_LABELS[k])
                        for k in CLASS_COLORS], loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))


def plot_pole_sphere(ax, centre, direction, radius, n=40):
    """Draw a shaded sphere with red and blue halves along its moment axis."""
    from matplotlib.colors import to_rgb

    axis = direction / np.linalg.norm(direction)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    theta, phi = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, n))
    unit = (np.cos(theta)[..., None] * axis
            + (np.sin(theta) * np.cos(phi))[..., None] * e1
            + (np.sin(theta) * np.sin(phi))[..., None] * e2)
    points = centre[None, None, :] + radius * unit
    face_theta = 0.5 * (theta[:-1, :-1] + theta[1:, 1:])
    normals = 0.25 * (unit[:-1, :-1] + unit[1:, :-1] + unit[:-1, 1:] + unit[1:, 1:])
    light = np.array([-0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.clip(normals @ light, 0, 1)
    base = np.where((face_theta < np.pi / 2)[..., None],
                    np.array(to_rgb("#e8384f")), np.array(to_rgb("#3f86e0")))
    colours = np.clip(base * shade[..., None], 0, 1)
    ax.plot_surface(points[..., 0], points[..., 1], points[..., 2], facecolors=colours,
                    linewidth=0, edgecolor="none", shade=False, antialiased=True)
