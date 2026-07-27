"""The single source of truth for how PyMAPF figures look.

Every plot, animation and chart in :mod:`pymapf.viz` pulls its colors, ink and
surfaces from here, so a gallery of twenty figures reads as one system instead
of twenty matplotlib defaults.

The categorical slots below are a validated, colorblind-safe order: adjacent
pairs keep a CVD separation of ~9 (OKLab x100) in both modes. They are assigned
to agents *by fixed index*, never cycled by rank, so agent ``C`` keeps its color
when you re-run with fewer agents. Past eight agents the hues repeat but the
marker shape changes, so identity never rests on hue alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

# Categorical slots, in the fixed order that passes the adjacent-pair gates.
_CATEGORICAL_LIGHT = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
_CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

# Shape is the secondary channel: agent 9 reuses hue 1 but not its marker.
MARKERS: Sequence[str] = ("o", "s", "^", "D", "P", "X")

# Single-hue blue ramp, light -> dark, for magnitude (congestion, time).
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}


@dataclass(frozen=True)
class Theme:
    """Resolved colors for one mode (``"dark"`` or ``"light"``)."""

    mode: str = "dark"
    surface: str = "#1a1a19"
    plane: str = "#0d0d0d"
    ink: str = "#ffffff"
    ink_secondary: str = "#c3c2b7"
    muted: str = "#898781"
    grid: str = "#2c2c2a"
    axis: str = "#383835"
    obstacle: str = "#33332f"
    categorical: List[str] = field(default_factory=lambda: list(_CATEGORICAL_DARK))

    def agent_color(self, index: int) -> str:
        return self.categorical[index % len(self.categorical)]

    def agent_marker(self, index: int) -> str:
        return MARKERS[(index // len(self.categorical)) % len(MARKERS)]

    def color_map(self, names: Sequence[str]) -> Dict[str, str]:
        """Stable ``{agent name: color}``, assigned by position in ``names``."""
        return {name: self.agent_color(i) for i, name in enumerate(names)}

    def marker_map(self, names: Sequence[str]) -> Dict[str, str]:
        return {name: self.agent_marker(i) for i, name in enumerate(names)}


DARK = Theme()
LIGHT = Theme(
    mode="light",
    surface="#fcfcfb",
    plane="#f9f9f7",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    obstacle="#d8d7d0",
    categorical=list(_CATEGORICAL_LIGHT),
)

THEMES = {"dark": DARK, "light": LIGHT}


def get_theme(theme="dark") -> Theme:
    """Resolve a :class:`Theme` from a name or pass one through."""
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(
            "Unknown theme %r. Available: %s" % (theme, ", ".join(sorted(THEMES)))
        )


def apply(theme="dark") -> Theme:
    """Push a theme into matplotlib's rcParams and return it.

    Called by every plotting helper, so figures are consistent whether they are
    produced one at a time in a notebook or in a batch by the gallery script.
    """
    import matplotlib as mpl

    resolved = get_theme(theme)
    mpl.rcParams.update(
        {
            "figure.facecolor": resolved.plane,
            "savefig.facecolor": resolved.plane,
            "axes.facecolor": resolved.surface,
            "axes.edgecolor": resolved.axis,
            "axes.labelcolor": resolved.ink_secondary,
            "axes.titlecolor": resolved.ink,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": resolved.grid,
            "grid.linewidth": 0.8,
            "text.color": resolved.ink,
            "xtick.color": resolved.muted,
            "ytick.color": resolved.muted,
            "xtick.labelcolor": resolved.muted,
            "ytick.labelcolor": resolved.muted,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Segoe UI",
                "Helvetica Neue",
                "Arial",
            ],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "legend.labelcolor": resolved.ink_secondary,
            "figure.dpi": 120,
            "lines.solid_capstyle": "round",
        }
    )
    return resolved


def sequential_colormap(theme="dark"):
    """A one-hue blue colormap for magnitude encodings (never a rainbow)."""
    from matplotlib.colors import LinearSegmentedColormap

    resolved = get_theme(theme)
    stops = (
        SEQUENTIAL_BLUE if resolved.mode == "light" else list(reversed(SEQUENTIAL_BLUE))
    )
    # On the dark surface the ramp runs dark -> light so "more" stays "brighter".
    return LinearSegmentedColormap.from_list("pymapf_seq", stops, N=256)
