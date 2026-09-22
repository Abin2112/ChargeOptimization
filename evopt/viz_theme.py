"""Shared visual language for every figure in the report.

Colours come from a validated categorical palette. The slot order is the
colour-blind-safety mechanism, so slots are assigned in fixed order and never
cycled: a chart that would need a ninth series is the wrong chart.

Validation (OKLab dE x100, Machado 2009 CVD simulation at full severity):
  5-slot line charts, adjacent pairs : worst normal-vision dE 19.6 light /
                                       19.3 dark (floor 15); worst CVD dE 9.1
                                       light / 8.4 dark (target 8)          PASS
  3-slot all-pairs (scatter, maps)   : worst normal-vision dE 24.0 / 20.9;
                                       worst CVD dE 9.2 / 9.4               PASS

On the light surface slots 3-5 fall below 3:1 contrast, so every chart using
them carries a legend plus direct labels or a data table - identity is never
carried by colour alone.
"""
from __future__ import annotations

# --- categorical slots, fixed order ---------------------------------------- #
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
               "#008300", "#9085e9", "#e66767"]

# --- sequential: one hue, light -> dark ------------------------------------ #
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
# Ordinal use (discrete ordered marks) must clear 2:1 against the surface.
SEQ_BLUE_ORDINAL_LIGHT = SEQ_BLUE[3:]
SEQ_BLUE_ORDINAL_DARK = SEQ_BLUE[:-2]

# --- diverging: two poles, neutral grey midpoint --------------------------- #
DIVERGING_LIGHT = [(0.0, "#0d366b"), (0.5, "#f0efec"), (1.0, "#d03b3b")]
DIVERGING_DARK = [(0.0, "#86b6ef"), (0.5, "#383835"), (1.0, "#e66767")]

# --- status: reserved, never reused as a series ---------------------------- #
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

THEME = {
    "light": {
        "surface": "#fcfcfb", "plane": "#f9f9f7",
        "text": "#0b0b0b", "text_secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
        "series": SERIES_LIGHT, "diverging": DIVERGING_LIGHT,
        "seq_ordinal": SEQ_BLUE_ORDINAL_LIGHT,
    },
    "dark": {
        "surface": "#1a1a19", "plane": "#0d0d0d",
        "text": "#ffffff", "text_secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
        "series": SERIES_DARK, "diverging": DIVERGING_DARK,
        "seq_ordinal": SEQ_BLUE_ORDINAL_DARK,
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def base_layout(theme: str = "light", title: str = "", height: int = 380,
                has_legend: bool = True) -> dict:
    """Recessive chrome: hairline grid, no chart junk, generous margins."""
    t = THEME[theme]
    return {
        # The title sits in container space and the legend in plot space, with
        # a top margin wide enough for both - otherwise a two-series legend
        # lands on top of the title.
        "title": {"text": title, "font": {"size": 15, "color": t["text"], "family": FONT},
                  "x": 0, "xanchor": "left", "yref": "container", "y": 0.97,
                  "yanchor": "top"},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "size": 12, "color": t["text_secondary"]},
        "height": height,
        # Room for a title, and for a legend under it only when one is shown.
        "margin": {"l": 62, "r": 24,
                   "t": (86 if has_legend else 54) if title else 30, "b": 48},
        "showlegend": has_legend,
        # Thin marks: a wide gap keeps bars from reading as blocks of colour.
        "bargap": 0.45,
        "xaxis": {"gridcolor": t["grid"], "zeroline": False, "linecolor": t["axis"],
                  "tickfont": {"color": t["muted"], "size": 11},
                  "title": {"font": {"color": t["text_secondary"], "size": 12}}},
        "yaxis": {"gridcolor": t["grid"], "zeroline": False, "linecolor": t["axis"],
                  "tickfont": {"color": t["muted"], "size": 11},
                  "title": {"font": {"color": t["text_secondary"], "size": 12}}},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.03,
                   "xanchor": "left", "x": 0,
                   "font": {"color": t["text_secondary"], "size": 11},
                   "bgcolor": "rgba(0,0,0,0)"},
        "hoverlabel": {"font": {"family": FONT, "size": 12}},
        "hovermode": "closest",
    }
