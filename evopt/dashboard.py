"""Deliverable 8 - the interactive dashboard.

Two artefacts:

  report.html   the analytical figures, one per stage of the pipeline
  map.html      the geographic view: network, demand, coverage, recommendations

Charts follow one rule throughout - the form is chosen by the job the data
does, and colour is assigned by that job rather than picked for looks.
Magnitude gets a single-hue sequential ramp, polarity gets a diverging pair
with a neutral midpoint, identity gets categorical slots in fixed order, and
state gets the reserved status colours with a label attached so meaning never
rides on hue alone.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .viz_theme import FONT, SEQ_BLUE, STATUS, THEME, base_layout

log = logging.getLogger("evopt.dashboard")


def _seq_scale(theme: str = "light"):
    stops = SEQ_BLUE if theme == "light" else SEQ_BLUE[::-1]
    return [[i / (len(stops) - 1), c] for i, c in enumerate(stops)]


# =========================================================================== #
# Figures
# =========================================================================== #
def make_figures(art: Dict, cfg, theme: str = "light") -> Dict[str, "go.Figure"]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = THEME[theme]
    S = t["series"]
    figs: Dict[str, go.Figure] = {}

    def style(fig, title, height=380, legend=None, **kw):
        # A single series needs no legend box - the title names it - so the
        # top margin shrinks to match rather than leaving a band of white.
        named = [tr for tr in fig.data
                 if getattr(tr, "name", None) and tr.showlegend is not False]
        lay = base_layout(theme, title, height,
                          has_legend=len(named) >= 2 if legend is None else legend)
        lay.update(kw)
        fig.update_layout(**lay)
        return fig

    # ---- 1. SVD scree: where signal stops and noise begins -------------- #
    svd = art["svd"]
    k = int(svd["rank_used"])
    sig = np.asarray(svd["s"])[:20]
    tau = float(svd.get("gd_threshold", np.nan))
    x = list(range(1, len(sig) + 1))
    # Plotted on a log axis against the singular values themselves, not their
    # share of variance: sigma_1 carries ~90% of the energy here, so on a
    # linear share axis every other component is a flat line and the chart
    # cannot show the thing it exists to show. The log axis makes the decay
    # and the noise floor legible, and the Gavish-Donoho threshold - which
    # acts on singular values, not on shares - can be drawn where it belongs.
    # Dots, not bars: on a log axis a bar's length is no longer proportional
    # to its value, so the bar stops being an honest mark. A dot plot is also
    # the conventional form for a scree plot.
    f = go.Figure()
    f.add_scatter(x=x, y=sig, mode="lines+markers", name="Singular value",
                  line=dict(color=t["axis"], width=1),
                  marker=dict(size=10, color=[S[0] if i < k else t["muted"]
                                              for i in range(len(sig))],
                              line=dict(width=2, color=t["surface"])),
                  customdata=np.asarray(svd["explained_variance_ratio"])[:20] * 100,
                  hovertemplate="Component %{x}<br>sigma = %{y:,.0f}"
                                "<br>%{customdata:.2f}% of variance<extra></extra>")
    if np.isfinite(tau):
        f.add_hline(y=tau, line_dash="dash", line_color=STATUS["critical"])
        f.add_annotation(x=len(sig), y=np.log10(tau), showarrow=False, yshift=11,
                         xanchor="right", text="noise floor (Gavish-Donoho threshold)",
                         font=dict(color=STATUS["critical"], size=11))
    f.add_annotation(x=k, y=np.log10(sig[k - 1]), ax=28, ay=-30, showarrow=True,
                     arrowhead=0, arrowcolor=t["muted"], xanchor="left",
                     text=f"{k} components kept<br>{svd['energy_captured_at_k']*100:.1f}% of variance",
                     font=dict(color=t["text"], size=11), align="left")
    style(f, "Signal sits above the noise floor; everything below it is discarded")
    f.update_yaxes(title_text="Singular value (log scale)", type="log")
    f.update_xaxes(title_text="Component", dtick=2)
    figs["svd_scree"] = f

    # ---- 2. Rank sweep: the bias-variance trade-off, made visible --------- #
    if "rank_sweep" in art:
        rs = art["rank_sweep"]
        best = int(rs.loc[rs["relative_error"].idxmin(), "rank"])
        f = go.Figure()
        f.add_scatter(x=rs["rank"], y=rs["relative_error"] * 100, mode="lines+markers",
                      name="Error after denoising", line=dict(color=S[0], width=2),
                      marker=dict(size=7), meta={"slot": 0},
                      hovertemplate="Rank %{x}<br>Error %{y:.2f}%<extra></extra>")
        eb = svd.get("relative_error_before")
        if eb:
            f.add_hline(y=eb * 100, line_dash="dash", line_color=STATUS["critical"])
            f.add_annotation(x=rs["rank"].max(), y=eb * 100, yshift=10, showarrow=False,
                             text=f"raw sensor error {eb*100:.1f}%", xanchor="right",
                             font=dict(color=STATUS["critical"], size=11))
        f.add_annotation(x=best, y=rs["relative_error"].min() * 100, ay=-34, ax=0,
                         text=f"optimum: rank {best}", showarrow=True, arrowhead=0,
                         arrowcolor=t["muted"], font=dict(color=t["text"], size=11))
        style(f, "Too few components loses signal, too many keeps noise")
        f.update_yaxes(title_text="Reconstruction error vs clean signal (%)")
        f.update_xaxes(title_text="Components retained", dtick=1)
        figs["svd_rank_sweep"] = f

    # ---- 3. The panel before and after cleaning --------------------------- #
    panel = art["panel"]
    Xo, Xd = panel["X_obs"], svd["X_denoised"]
    si = np.linspace(0, Xo.shape[0] - 1, min(60, Xo.shape[0])).astype(int)
    ti = np.arange(min(168, Xo.shape[1]))
    zmax = float(np.nanpercentile(Xd[np.ix_(si, ti)], 99))
    f = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                      subplot_titles=("Observed (noise + dropouts)", f"SVD rank-{k} reconstruction"))
    for c, Z in enumerate([Xo[np.ix_(si, ti)], Xd[np.ix_(si, ti)]], start=1):
        f.add_heatmap(z=Z, colorscale=_seq_scale(theme), zmin=0, zmax=zmax,
                      showscale=(c == 2), row=1, col=c,
                      colorbar=dict(title=dict(text="veh/h", side="right"), thickness=12,
                                    tickfont=dict(color=t["muted"], size=10)),
                      hovertemplate="Sensor %{y}<br>Hour %{x}<br>%{z:.0f} veh/h<extra></extra>")
    style(f, "Sensor panel: measurement noise removed, gaps filled", height=430)
    f.update_xaxes(title_text="Hour of the study period", showgrid=False,
                   title_font=dict(color=t["text_secondary"], size=12),
                   tickfont=dict(color=t["muted"], size=11))
    f.update_yaxes(title_text="Monitoring point", showgrid=False, row=1, col=1)
    for a in f.layout.annotations:
        a.font.update(size=12, color=t["text_secondary"])
    figs["traffic_panel"] = f

    # ---- 4. Recovered daily regimes --------------------------------------- #
    hours = np.array([h.hour for h in panel["time_index"]])
    Vt = svd["Vt"][:min(k, 5)]
    f = go.Figure()
    for i, row in enumerate(Vt):
        prof = np.array([row[hours == h].mean() for h in range(24)])
        prof = prof / (np.abs(prof).max() or 1.0)
        f.add_scatter(x=list(range(24)), y=prof, mode="lines", name=f"Component {i+1}",
                      line=dict(color=S[i], width=2), meta={"slot": i},
                      hovertemplate=f"Component {i+1}<br>%{{x}}:00<br>%{{y:.2f}}<extra></extra>")
    style(f, "What the components mean: average shape over the day")
    f.update_yaxes(title_text="Normalised loading")
    f.update_xaxes(title_text="Hour of day", dtick=3)
    figs["temporal_components"] = f

    # ---- 5. Gram-Schmidt feature independence ----------------------------- #
    fi = art["feature_independence"]
    rr = fi["residual_ratio"].sort_values()
    thr = art.get("redundancy_threshold", 0.15)
    colors = [STATUS["critical"] if v < thr else S[0] for v in rr.values]
    # Redundancy is spelled out in the label as well as coloured, so the
    # finding never depends on distinguishing two hues.
    labels = [f"{v:.2f}  redundant" if v < thr else f"{v:.2f}" for v in rr.values]
    f = go.Figure()
    f.add_bar(x=rr.values, y=[n.replace("_", " ") for n in rr.index], orientation="h",
              marker_color=colors, marker_line_width=0, marker_cornerradius=3,
              text=labels, textposition="outside",
              textfont=dict(color=t["text_secondary"], size=11),
              hovertemplate="%{y}<br>Independent share %{x:.3f}<extra></extra>")
    f.add_vline(x=thr, line_dash="dash", line_color=t["muted"])
    f.add_annotation(x=thr, y=-0.72, text=f"redundancy threshold {thr}",
                     showarrow=False, xanchor="left", xshift=6,
                     font=dict(color=t["muted"], size=11))
    style(f, "Gram-Schmidt: how much each feature adds that the others do not",
          height=max(300, 34 * len(rr) + 110))
    f.update_xaxes(title_text="Residual after projecting out earlier features", range=[0, 1.12])
    figs["gram_schmidt"] = f

    # ---- 6. Do the centrality measures agree? ----------------------------- #
    corr = art["centrality_corr"]
    labels = [c.replace("_", " ") for c in corr.columns]
    f = go.Figure()
    f.add_heatmap(z=corr.values, x=labels, y=labels,
                  colorscale=[[s, c] for s, c in t["diverging"]], zmid=0, zmin=-1, zmax=1,
                  text=np.round(corr.values, 2), texttemplate="%{text}",
                  textfont=dict(size=11),
                  colorbar=dict(thickness=12, tickfont=dict(color=t["muted"], size=10)),
                  hovertemplate="%{y} vs %{x}<br>rho = %{z:.2f}<extra></extra>")
    style(f, "Spearman correlation between the five centrality measures", height=400)
    figs["centrality_corr"] = f

    # ---- 7. Coverage curve, before and after ------------------------------ #
    cc = art["coverage_curves"]
    f = go.Figure()
    for i, (name, df) in enumerate(cc.items()):
        f.add_scatter(x=df["radius_m"] / 1000.0, y=df["demand_covered_pct"], mode="lines",
                      name=name, line=dict(color=S[i], width=2), meta={"slot": i},
                      hovertemplate=f"{name}<br>%{{x:.1f}} km<br>%{{y:.1f}}% of demand<extra></extra>")
    f.add_vline(x=cfg.coverage_radius_m / 1000.0, line_dash="dot", line_color=t["muted"])
    f.add_annotation(x=cfg.coverage_radius_m / 1000.0, y=8, showarrow=False, xanchor="left",
                     xshift=6, text=f"service standard {cfg.coverage_radius_m/1000:.1f} km",
                     font=dict(color=t["muted"], size=11))
    style(f, "Share of charging demand within reach of a station")
    f.update_yaxes(title_text="Demand covered (%)", range=[0, 102])
    f.update_xaxes(title_text="Acceptable driving distance (km)")
    figs["coverage_curve"] = f

    # ---- 8. Pareto front: coverage against budget ------------------------- #
    pf, knee = art["pareto"], art["knee"]
    f = go.Figure()
    f.add_scatter(x=pf["total_cost_lakh"], y=pf["demand_covered_pct"], mode="lines+markers",
                  name="Greedy frontier", line=dict(color=S[0], width=2),
                  marker=dict(size=8), meta={"slot": 0}, customdata=pf["n_stations"],
                  hovertemplate="%{customdata} stations<br>Rs %{x:.0f} lakh<br>%{y:.1f}% covered<extra></extra>")
    f.add_scatter(x=[knee["total_cost_lakh"]], y=[knee["demand_covered_pct"]], mode="markers",
                  name=f"Best value: {knee['n_stations']} stations",
                  marker=dict(size=15, color="rgba(0,0,0,0)",
                              line=dict(color=STATUS["good"], width=3)),
                  hovertemplate="Knee of the curve<extra></extra>")
    f.add_annotation(x=knee["total_cost_lakh"], y=knee["demand_covered_pct"], ay=32, ax=10,
                     text=f"{knee['n_stations']} stations<br>{knee['demand_covered_pct']:.0f}% covered",
                     showarrow=True, arrowhead=0, arrowcolor=t["muted"],
                     font=dict(color=t["text"], size=11), align="left")
    style(f, "Diminishing returns: each extra station buys less coverage")
    f.update_yaxes(title_text="Demand covered (%)")
    f.update_xaxes(title_text="Capital cost (Rs lakh)")
    figs["pareto"] = f

    # ---- 9. Resilience ---------------------------------------------------- #
    if "resilience" in art:
        rc = art["resilience"]["curve"]
        f = go.Figure()
        for i, (mode, label) in enumerate([("targeted", "Busiest stations fail"),
                                           ("random", "Random failures")]):
            sub = rc[rc["mode"] == mode]
            f.add_scatter(x=sub["removed"], y=sub["demand_covered_pct"], mode="lines+markers",
                          name=label, line=dict(color=S[i], width=2), marker=dict(size=7),
                          meta={"slot": i},
                          hovertemplate=f"{label}<br>%{{x}} lost<br>%{{y:.1f}}% covered<extra></extra>")
        style(f, "Coverage under station outages")
        f.update_yaxes(title_text="Demand covered (%)")
        f.update_xaxes(title_text="Stations out of service", dtick=1)
        figs["resilience"] = f

    # ---- 10. Where the remaining gaps are --------------------------------- #
    if "community_coverage" in art:
        cbc = art["community_coverage"].head(14).iloc[::-1]
        vals = cbc["demand_covered_pct"].values
        cols = [STATUS["critical"] if v < 50 else STATUS["warning"] if v < 80
                else STATUS["good"] for v in vals]
        state = ["critical" if v < 50 else "at risk" if v < 80 else "served" for v in vals]
        f = go.Figure()
        f.add_bar(x=vals, y=[f"Neighbourhood {int(i)}" for i in cbc.index], orientation="h",
                  marker_color=cols, marker_line_width=0, marker_cornerradius=3,
                  text=[f"{v:.0f}%  {s}" for v, s in zip(vals, state)],
                  textposition="outside", textfont=dict(color=t["text_secondary"], size=11),
                  customdata=np.stack([cbc["demand_units"], cbc["mean_distance_m"]], axis=-1),
                  hovertemplate=("%{y}<br>%{x:.1f}% covered<br>%{customdata[0]:.0f} demand units"
                                 "<br>mean drive %{customdata[1]:.0f} m<extra></extra>"))
        style(f, "Worst-served neighbourhoods after the recommended build",
              height=max(320, 30 * len(cbc) + 110))
        f.update_xaxes(title_text="Demand covered (%)", range=[0, 118])
        figs["community_gaps"] = f

    # ---- 11. Will the chosen stations be busy enough? --------------------- #
    if "station_load" in art:
        sl = art["station_load"].sort_values("utilisation_pct", ascending=True)
        vals = sl["utilisation_pct"].values
        cols = [STATUS["critical"] if v > 100 else STATUS["warning"] if v > 80
                else S[0] for v in vals]
        f = go.Figure()
        # "Over capacity" is written on the bar, not just coloured: a status
        # colour never carries meaning on its own.
        tags = ["over" if v > 100 else "" for v in vals]
        f.add_bar(x=[f"{int(n)}" for n in sl["station_node"]], y=vals,
                  marker_color=cols, marker_line_width=0, marker_cornerradius=3,
                  text=tags, textposition="outside",
                  textfont=dict(color=STATUS["critical"], size=11),
                  customdata=sl["assigned_demand"],
                  hovertemplate=("Junction %{x}<br>%{y:.0f}% of capacity"
                                 "<br>%{customdata:.0f} demand units<extra></extra>"))
        f.add_hline(y=100, line_dash="dash", line_color=t["muted"])
        f.add_annotation(x=1, xref="paper", y=100, text="capacity", showarrow=False,
                         xanchor="right", yshift=10, font=dict(color=t["muted"], size=11))
        style(f, "Projected utilisation of each station")
        f.update_yaxes(title_text="Utilisation (% of capacity)")
        f.update_xaxes(title_text="Junction id", type="category")
        figs["station_load"] = f

    return figs


# =========================================================================== #
# HTML report
# =========================================================================== #
SECTIONS = [
    ("Unit 1.3 - Cleaning the traffic data with SVD",
     "Hourly flow from the monitoring network is noisy and full of gaps. The "
     "singular value decomposition separates the handful of real traffic regimes "
     "from the measurement noise spread thinly across every other direction.",
     ["svd_scree", "svd_rank_sweep", "traffic_panel", "temporal_components"]),
    ("Unit 1.4 - Checking the features are independent",
     "Gram-Schmidt orthogonalisation measures how much of each feature is left "
     "once the earlier features are projected out. A short bar means the feature "
     "is already implied by the others and is carrying no new information.",
     ["gram_schmidt", "centrality_corr"]),
    ("Units 4.3 and 4.4 - Coverage and the siting decision",
     "Coverage is measured along the road network, never in a straight line, and "
     "weighted by charging demand rather than by counting junctions.",
     ["coverage_curve", "pareto"]),
    ("Where the plan is weakest",
     "A good city-wide average can still hide neighbourhoods that are stranded, "
     "and a plan that collapses when one station goes offline is not a good plan.",
     ["community_gaps", "station_load", "resilience"]),
]


def build_report(cfg, art: Dict, path: Optional[Path] = None) -> Path:
    """Render every figure into one self-contained page with a theme toggle."""
    import plotly.io as pio

    path = Path(path) if path else cfg.out_path / "report.html"
    specs = {}
    for theme in ("light", "dark"):
        for name, fig in make_figures(art, cfg, theme).items():
            specs.setdefault(name, {})[theme] = json.loads(pio.to_json(fig))

    tiles = _stat_tiles(art, cfg)
    tables = _tables(art)

    body = []
    for title, blurb, names in SECTIONS:
        present = [n for n in names if n in specs]
        if not present:
            continue
        cards = "\n".join(
            f'<figure class="card"><div class="plot" id="fig-{n}"></div></figure>'
            for n in present)
        body.append(
            f'<section><h2>{title}</h2><p class="lede">{blurb}</p>{cards}</section>')

    html = _REPORT_TEMPLATE.format(
        area=cfg.area_name,
        tiles=tiles,
        sections="\n".join(body),
        tables=tables,
        specs=json.dumps(specs),
        plotly=_plotly_js(),
    )
    path.write_text(html, encoding="utf-8")
    log.info("report written to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def _plotly_js() -> str:
    """Inline the Plotly bundle so the report works with no network."""
    import plotly

    p = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if p.exists():
        return f"<script>{p.read_text(encoding='utf-8')}</script>"
    return ('<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/'
            'plotly.min.js"></script>')


def _stat_tiles(art: Dict, cfg) -> str:
    b, a = art["coverage_before"], art["coverage_after"]
    recs = art["result"]["recommendations"]
    items = [
        ("Demand within reach of a charger", f"{a['demand_covered_pct']:.0f}%",
         f"was {b['demand_covered_pct']:.0f}% with the {art['n_existing']} stations mapped today",
         "good"),
        ("Average drive to a charger", f"{a['mean_distance_m']/1000:.2f} km",
         f"down from {b['mean_distance_m']/1000:.2f} km", "good"),
        ("Stations recommended", f"{len(recs)}",
         f"Rs {art['result']['total_cost_lakh']:.0f} lakh of capital", None),
        ("Worst-case drive", f"{a['p90_distance_m']/1000:.1f} km",
         "90th percentile of demand", None),
    ]
    out = []
    for label, value, sub, status in items:
        dot = (f'<span class="dot" style="background:{STATUS[status]}"></span>'
               if status else "")
        out.append(f'<div class="tile"><div class="tile-label">{label}</div>'
                   f'<div class="tile-value">{value}</div>'
                   f'<div class="tile-sub">{dot}{sub}</div></div>')
    return "\n".join(out)


def _tables(art: Dict) -> str:
    """The table view - the relief for low-contrast slots and for screen readers."""
    recs = art["result"]["recommendations"].copy()
    cols = {"rank": "Rank", "node": "Junction", "lat": "Latitude", "lon": "Longitude",
            "priority_score": "Priority", "marginal_coverage_pct": "Coverage added (pp)",
            "cost_lakh": "Cost (Rs lakh)"}
    recs = recs[list(cols)].rename(columns=cols).round(
        {"Latitude": 5, "Longitude": 5, "Priority": 1,
         "Coverage added (pp)": 2, "Cost (Rs lakh)": 1})
    return ('<section><h2>Recommended sites, in build order</h2>'
            '<p class="lede">Ordered by the coverage each site adds given the ones '
            'before it, which is the order a phased rollout should follow.</p>'
            + recs.to_html(index=False, border=0, classes="data")
            + "</section>")


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EV Charging Analysis</title>
{plotly}
<style>
  :root {{
    color-scheme: light;
    --surface:#fcfcfb; --plane:#f9f9f7; --text:#0b0b0b; --text2:#52514e;
    --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,0.10);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface:#1a1a19; --plane:#0d0d0d; --text:#fff; --text2:#c3c2b7;
      --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,0.10);
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface:#1a1a19; --plane:#0d0d0d; --text:#fff; --text2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,0.10);
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--plane); color:var(--text);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:40px 16px 96px; }}
  header {{ display:flex; justify-content:space-between; align-items:flex-start;
    gap:16px; flex-wrap:wrap; margin-bottom:8px; }}
  h1 {{ font-size:1.6rem; margin:0 0 4px; letter-spacing:-0.01em; }}
  .sub {{ color:var(--text2); margin:0 0 28px; }}
  h2 {{ font-size:1.12rem; margin:44px 0 6px; letter-spacing:-0.01em; }}
  .lede {{ color:var(--text2); margin:0 0 18px; max-width:68ch; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(196px,1fr));
    gap:12px; margin:22px 0 10px; }}
  .tile {{ background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:16px 18px; }}
  .tile-label {{ font-size:.78rem; color:var(--muted); text-transform:uppercase;
    letter-spacing:.04em; }}
  .tile-value {{ font-size:1.95rem; font-weight:600; margin:6px 0 2px;
    letter-spacing:-0.02em; }}
  .tile-sub {{ font-size:.85rem; color:var(--text2); }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
    margin-right:6px; vertical-align:middle; }}
  .card {{ background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:10px 8px 6px; margin:0 0 16px; }}
  table.data {{ width:100%; border-collapse:collapse; background:var(--surface);
    border:1px solid var(--border); border-radius:12px; overflow:hidden;
    font-size:.88rem; font-variant-numeric:tabular-nums; }}
  table.data th {{ text-align:left; color:var(--muted); font-weight:600;
    font-size:.76rem; text-transform:uppercase; letter-spacing:.04em;
    padding:10px 12px; border-bottom:1px solid var(--border); }}
  table.data td {{ padding:9px 12px; border-bottom:1px solid var(--grid);
    color:var(--text2); }}
  table.data tr:last-child td {{ border-bottom:none; }}
  .toggle {{ background:var(--surface); color:var(--text2); cursor:pointer;
    border:1px solid var(--border); border-radius:999px; padding:7px 15px;
    font-size:.84rem; font-family:inherit; }}
  .toggle:hover {{ color:var(--text); }}
  footer {{ margin-top:56px; padding-top:20px; border-top:1px solid var(--border);
    color:var(--muted); font-size:.83rem; }}
  @media (max-width:560px) {{ .wrap {{ padding:24px 16px 72px; }} h1 {{ font-size:1.35rem; }} }}
</style></head>
<body><div class="wrap">
<header>
  <div>
    <h1>Optimal placement of EV charging stations</h1>
    <p class="sub">{area} &middot; road-network optimisation and centrality analysis</p>
  </div>
  <button class="toggle" id="themeBtn" type="button">Switch theme</button>
</header>
<div class="tiles">{tiles}</div>
{sections}
{tables}
<footer>
  Road network and points of interest from OpenStreetMap. The hourly traffic
  panel is simulated from a known low-rank structure so that the SVD stage can
  be validated against a clean signal; every other input is measured data.
</footer>
</div>
<script>
const SPECS = {specs};
function currentTheme() {{
  const set = document.documentElement.getAttribute('data-theme');
  if (set) return set;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}}
function draw() {{
  const th = currentTheme();
  for (const [name, byTheme] of Object.entries(SPECS)) {{
    const el = document.getElementById('fig-' + name);
    if (!el) continue;
    const s = byTheme[th] || byTheme.light;
    Plotly.react(el, s.data, s.layout, {{displayModeBar:false, responsive:true}});
  }}
}}
document.getElementById('themeBtn').addEventListener('click', () => {{
  document.documentElement.setAttribute('data-theme',
    currentTheme() === 'dark' ? 'light' : 'dark');
  draw();
}});
window.matchMedia('(prefers-color-scheme: dark)')
  .addEventListener('change', () => {{
    if (!document.documentElement.getAttribute('data-theme')) draw();
  }});
draw();
</script>
</body></html>"""


# =========================================================================== #
# Interactive map
# =========================================================================== #
def _ramp_color(value: float, vmin: float, vmax: float, stops: List[str]) -> str:
    if not np.isfinite(value) or vmax <= vmin:
        return stops[-1]
    f = (value - vmin) / (vmax - vmin)
    return stops[int(np.clip(round(f * (len(stops) - 1)), 0, len(stops) - 1))]


def _point_layer(name: str, frame: pd.DataFrame, value_col: str, stops: List[str],
                 popup_fmt, vmin=None, vmax=None, radius=3.4, show=False):
    """One compact GeoJSON layer, styled by magnitude."""
    import folium

    vals = frame[value_col].values.astype(float)
    finite = vals[np.isfinite(vals)]
    vmin = float(np.nanpercentile(finite, 2)) if vmin is None else vmin
    vmax = float(np.nanpercentile(finite, 98)) if vmax is None else vmax

    features = []
    for (_, row), v in zip(frame.iterrows(), vals):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": {"c": _ramp_color(v, vmin, vmax, stops),
                           "tip": popup_fmt(row, v)},
        })
    gj = {"type": "FeatureCollection", "features": features}

    return folium.GeoJson(
        gj, name=name, show=show,
        marker=folium.CircleMarker(radius=radius, weight=0, fill=True, fill_opacity=0.85),
        style_function=lambda f: {"fillColor": f["properties"]["c"],
                                  "color": f["properties"]["c"], "fillOpacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False, sticky=False),
    )


def build_map(cfg, art: Dict, path: Optional[Path] = None) -> Path:
    """Geographic view: demand, priority, coverage before/after, recommendations."""
    import folium

    path = Path(path) if path else cfg.out_path / "map.html"
    nodes_df = art["node_view"]
    recs = art["result"]["recommendations"]
    existing = art["existing"]

    # OpenStreetMap's own tiles need no API key. CARTO's Positron basemap is
    # the better-looking choice but now watermarks every tile unless a key is
    # supplied, so it is offered as an opt-in rather than the default.
    m = folium.Map(
    location=list(cfg.center),
    zoom_start=13,
    tiles=None,
    control_scale=True
)
    folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri — Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom, 2012",
    name="Street Map",
    overlay=False,
    control=True
).add_to(m)

    # --- analysis layers ---------------------------------------------------
    _point_layer("Charging demand", nodes_df, "demand", SEQ_BLUE,
                 lambda r, v: f"Junction {r.name}<br>{v:.1f} demand units",
                 show=False).add_to(m)

    _point_layer("Siting priority score", nodes_df, "priority_score", SEQ_BLUE,
                 lambda r, v: f"Junction {r.name}<br>priority {v:.0f}/100",
                 vmin=0, vmax=100, show=False).add_to(m)

    _point_layer("Drive to charger - today", nodes_df, "dist_before", SEQ_BLUE,
                 lambda r, v: (f"Junction {r.name}<br>{v/1000:.2f} km to nearest charger"
                               if np.isfinite(v) else "unreachable"),
                 vmin=0, vmax=6000, show=False).add_to(m)

    _point_layer("Drive to charger - after build", nodes_df, "dist_after", SEQ_BLUE,
                 lambda r, v: f"Junction {r.name}<br>{v/1000:.2f} km to nearest charger",
                 vmin=0, vmax=6000, show=True).add_to(m)

    # --- service radius of the recommended sites ---------------------------
    ring = folium.FeatureGroup(name=f"Service radius ({cfg.coverage_radius_m/1000:.0f} km)",
                               show=True)
    for _, r in recs.iterrows():
        folium.Circle(location=[r["lat"], r["lon"]], radius=cfg.coverage_radius_m,
                      color="#2a78d6", weight=1, opacity=0.35,
                      fill=True, fill_opacity=0.045).add_to(ring)
    ring.add_to(m)

    # --- existing stations -------------------------------------------------
    grp = folium.FeatureGroup(name=f"Existing stations ({len(existing)})", show=True)
    for _, r in existing.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=7, color="#ffffff", weight=2,
            fill=True, fill_color="#52514e", fill_opacity=1,
            tooltip=f"Existing: {r.get('name','charging point')}").add_to(grp)
    grp.add_to(m)

    # --- recommendations ---------------------------------------------------
    grp = folium.FeatureGroup(name=f"Recommended sites ({len(recs)})", show=True)
    for _, r in recs.iterrows():
        html = (f"<div style='font-family:system-ui,sans-serif;font-size:13px;min-width:190px'>"
                f"<b>Priority {int(r['rank'])}</b><br>Junction {int(r['node'])}<br>"
                f"Adds <b>{r['marginal_coverage_pct']:.1f} pp</b> of coverage<br>"
                f"Siting score {r['priority_score']:.0f}/100<br>"
                f"Capital cost Rs {r['cost_lakh']:.0f} lakh<br>"
                f"<span style='color:#898781'>{r['lat']:.5f}, {r['lon']:.5f}</span></div>")
        folium.Marker(
            [r["lat"], r["lon"]],
            icon=folium.DivIcon(html=(
                f"<div style='background:#eb6834;color:#fff;width:24px;height:24px;"
                f"border-radius:50%;border:2px solid #fff;display:flex;"
                f"align-items:center;justify-content:center;font:600 12px "
                f"system-ui,sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.4)'>"
                f"{int(r['rank'])}</div>"), icon_size=(24, 24), icon_anchor=(12, 12)),
            popup=folium.Popup(html, max_width=260),
            tooltip=f"Priority {int(r['rank'])}: +{r['marginal_coverage_pct']:.1f} pp"
        ).add_to(grp)
    grp.add_to(m)

    # Frame the study area rather than trusting a fixed zoom level, which
    # would leave a 2 km run lost in the middle of the city.
    m.fit_bounds([[float(nodes_df["lat"].min()), float(nodes_df["lon"].min())],
                  [float(nodes_df["lat"].max()), float(nodes_df["lon"].max())]],
                 padding=(18, 18))

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_MAP_LEGEND.format(
        after=art["coverage_after"]["demand_covered_pct"],
        before=art["coverage_before"]["demand_covered_pct"],
        n=len(recs), radius=cfg.coverage_radius_m / 1000.0,
        ramp="".join(f"<span style='flex:1;height:8px;background:{c}'></span>"
                     for c in SEQ_BLUE))))

    m.save(str(path))
    log.info("map written to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


_MAP_LEGEND = """
<div style="position:fixed; bottom:22px; left:22px; z-index:9999; width:236px;
  background:#fcfcfb; border:1px solid rgba(11,11,11,.12); border-radius:10px;
  padding:13px 15px; font:13px/1.5 system-ui,-apple-system,sans-serif;
  color:#52514e; box-shadow:0 2px 10px rgba(0,0,0,.09)">
  <div style="font-weight:600;color:#0b0b0b;margin-bottom:7px">Recommended build</div>
  <div><b style="color:#0b0b0b;font-size:19px">{after:.0f}%</b> of demand covered</div>
  <div style="color:#898781;margin-bottom:9px">from {before:.0f}% today,
    with {n} new stations</div>
  <div style="display:flex;gap:1px;margin-bottom:3px">{ramp}</div>
  <div style="display:flex;justify-content:space-between;color:#898781;font-size:11px">
    <span>near</span><span>drive to charger</span><span>far</span></div>
  <div style="margin-top:9px;display:flex;align-items:center;gap:7px">
    <span style="width:15px;height:15px;border-radius:50%;background:#eb6834;
      border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.15)"></span>
    <span>recommended site</span></div>
  <div style="margin-top:5px;display:flex;align-items:center;gap:7px">
    <span style="width:15px;height:15px;border-radius:50%;background:#52514e;
      border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.15)"></span>
    <span>existing station</span></div>
  <div style="margin-top:8px;color:#898781;font-size:11px">
    Rings show the {radius:.0f} km service standard.</div>
</div>"""
