"""Traffic monitoring panel - the data matrix that Unit 1.3 (SVD) operates on.

Public hourly loop-detector feeds do not exist for this study area, so the
panel is **simulated**, and that is stated plainly everywhere it is used.
What makes the simulation worth doing is that it is built the way real urban
traffic actually is: a small number of latent regimes (commute, errand,
freight, weekend) mixed differently at each monitoring point, then corrupted
by sensor noise and dropouts.

That structure is the point. Because the clean signal is known by
construction, the SVD stage can be *validated* rather than merely performed:
we can measure how much of the noise truncated SVD actually removes and
whether the recovered rank matches the rank that was put in.

Matrix produced
---------------
    X  (n_sensors x n_timesteps)   hourly vehicle flow index
    rows    = monitoring points on the road network
    columns = consecutive hours, starting on a Monday 00:00
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger("evopt.traffic")

# Relative carrying importance of each OSM road class.
ROAD_CLASS_RANK: Dict[str, float] = {
    "motorway": 6.0, "motorway_link": 4.5, "trunk": 5.0, "trunk_link": 3.5,
    "primary": 4.0, "primary_link": 3.0, "secondary": 3.0, "secondary_link": 2.4,
    "tertiary": 2.0, "tertiary_link": 1.6, "unclassified": 1.5,
    "residential": 1.0, "living_street": 0.5, "service": 0.4, "road": 1.2,
}

FACTOR_NAMES = ["commute", "errand_retail", "freight_night", "weekend_leisure", "baseline"]


# --------------------------------------------------------------------------- #
# Where to put the sensors
# --------------------------------------------------------------------------- #
def arterial_score(G: nx.DiGraph) -> pd.Series:
    """Road-class importance of each junction, from the roads that meet there.

    Uses only raw OSM tags, so it is available before any centrality is
    computed and introduces no circularity into the pipeline.
    """
    scores = {}
    for n in G.nodes:
        ranks = []
        for _, _, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True)):
            hwy = d.get("highway", "residential")
            if isinstance(hwy, (list, tuple)):
                hwy = hwy[0] if hwy else "residential"
            hwy = str(hwy).strip("[]'\" ")
            ranks.append(ROAD_CLASS_RANK.get(hwy, 1.0))
        scores[n] = float(np.mean(ranks) * np.sqrt(len(ranks))) if ranks else 0.0
    return pd.Series(scores, name="arterial").sort_index()


def select_sensor_nodes(cfg, G: nx.DiGraph, dem: pd.DataFrame, rng=None) -> List:
    """Choose monitoring points: mostly arterial, with spatial spread.

    Real sensor networks are not placed at random and not purely greedily
    either, so we take 70% by importance and 30% at random to keep quiet
    residential areas represented.
    """
    rng = rng or np.random.default_rng(cfg.seed)
    art = arterial_score(G)
    d = dem["demand"].reindex(art.index).fillna(0.0)
    imp = (art / art.max()) * 0.6 + (d / d.max()) * 0.4

    k = int(min(cfg.n_sensors, len(imp)))
    n_top = int(0.7 * k)
    top = imp.nlargest(n_top).index.tolist()
    rest = [n for n in imp.index if n not in set(top)]
    extra = rng.choice(rest, size=min(k - n_top, len(rest)), replace=False).tolist() if rest else []
    return sorted(top + extra)


# --------------------------------------------------------------------------- #
# Latent temporal regimes
# --------------------------------------------------------------------------- #
def temporal_factors(n_timesteps: int, rank: int, rng) -> pd.DataFrame:
    """Build `rank` interpretable hourly profiles over `n_timesteps` hours."""
    t = np.arange(n_timesteps)
    hour = t % 24
    dow = (t // 24) % 7
    weekday = (dow < 5).astype(float)
    weekend = 1.0 - weekday

    def bump(centre, width):
        d = np.minimum(np.abs(hour - centre), 24 - np.abs(hour - centre))
        return np.exp(-0.5 * (d / width) ** 2)

    profiles = {
        # Morning and evening office peaks, weekdays only.
        "commute": weekday * (1.00 * bump(9.0, 1.5) + 0.95 * bump(19.0, 2.0)) + 0.05,
        # Shopping and errands: broad midday hump, stronger at the weekend.
        "errand_retail": (0.7 * weekday + 1.3 * weekend) * bump(14.5, 3.2) + 0.05,
        # Goods movement, permitted in Indian cities mainly at night.
        "freight_night": (0.9 * weekday + 0.6 * weekend) * bump(2.5, 2.8) + 0.04,
        # Weekend leisure: late afternoon into the evening.
        "weekend_leisure": weekend * (bump(17.5, 3.0) + 0.6 * bump(21.0, 2.0)) + 0.03,
        # Slowly drifting background level (weather, season, fuel price).
        "baseline": 0.55 + 0.12 * np.sin(2 * np.pi * t / max(n_timesteps, 1))
                    + 0.05 * np.sin(2 * np.pi * t / 72.0),
    }
    names = FACTOR_NAMES[:rank]
    if rank > len(FACTOR_NAMES):  # extra regimes, if ever asked for
        for i in range(len(FACTOR_NAMES), rank):
            names.append(f"regime_{i}")
            profiles[names[-1]] = 0.3 + 0.2 * rng.standard_normal(n_timesteps).cumsum() / np.sqrt(n_timesteps)

    V = np.vstack([profiles[k] for k in names])
    idx = pd.date_range("2025-01-06 00:00", periods=n_timesteps, freq="h")  # a Monday
    return pd.DataFrame(V, index=names, columns=idx)


def sensor_loadings(cfg, G: nx.DiGraph, dem: pd.DataFrame, sensor_nodes: List,
                    rank: int, rng) -> pd.DataFrame:
    """How strongly each monitoring point expresses each traffic regime.

    Loadings are tied to the physical character of the junction: arterial
    roads carry commuters and freight, retail-heavy junctions carry errands.
    """
    art = arterial_score(G).reindex(sensor_nodes)
    art_n = (art / art.max()).values

    dsub = dem.reindex(sensor_nodes)
    d_n = (dsub["demand"] / dsub["demand"].max()).values

    def cat(col):
        if col in dsub.columns:
            v = dsub[col].fillna(0.0).values
            return v / (v.max() + 1e-12)
        return np.zeros(len(sensor_nodes))

    retail = np.maximum(cat("demand_shop"), cat("demand_amenity"))
    resid = cat("demand_building")

    cols = {
        "commute": 0.25 + 0.75 * art_n * (0.5 + 0.5 * resid),
        "errand_retail": 0.10 + 0.90 * retail,
        "freight_night": 0.05 + 0.95 * np.clip(art_n - 0.35, 0, None) / 0.65,
        "weekend_leisure": 0.08 + 0.92 * (0.6 * retail + 0.4 * d_n),
        "baseline": 0.40 + 0.60 * (0.5 * art_n + 0.5 * d_n),
    }
    names = FACTOR_NAMES[:rank] + [f"regime_{i}" for i in range(len(FACTOR_NAMES), rank)]
    U = np.column_stack([
        cols.get(k, rng.random(len(sensor_nodes))) for k in names
    ])
    # Small idiosyncratic variation: two junctions of the same type still differ.
    U *= (1.0 + 0.12 * rng.standard_normal(U.shape))
    return pd.DataFrame(np.clip(U, 0.01, None), index=sensor_nodes, columns=names)


# --------------------------------------------------------------------------- #
def simulate_traffic_panel(cfg, G: nx.DiGraph, dem: pd.DataFrame,
                           sensor_nodes: Optional[List] = None) -> Dict:
    """Produce the clean signal, the observed noisy panel and the metadata."""
    rng = np.random.default_rng(cfg.seed)
    rank = int(cfg.true_rank)
    sensors = sensor_nodes or select_sensor_nodes(cfg, G, dem, rng)

    V = temporal_factors(cfg.n_timesteps, rank, rng)          # rank x T
    U = sensor_loadings(cfg, G, dem, sensors, rank, rng)      # S x rank

    # Regime strengths: commute dominates, later regimes matter progressively
    # less. These become the singular values the SVD should recover.
    strengths = np.array([1.0, 0.58, 0.34, 0.22, 0.14])[:rank]
    if rank > 5:
        strengths = np.concatenate([strengths, 0.12 * 0.7 ** np.arange(rank - 5)])

    X_true = (U.values * strengths) @ V.values                 # S x T
    X_true *= 1200.0 / X_true.mean()                           # vehicles/hour scale

    noise = cfg.noise_sigma * X_true.std() * rng.standard_normal(X_true.shape)
    X_obs = X_true + noise

    # Detectors also fail: a share of readings comes back missing.
    mask = rng.random(X_true.shape) < cfg.dropout_rate
    X_obs = np.where(mask, np.nan, X_obs)
    X_obs = np.clip(X_obs, 0.0, None)

    log.info("traffic panel: %d sensors x %d hours, true rank %d, "
             "noise sigma %.0f veh/h, %.1f%% readings missing",
             X_true.shape[0], X_true.shape[1], rank,
             cfg.noise_sigma * X_true.std(), 100.0 * mask.mean())

    return {
        "X_true": X_true,
        "X_obs": X_obs,
        "missing_mask": mask,
        "sensor_nodes": list(sensors),
        "time_index": V.columns,
        "V_true": V,
        "U_true": U,
        "strengths": strengths,
        "is_simulated": True,
    }


def panel_frame(panel: Dict, which: str = "X_obs") -> pd.DataFrame:
    """The panel as a labelled DataFrame (sensors x hours)."""
    return pd.DataFrame(panel[which], index=panel["sensor_nodes"], columns=panel["time_index"])


def interpolate_to_nodes(values: np.ndarray, sensors: List, nodes: List,
                         coords: pd.DataFrame, k: int = 6) -> pd.Series:
    """Carry a per-sensor quantity to every junction by inverse-distance weighting.

    The monitoring network covers only a subset of junctions, so any
    sensor-derived feature has to be extended to the rest. This is the
    weakest link in the traffic chain and is stated rather than hidden: the
    honest fix is more sensors, not a cleverer interpolator.
    """
    from scipy.spatial import cKDTree

    sxy = coords.reindex(sensors)[["x", "y"]].values
    axy = coords.reindex(nodes)[["x", "y"]].values
    tree = cKDTree(sxy)
    d, i = tree.query(axy, k=min(k, len(sensors)))
    d = np.atleast_2d(d.T).T if d.ndim == 1 else d
    i = np.atleast_2d(i.T).T if i.ndim == 1 else i
    w = 1.0 / np.maximum(d, 1.0)
    w /= w.sum(axis=1, keepdims=True)
    return pd.Series((np.asarray(values)[i] * w).sum(axis=1),
                     index=pd.Index(nodes, name="node"))
