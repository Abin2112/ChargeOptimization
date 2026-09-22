"""Charging-demand surface built from real OpenStreetMap land use.

Where people charge is driven by where they already stop: homes overnight,
offices during the working day, malls and hospitals for opportunity charging.
This module turns OSM points of interest into a demand value on every
intersection of the road graph.

Method
------
1. Download POIs (amenity / shop / office / building / landuse) for the area.
2. Give each POI a weight reflecting how much charging demand it generates.
3. Spread each POI's weight onto nearby intersections with a Gaussian kernel
   of bandwidth `demand_smoothing_m`. A driver parks *near* a junction, not on
   it, so a point mass at the nearest node would be a worse model than a
   smooth one.
4. Normalise to 10,000 demand units for the whole study area, so a node's
   value reads directly as "basis points of area-wide charging demand".

The POI inventory is real; the category weights are a stated modelling
assumption, collected in WEIGHTS below so they can be challenged and changed.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger("evopt.demand")

TOTAL_DEMAND_UNITS = 10_000.0

# How much charging demand one feature of each class generates, relative to a
# single dwelling. Long-dwell locations score highest because that is where
# charging actually happens.
WEIGHTS: Dict[Tuple[str, str], float] = {
    # --- residential: overnight charging, the largest single block ---------
    ("building", "apartments"): 6.0,
    ("building", "residential"): 4.0,
    ("building", "house"): 1.0,
    ("landuse", "residential"): 40.0,        # polygons cover many dwellings
    # --- workplace: daytime top-up ---------------------------------------
    ("building", "office"): 5.0,
    ("building", "commercial"): 4.0,
    ("office", "*"): 3.0,
    ("landuse", "commercial"): 25.0,
    ("landuse", "industrial"): 12.0,
    # --- destination / opportunity charging -------------------------------
    ("shop", "mall"): 30.0,
    ("shop", "supermarket"): 12.0,
    ("shop", "department_store"): 12.0,
    ("shop", "*"): 1.5,
    ("building", "retail"): 3.0,
    ("landuse", "retail"): 20.0,
    ("amenity", "parking"): 14.0,            # natural home for chargers
    ("amenity", "fuel"): 18.0,               # prime retrofit site
    ("amenity", "hospital"): 16.0,
    ("amenity", "college"): 10.0,
    ("amenity", "university"): 12.0,
    ("amenity", "school"): 5.0,
    ("amenity", "cinema"): 8.0,
    ("amenity", "marketplace"): 8.0,
    ("amenity", "bank"): 2.0,
    ("amenity", "restaurant"): 2.0,
    ("amenity", "cafe"): 1.5,
    ("amenity", "bus_station"): 10.0,
    ("amenity", "taxi"): 12.0,               # fleet charging demand
    ("amenity", "*"): 1.0,
}

TAG_QUERIES = [
    ("amenity", {"amenity": True}),
    ("shop", {"shop": True}),
    ("office", {"office": True}),
    ("building", {"building": ["residential", "apartments", "house",
                               "commercial", "retail", "office"]}),
    ("landuse", {"landuse": ["residential", "commercial", "retail", "industrial"]}),
]


def _weight_for(key: str, value: str) -> float:
    if (key, value) in WEIGHTS:
        return WEIGHTS[(key, value)]
    return WEIGHTS.get((key, "*"), 0.0)


# --------------------------------------------------------------------------- #
def fetch_pois(cfg, force: bool = False) -> pd.DataFrame:
    """Download and cache the POI inventory for the study area."""
    from .network import _configure_osmnx

    cache_file = Path(cfg.cache_dir) / f"pois_{cfg.slug()}.csv"
    if cache_file.exists() and not force:
        return pd.read_csv(cache_file)

    ox = _configure_osmnx(cfg)
    frames = []
    for key, tags in TAG_QUERIES:
        try:
            gdf = ox.features_from_point(cfg.center, tags=tags, dist=cfg.radius_m)
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("POI query %s failed: %s", key, exc)
            continue
        if not len(gdf):
            continue
        gdf = gdf.to_crs("EPSG:4326")
        # Polygons (buildings, land use) are represented by their centroid;
        # the kernel step then spreads them over the surrounding junctions.
        cent = gdf.geometry.centroid
        vals = gdf[key] if key in gdf.columns else pd.Series("yes", index=gdf.index)
        area_m2 = np.zeros(len(gdf))
        try:
            proj = gdf.to_crs(gdf.estimate_utm_crs())
            area_m2 = proj.geometry.area.values
        except Exception:
            pass
        frames.append(pd.DataFrame({
            "key": key,
            "value": vals.astype(str).values,
            "lat": cent.y.values,
            "lon": cent.x.values,
            "area_m2": area_m2,
        }))
        log.info("POIs %-9s %5d", key, len(gdf))

    if not frames:
        log.warning("no POI data retrieved - demand will fall back to topology")
        pois = pd.DataFrame(columns=["key", "value", "lat", "lon", "area_m2"])
    else:
        pois = pd.concat(frames, ignore_index=True)
        pois = pois.dropna(subset=["lat", "lon"])

    pois.to_csv(cache_file, index=False)
    return pois


def score_pois(pois: pd.DataFrame) -> pd.DataFrame:
    """Attach the demand weight of every POI."""
    if not len(pois):
        return pois.assign(weight=[])
    w = np.array([_weight_for(k, v) for k, v in zip(pois["key"], pois["value"])])

    # Land-use polygons vary enormously in size; scale them by footprint so a
    # 10-hectare residential block is not treated like a single plot. The
    # square root keeps one huge polygon from dominating the whole surface.
    is_poly = pois["key"].isin(["landuse", "building"]).values
    area = np.nan_to_num(pois["area_m2"].values, nan=0.0)
    scale = np.ones(len(pois))
    ref = 1000.0  # m^2, a typical urban plot
    scale[is_poly] = np.sqrt(np.clip(area[is_poly], ref, None) / ref)
    out = pois.copy()
    out["weight"] = w * scale
    return out[out["weight"] > 0].reset_index(drop=True)


# --------------------------------------------------------------------------- #
def build_demand_surface(cfg, G: nx.Graph, pois: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Project POI weights onto network nodes with a Gaussian kernel.

    Returns a frame indexed by node with `demand` (units summing to 10,000)
    and the per-category contributions that produced it.
    """
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.geometry import Point

    from .network import node_frame

    nf = node_frame(G)
    coords = nf[["x", "y"]].values
    tree = cKDTree(coords)
    n = len(nf)

    pois = fetch_pois(cfg) if pois is None else pois
    scored = score_pois(pois)

    if not len(scored):
        return _topological_fallback(G, nf)

    pts = gpd.GeoSeries([Point(lo, la) for la, lo in zip(scored["lat"], scored["lon"])],
                        crs="EPSG:4326").to_crs(G.graph.get("crs"))
    pxy = np.column_stack([pts.x.values, pts.y.values])

    sigma = float(cfg.demand_smoothing_m)
    cutoff = 3.0 * sigma
    demand = np.zeros(n)
    by_cat: Dict[str, np.ndarray] = {k: np.zeros(n) for k in scored["key"].unique()}

    neighbours = tree.query_ball_point(pxy, r=cutoff)
    weights = scored["weight"].values
    cats = scored["key"].values

    dropped = 0
    for i, idxs in enumerate(neighbours):
        if not idxs:
            # POI further than 3 sigma from any junction: give it to the
            # single nearest node rather than discarding real demand.
            _, j = tree.query(pxy[i], k=1)
            idxs = [int(j)]
            dropped += 1
        idxs = np.asarray(idxs, dtype=int)
        d2 = np.sum((coords[idxs] - pxy[i]) ** 2, axis=1)
        k = np.exp(-d2 / (2.0 * sigma ** 2))
        s = k.sum()
        if s <= 0:
            continue
        contrib = weights[i] * k / s
        demand[idxs] += contrib
        by_cat[cats[i]][idxs] += contrib

    if demand.sum() <= 0:
        return _topological_fallback(G, nf)

    scale = TOTAL_DEMAND_UNITS / demand.sum()
    out = nf.copy()
    out["demand"] = demand * scale
    for k, v in by_cat.items():
        out[f"demand_{k}"] = v * scale
    out["demand_share_pct"] = 100.0 * out["demand"] / out["demand"].sum()

    log.info("demand surface: %d POIs -> %d nodes (%d POIs snapped to nearest node)",
             len(scored), n, dropped)
    log.info("demand concentration: top 10%% of junctions hold %.1f%% of demand",
             100.0 * out["demand"].nlargest(max(1, n // 10)).sum() / out["demand"].sum())
    return out


def _topological_fallback(G: nx.Graph, nf: pd.DataFrame) -> pd.DataFrame:
    """If OSM POIs are unavailable, proxy demand by local road density.

    Weaker than real land use, but keeps the pipeline runnable offline.
    """
    log.warning("using topological demand fallback (no POI data)")
    from scipy.spatial import cKDTree

    tree = cKDTree(nf[["x", "y"]].values)
    counts = np.array([len(tree.query_ball_point(p, r=500.0))
                       for p in nf[["x", "y"]].values], dtype=float)
    deg = np.array([G.degree(n) for n in nf.index], dtype=float)
    raw = counts * (1.0 + 0.25 * deg)
    out = nf.copy()
    out["demand"] = raw * (TOTAL_DEMAND_UNITS / raw.sum())
    out["demand_share_pct"] = 100.0 * out["demand"] / out["demand"].sum()
    return out


def demand_summary(dem: pd.DataFrame) -> Dict:
    """Headline numbers plus a Gini coefficient of demand concentration."""
    d = np.sort(dem["demand"].values)
    n = len(d)
    cum = np.cumsum(d)
    gini = float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
    cat_cols = [c for c in dem.columns if c.startswith("demand_") and c != "demand_share_pct"]
    return {
        "n_nodes": int(n),
        "total_demand_units": float(dem["demand"].sum()),
        "mean_demand": float(dem["demand"].mean()),
        "median_demand": float(np.median(d)),
        "max_demand": float(d[-1]),
        "gini": gini,
        "top10pct_share": float(dem["demand"].nlargest(max(1, n // 10)).sum()
                                / dem["demand"].sum()),
        "by_category": {c.replace("demand_", ""): float(dem[c].sum()) for c in cat_cols},
    }
