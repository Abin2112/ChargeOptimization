"""Unit 4.3 - Shortest paths and coverage. The project's primary objective.

Coverage is measured along the *road network*, never as a straight line. In a
city cut by railway lines, creeks and one-way systems, two points 800 m apart
on a map can be a 3 km drive, and a straight-line coverage figure would
quietly overstate how well served people are.

The key metric is demand-weighted: the share of charging demand whose nearest
station is within `coverage_radius_m` of network driving distance. Counting
junctions instead of demand would let a station in an empty industrial estate
look as valuable as one serving a dense residential block.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger("evopt.coverage")


# --------------------------------------------------------------------------- #
def distance_to_nearest(D: np.ndarray, nodes: Sequence, station_nodes: Sequence
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Driving distance from every junction to its nearest station.

    D[i, j] is the distance from junction i to junction j, so taking the
    minimum across the station columns gives the *outbound* drive a user
    makes to reach a charger - the direction that matters, and not the same
    as the inbound distance on a one-way network.

    Returns (distance, index-of-serving-station-within-station_nodes).
    """
    if len(station_nodes) == 0:
        return np.full(len(nodes), np.inf), np.full(len(nodes), -1, dtype=int)

    pos = {n: i for i, n in enumerate(nodes)}
    cols = [pos[s] for s in station_nodes if s in pos]
    if not cols:
        return np.full(len(nodes), np.inf), np.full(len(nodes), -1, dtype=int)

    sub = D[:, cols]
    nearest = sub.min(axis=1)
    which = np.asarray(cols)[sub.argmin(axis=1)]
    back = {c: i for i, c in enumerate(cols)}
    which_local = np.array([back[c] for c in which], dtype=int)
    return nearest, which_local


def multi_source_distance(G: nx.DiGraph, nodes: Sequence, station_nodes: Sequence,
                          weight: str = "length", cutoff: Optional[float] = None
                          ) -> np.ndarray:
    """Same quantity without a full distance matrix, for large networks.

    Runs one Dijkstra sweep on the reversed graph seeded from every station at
    once, which is O(m log n) in total rather than once per station.
    """
    from scipy.sparse.csgraph import dijkstra

    from .centrality import to_csgraph

    pos = {n: i for i, n in enumerate(nodes)}
    A = to_csgraph(G, list(nodes), weight)
    src = [pos[s] for s in station_nodes if s in pos]
    if not src:
        return np.full(len(nodes), np.inf)
    # Reversed graph: distance station -> i in reverse equals i -> station.
    D = dijkstra(A.T.tocsr(), directed=True, indices=src, limit=cutoff or np.inf)
    return D.min(axis=0)


# --------------------------------------------------------------------------- #
def coverage_metrics(dist: np.ndarray, demand: np.ndarray, radius: float,
                     label: str = "") -> Dict:
    """Checkpoint 4.3: the coverage scorecard."""
    dist = np.asarray(dist, dtype=float)
    demand = np.asarray(demand, dtype=float)
    total = demand.sum()
    covered = dist <= radius
    finite = np.isfinite(dist)

    w = demand / total if total > 0 else np.zeros_like(demand)
    order = np.argsort(dist)
    cum = np.cumsum(w[order])
    med_i = int(np.searchsorted(cum, 0.50))
    p90_i = int(np.searchsorted(cum, 0.90))
    med_i = min(med_i, len(order) - 1)
    p90_i = min(p90_i, len(order) - 1)

    d_fin = dist[finite]
    w_fin = demand[finite]
    mean_w = float((d_fin * w_fin).sum() / w_fin.sum()) if w_fin.sum() > 0 else np.inf

    # Gini of access distance: 0 = everyone travels the same, 1 = extreme
    # inequality. Equity matters politically as much as the average does.
    ds = np.sort(d_fin)
    gini = np.nan
    if len(ds) and ds.sum() > 0:
        n = len(ds)
        gini = float((2 * np.arange(1, n + 1) - n - 1) @ ds / (n * ds.sum()))

    return {
        "label": label,
        "radius_m": float(radius),
        "demand_covered_pct": float(100.0 * demand[covered].sum() / total) if total else 0.0,
        "nodes_covered_pct": float(100.0 * covered.mean()),
        "mean_distance_m": mean_w,
        "median_distance_m": float(dist[order][med_i]),
        "p90_distance_m": float(dist[order][p90_i]),
        "max_distance_m": float(d_fin.max()) if len(d_fin) else np.inf,
        "unreachable_nodes": int((~finite).sum()),
        "uncovered_demand_units": float(demand[~covered].sum()),
        "access_gini": gini,
    }


def coverage_curve(dist: np.ndarray, demand: np.ndarray,
                   radii: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Covered demand as the acceptable travel distance is relaxed.

    More informative than a single threshold: it shows whether a network is
    "nearly there" for most users or genuinely stranding them.
    """
    radii = radii if radii is not None else np.arange(250, 6001, 250)
    total = demand.sum()
    rows = []
    for r in radii:
        c = dist <= r
        rows.append({
            "radius_m": float(r),
            "demand_covered_pct": float(100.0 * demand[c].sum() / total) if total else 0.0,
            "nodes_covered_pct": float(100.0 * c.mean()),
        })
    return pd.DataFrame(rows)


def coverage_by_community(dist: np.ndarray, demand: np.ndarray, nodes: Sequence,
                          comm: pd.Series, radius: float) -> pd.DataFrame:
    """Per-neighbourhood coverage - where a good city-wide average hides gaps."""
    df = pd.DataFrame({"node": list(nodes), "dist": dist, "demand": demand})
    df["community"] = comm.reindex(df["node"]).values
    df["covered"] = df["dist"] <= radius
    g = df.groupby("community", dropna=True)
    out = pd.DataFrame({
        "n_nodes": g.size(),
        "demand_units": g["demand"].sum(),
        "demand_covered_pct": 100.0 * g.apply(
            lambda x: x.loc[x["covered"], "demand"].sum() / max(x["demand"].sum(), 1e-12)),
        "mean_distance_m": g.apply(
            lambda x: float((x["dist"] * x["demand"]).sum() / max(x["demand"].sum(), 1e-12))),
        "max_distance_m": g["dist"].max(),
    })
    return out.sort_values("demand_covered_pct")


def station_load(dist: np.ndarray, which: np.ndarray, demand: np.ndarray,
                 station_nodes: Sequence, radius: float, capacity: float) -> pd.DataFrame:
    """Demand assigned to each station, against its capacity.

    Assignment is nearest-station within the coverage radius, which is how
    drivers actually behave. Utilisation above 100% marks a station that will
    queue - the signal that a site needs more bays rather than a new site.
    """
    rows = []
    for k, s in enumerate(station_nodes):
        served = (which == k) & (dist <= radius)
        load = float(demand[served].sum())
        rows.append({
            "station_node": s,
            "assigned_nodes": int(served.sum()),
            "assigned_demand": load,
            "capacity": capacity,
            "utilisation_pct": float(100.0 * load / capacity) if capacity > 0 else np.inf,
            "mean_user_distance_m": float(demand[served] @ dist[served] / load) if load > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def compare_scenarios(scenarios: Dict[str, Dict]) -> pd.DataFrame:
    """Side-by-side scorecard for before/after and variant comparisons."""
    return pd.DataFrame(scenarios).T[[
        "demand_covered_pct", "nodes_covered_pct", "mean_distance_m",
        "median_distance_m", "p90_distance_m", "max_distance_m", "access_gini",
    ]]


def capacity_check(load: pd.DataFrame, covered_demand: float, capacity: float,
                   n_stations: int) -> Dict:
    """Which constraint actually binds - coverage or capacity?

    Reaching every driver within the service radius and being able to serve
    them once they arrive are different problems. A plan can pass the first
    and fail the second, and the fix is different in each case: more sites
    for a coverage gap, more bays per site for a capacity gap.
    """
    needed = int(np.ceil(covered_demand / capacity)) if capacity > 0 else 0
    over = load[load["utilisation_pct"] > 100.0]
    return {
        "covered_demand_units": float(covered_demand),
        "capacity_per_station": float(capacity),
        "stations_built": int(n_stations),
        "stations_required_by_capacity": needed,
        "binding_constraint": "capacity" if needed > n_stations else "coverage",
        "stations_over_capacity": int(len(over)),
        "mean_utilisation_pct": float(load["utilisation_pct"].replace(
            [np.inf, -np.inf], np.nan).mean()),
        "max_utilisation_pct": float(load["utilisation_pct"].replace(
            [np.inf, -np.inf], np.nan).max()),
        "recommendation": (
            f"coverage is met by {n_stations} sites, but serving the covered "
            f"demand needs about {needed} sites at this capacity - add bays to "
            f"the busiest sites rather than new locations"
            if needed > n_stations else
            f"{n_stations} sites satisfy both the coverage standard and the "
            f"capacity assumption"),
    }
