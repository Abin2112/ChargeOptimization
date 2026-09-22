"""Units 4.2, 4.5, 4.6 - community structure, small-world tests, hubs and cores.

Why an EV siting study cares about community structure: a road network breaks
into neighbourhoods that are well connected internally and joined to the rest
of the city by a few corridors. Two stations inside one such neighbourhood
compete for the same drivers, while a neighbourhood with none is stranded no
matter how good the city-wide coverage number looks. Communities therefore
become an *equity constraint* on the optimiser, not just a description.

Contents
--------
4.2/4.5  louvain_communities, community_profile, small_world_analysis
4.6      core_decomposition, identify_hubs
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger("evopt.communities")


# --------------------------------------------------------------------------- #
# 4.2 / 4.5  Communities
# --------------------------------------------------------------------------- #
def detect_communities(cfg, U: nx.Graph, resolution: float = 1.0) -> pd.Series:
    """Louvain modularity maximisation on the undirected road graph.

    Edges are weighted by `affinity` (inverse road length), so two junctions
    joined by a short street are treated as strongly tied and one joined by a
    long arterial link weakly - which is what actually makes a neighbourhood.
    """
    t0 = time.time()
    parts = nx.community.louvain_communities(
        U, weight="affinity", resolution=resolution, seed=cfg.seed)
    parts = sorted(parts, key=len, reverse=True)
    label = {}
    for cid, members in enumerate(parts):
        for v in members:
            label[v] = cid
    log.info("louvain: %d communities in %.1fs (largest %d nodes)",
             len(parts), time.time() - t0, len(parts[0]))
    return pd.Series(label, name="community").sort_index()


def modularity(U: nx.Graph, comm: pd.Series, weight: str = "affinity") -> float:
    groups = [set(comm.index[comm == c]) for c in sorted(comm.unique())]
    return float(nx.community.modularity(U, groups, weight=weight))


def community_profile(comm: pd.Series, dem: pd.DataFrame, cent: pd.DataFrame,
                      station_nodes: Optional[List] = None) -> pd.DataFrame:
    """Per-community demand, centrality and current charger provision."""
    station_nodes = set(station_nodes or [])
    rows = []
    for c in sorted(comm.unique()):
        members = comm.index[comm == c]
        d = dem["demand"].reindex(members).fillna(0.0)
        rows.append({
            "community": int(c),
            "n_nodes": int(len(members)),
            "demand_units": float(d.sum()),
            "demand_share_pct": float(100.0 * d.sum() / dem["demand"].sum()),
            "mean_priority": float(cent["priority_score"].reindex(members).mean())
            if "priority_score" in cent else np.nan,
            "mean_betweenness": float(cent["betweenness"].reindex(members).mean()),
            "existing_stations": int(sum(1 for v in members if v in station_nodes)),
            "centroid_lat": float(dem["lat"].reindex(members).mean()),
            "centroid_lon": float(dem["lon"].reindex(members).mean()),
        })
    df = pd.DataFrame(rows).set_index("community")
    df["demand_per_station"] = np.where(
        df["existing_stations"] > 0,
        df["demand_units"] / df["existing_stations"].replace(0, np.nan),
        np.inf)
    return df.sort_values("demand_units", ascending=False)


# --------------------------------------------------------------------------- #
def small_world_analysis(U: nx.Graph, n_random: int = 3, seed: int = 42,
                         sample_nodes: int = 1200) -> Dict:
    """Is the road network a small world? (Unit 4.5)

    Compares clustering C and characteristic path length L against an
    Erdos-Renyi graph of the same size and a ring lattice of the same average
    degree, via the two standard indices

        sigma = (C/C_rand) / (L/L_rand)      small-world if sigma >> 1
        omega = L_rand/L - C/C_latt          small-world if omega ~ 0,
                                             -1 = lattice, +1 = random

    The verdict is taken from omega, not sigma, and the reason is worth
    stating. On a sparse graph an Erdos-Renyi null model has almost no
    triangles at all, so C_rand collapses towards zero and the ratio C/C_rand
    explodes: sigma will declare almost any clustered sparse network a small
    world regardless of its path lengths. Omega is bounded and compares
    against both null models, so it does not have that failure mode.
    Both are reported so the gap between them is visible.
    """
    rng = np.random.default_rng(seed)
    n, m = U.number_of_nodes(), U.number_of_edges()
    C = float(nx.average_clustering(U))
    L = _mean_path_length(U, sample_nodes, rng)

    Cr, Lr = [], []
    for i in range(n_random):
        R = nx.gnm_random_graph(n, m, seed=int(rng.integers(1e6)))
        if not nx.is_connected(R):
            R = R.subgraph(max(nx.connected_components(R), key=len)).copy()
        Cr.append(nx.average_clustering(R))
        Lr.append(_mean_path_length(R, sample_nodes, rng))
    C_rand, L_rand = float(np.mean(Cr)), float(np.mean(Lr))

    k = max(int(round(2 * m / n)), 2)
    C_latt = float(3 * (k - 2) / (4 * (k - 1))) if k > 2 else 0.75

    sigma = ((C / C_rand) / (L / L_rand)) if C_rand > 0 and L_rand > 0 and L > 0 else np.nan
    omega = (L_rand / L - C / C_latt) if L > 0 and C_latt > 0 else np.nan

    if omega <= -0.25:
        regime = "lattice-like"
        reading = ("paths are long relative to a random graph and clustering is "
                   "high - the signature of a spatially embedded network, where "
                   "travel distance scales with geographic distance")
    elif omega >= 0.25:
        regime = "random-like"
        reading = "paths and clustering both resemble a random graph"
    else:
        regime = "small-world"
        reading = ("clustering far above a random graph while paths stay far "
                   "below a lattice - consistent with arterial roads acting as "
                   "long-range shortcuts across an otherwise local street grid")

    return {
        "clustering_actual": C,
        "clustering_random": C_rand,
        "clustering_lattice": C_latt,
        "path_length_actual": L,
        "path_length_random": L_rand,
        "path_length_ratio_vs_random": float(L / L_rand) if L_rand > 0 else np.nan,
        "clustering_ratio_vs_random": float(C / C_rand) if C_rand > 0 else np.nan,
        "mean_degree": float(2 * m / n),
        "sigma": float(sigma),
        "omega": float(omega),
        "regime": regime,
        "is_small_world": bool(abs(omega) < 0.25),
        "verdict_index": "omega",
        "interpretation": reading,
        "sigma_caveat": (
            "sigma is inflated on sparse graphs because the Erdos-Renyi null "
            "model has near-zero clustering (C_rand = %.2e here), so the "
            "verdict above is taken from omega." % C_rand),
    }


def _mean_path_length(G: nx.Graph, sample_nodes: int, rng) -> float:
    """Characteristic path length (hops), sampled for tractability."""
    from scipy.sparse.csgraph import shortest_path

    import scipy.sparse as sp

    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols = [], []
    for u, v in G.edges():
        rows += [idx[u], idx[v]]
        cols += [idx[v], idx[u]]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes),) * 2)
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)

    k = min(sample_nodes, len(nodes))
    src = rng.choice(len(nodes), size=k, replace=False)
    D = shortest_path(A, method="D", unweighted=True, directed=False, indices=src)
    finite = np.isfinite(D) & (D > 0)
    return float(D[finite].mean()) if finite.any() else np.inf


# --------------------------------------------------------------------------- #
# 4.6  Cores and hubs
# --------------------------------------------------------------------------- #
def core_decomposition(U: nx.Graph) -> pd.Series:
    """k-core number of every node (Unit 4.6).

    A node's core number is the largest k for which it survives repeated
    removal of all nodes of degree < k. The maximum core is the structural
    backbone of the network: the part that stays connected when the
    peripheral tree-like streets are peeled away.
    """
    G = nx.Graph(U)
    G.remove_edges_from(nx.selfloop_edges(G))
    return pd.Series(nx.core_number(G), name="core_number").sort_index()


def identify_hubs(cent: pd.DataFrame, core: pd.Series, dem: pd.DataFrame,
                  top_n: int = 25) -> pd.DataFrame:
    """Junctions that are simultaneously central, deep in the core and busy.

    A hub is not just a high-degree node. The score multiplies three
    independent signals - through-traffic (betweenness), structural depth
    (core number) and local demand - so a junction has to earn all three.
    """
    df = cent.copy()
    df["core_number"] = core.reindex(df.index).fillna(0).astype(int)
    df["demand"] = dem["demand"].reindex(df.index).fillna(0.0)

    def pct(s):
        return s.rank(pct=True)

    df["hub_score"] = 100.0 * (
        pct(df["betweenness"]) ** 0.4 *
        pct(df["core_number"]) ** 0.3 *
        pct(df["demand"]) ** 0.3
    )
    df["is_max_core"] = df["core_number"] == df["core_number"].max()
    return df.nlargest(top_n, "hub_score")


def community_summary(comm: pd.Series, U: nx.Graph, profile: pd.DataFrame) -> Dict:
    sizes = comm.value_counts()
    underserved = profile[profile["existing_stations"] == 0]
    return {
        "n_communities": int(comm.nunique()),
        "modularity": modularity(U, comm),
        "largest_community_nodes": int(sizes.max()),
        "smallest_community_nodes": int(sizes.min()),
        "median_community_nodes": float(sizes.median()),
        "communities_without_a_station": int(len(underserved)),
        "demand_in_unserved_communities_pct": float(underserved["demand_share_pct"].sum()),
    }
