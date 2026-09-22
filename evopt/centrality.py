"""Unit 4.4 - Centrality measures, the core siting signal.

Five measures, each answering a different question about where a charging
station should go:

  betweenness   how much through-traffic passes here      -> interception
  closeness     how far is everywhere else from here      -> average access
  degree        how many roads meet here                  -> redundant approach
  eigenvector   how well connected are my neighbours      -> strategic position
  harmonic      inverse mean distance, robust to sprawl   -> isolated pockets

Implementation note: closeness and harmonic both need all-pairs shortest path
distances, and betweenness needs them again. Rather than three separate
Python-level Dijkstra sweeps, the distance matrix is computed once through
SciPy's compiled `csgraph.dijkstra` and the measures are read off it. On this
study area that turns minutes into under a second.

Direction matters and is a modelling choice. The syllabus defines closeness
as "average distance from node to all others", so the outbound direction is
used; the inbound variant (how far drivers must travel *to* reach the site)
is computed alongside it, since that is what actually matters for a station.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .linalg import power_iteration

log = logging.getLogger("evopt.centrality")

MEASURES = ["betweenness", "closeness", "degree", "eigenvector", "harmonic"]


# --------------------------------------------------------------------------- #
def to_csgraph(G: nx.DiGraph, nodes: List, weight: str = "length") -> sp.csr_matrix:
    """Weight matrix in the exact form SciPy's csgraph wants.

    Two traps are handled here. SciPy requires 32-bit index arrays, and it
    reads a stored zero as "no edge" - so the zero-length edges that
    intersection consolidation can create would silently disconnect the
    graph. Lengths are floored at 10 cm instead.
    """
    index = {n: i for i, n in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for u, v, d in G.edges(data=True):
        if u in index and v in index:
            rows.append(index[u]); cols.append(index[v])
            vals.append(max(float(d.get(weight, 1.0)), 0.1))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=np.float64)
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)
    return A


def distance_matrix(G: nx.DiGraph, weight: str = "length",
                    nodes: Optional[List] = None) -> Tuple[np.ndarray, List]:
    """All-pairs shortest path distances, in metres. D[i, j] = i -> j."""
    from scipy.sparse.csgraph import dijkstra

    nodes = nodes or sorted(G.nodes())
    A = to_csgraph(G, nodes, weight)
    t0 = time.time()
    D = dijkstra(A, directed=True, unweighted=False)
    log.info("all-pairs distances: %d x %d in %.1fs", len(nodes), len(nodes), time.time() - t0)
    return D, nodes


# --------------------------------------------------------------------------- #
def compute_centralities(cfg, G: nx.DiGraph, U: Optional[nx.Graph] = None,
                         D: Optional[np.ndarray] = None,
                         nodes: Optional[List] = None) -> pd.DataFrame:
    """Checkpoint 4.4: all five measures on the road network."""
    from .network import undirected_view

    U = U if U is not None else undirected_view(G)
    if D is None:
        D, nodes = distance_matrix(G, weight="length")
    nodes = nodes or sorted(G.nodes())
    n = len(nodes)
    idx = pd.Index(nodes, name="node")

    # --- closeness and harmonic, straight off the distance matrix ---------
    finite = np.isfinite(D)
    np.fill_diagonal(finite, False)
    Dm = np.where(finite, D, np.nan)

    out_sum = np.nansum(Dm, axis=1)
    out_cnt = finite.sum(axis=1)
    closeness_out = np.where(out_sum > 0, out_cnt / out_sum, 0.0)

    in_sum = np.nansum(Dm, axis=0)
    in_cnt = finite.sum(axis=0)
    closeness_in = np.where(in_sum > 0, in_cnt / in_sum, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(finite, 1.0 / np.where(Dm > 0, Dm, np.nan), 0.0)
    harmonic = np.nansum(inv, axis=1) / max(n - 1, 1)

    # --- degree -----------------------------------------------------------
    degree = np.array([U.degree(v) for v in nodes], dtype=float)

    # --- eigenvector, via the Unit 1.2 power iteration --------------------
    A_und = nx.to_scipy_sparse_array(U, nodelist=nodes, weight=None, format="csr").astype(float)
    pi = power_iteration(A_und, tol=1e-12, max_iter=20000)
    eigenvector = np.abs(pi["eigenvector"])
    eigenvector = eigenvector / (eigenvector.max() or 1.0)
    log.info("eigenvector centrality: power iteration converged=%s in %d steps (lambda=%.4f)",
             pi["converged"], pi["iterations"], pi["eigenvalue"])

    # --- betweenness ------------------------------------------------------
    k = cfg.betweenness_k
    if k is None and n > cfg.exact_betweenness_max_nodes:
        k = min(500, n)
    t0 = time.time()
    bc = nx.betweenness_centrality(G, k=k, weight="length", normalized=True,
                                   seed=cfg.seed if k else None)
    log.info("betweenness (%s) in %.1fs", f"sampled k={k}" if k else "exact", time.time() - t0)
    betweenness = np.array([bc[v] for v in nodes], dtype=float)

    df = pd.DataFrame({
        "betweenness": betweenness,
        "closeness": closeness_out,
        "closeness_inbound": closeness_in,
        "degree": degree,
        "eigenvector": eigenvector,
        "harmonic": harmonic,
        "mean_distance_out_m": np.where(out_cnt > 0, out_sum / out_cnt, np.inf),
        "max_distance_out_m": np.nanmax(Dm, axis=1),
        "reachable_nodes": out_cnt.astype(int),
    }, index=idx)
    df["eigenvalue_lambda1"] = pi["eigenvalue"]
    return df


def validate_eigenvector(cent: pd.DataFrame, U: nx.Graph) -> Dict:
    """Cross-check our power iteration against NetworkX's eigensolver."""
    nodes = list(cent.index)
    try:
        ref = nx.eigenvector_centrality_numpy(U, weight=None)
    except Exception as exc:  # pragma: no cover
        return {"validated": False, "reason": str(exc)}
    r = np.array([ref[v] for v in nodes], dtype=float)
    r = np.abs(r) / (np.abs(r).max() or 1.0)
    m = cent["eigenvector"].values
    cos = float(m @ r / ((np.linalg.norm(m) * np.linalg.norm(r)) or 1.0))
    return {
        "validated": bool(cos > 0.9999),
        "cosine_similarity": cos,
        "max_abs_difference": float(np.abs(m - r).max()),
        "spearman": float(pd.Series(m).corr(pd.Series(r), method="spearman")),
    }


# --------------------------------------------------------------------------- #
def normalise(cent: pd.DataFrame, method: str = "percentile") -> pd.DataFrame:
    """Put the five measures on a common 0-1 scale.

    Percentile ranking is the default because centrality distributions on road
    networks are extremely heavy-tailed - a handful of junctions carry
    betweenness orders of magnitude above the median - and min-max scaling
    would let those few points dominate the combined score entirely.
    """
    out = pd.DataFrame(index=cent.index)
    for m in MEASURES:
        v = cent[m]
        if method == "percentile":
            out[m] = v.rank(pct=True)
        elif method == "minmax":
            rng = v.max() - v.min()
            out[m] = (v - v.min()) / rng if rng > 0 else 0.0
        elif method == "zscore":
            sd = v.std()
            out[m] = (v - v.mean()) / sd if sd > 0 else 0.0
        else:
            raise ValueError(f"unknown normalisation: {method}")
    return out


def combined_score(cfg, cent: pd.DataFrame, weights: Optional[Dict[str, float]] = None,
                   method: str = "percentile") -> pd.DataFrame:
    """Checkpoint 4.4: one priority score per junction, 0-100.

    The weights encode a planning stance: betweenness leads because a charger
    on a corridor people already drive costs the driver almost no detour,
    harmonic and closeness follow because they measure access, and degree
    matters least on its own - a six-way junction in a quiet colony is not a
    good site just for being a six-way junction.
    """
    if weights is None:
        weights = dict(zip(MEASURES, cfg.centrality_weights))
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    norm = normalise(cent, method=method)
    score = sum(norm[m] * w for m, w in weights.items()) * 100.0

    out = cent.copy()
    for m in MEASURES:
        out[f"{m}_norm"] = norm[m]
    out["priority_score"] = score
    out["priority_rank"] = score.rank(ascending=False, method="min").astype(int)
    out.attrs["weights"] = weights
    out.attrs["normalisation"] = method
    return out


def centrality_correlations(cent: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """How much the five measures actually agree - input to Unit 1.4."""
    return cent[MEASURES].corr(method=method)


def centrality_summary(cent: pd.DataFrame, top_n: int = 10) -> Dict:
    """Headline numbers plus the leading junction for each measure."""
    corr = centrality_correlations(cent)
    off = corr.where(~np.eye(len(corr), dtype=bool))
    return {
        "n_nodes": int(len(cent)),
        "top_by_measure": {
            m: [int(i) if isinstance(i, (int, np.integer)) else str(i)
                for i in cent[m].nlargest(3).index.tolist()]
            for m in MEASURES
        },
        "mean_spearman_between_measures": float(off.stack().mean()),
        "max_spearman_pair": {
            "pair": list(off.stack().idxmax()),
            "rho": float(off.stack().max()),
        },
        "min_spearman_pair": {
            "pair": list(off.stack().idxmin()),
            "rho": float(off.stack().min()),
        },
        "betweenness_top10pct_share": float(
            cent["betweenness"].nlargest(max(1, len(cent) // 10)).sum()
            / max(cent["betweenness"].sum(), 1e-12)),
        "lambda1_adjacency": float(cent["eigenvalue_lambda1"].iloc[0]),
    }
