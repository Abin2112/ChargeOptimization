"""Bonus analyses - spectral methods, perturbation theory, time and resilience.

Four questions the core pipeline does not answer:

  spectral_analysis      how tightly is the network held together, and where
                         would it split if you cut it?               (Laplacian)
  perturbation_analysis  if a major road closes, how much does the centrality
                         structure move?              (matrix perturbation theory)
  temporal_optimisation  do the best sites change between the morning peak and
                         midnight?                        (temporal networks)
  resilience_analysis    what happens to coverage when stations fail?
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("evopt.advanced")


# =========================================================================== #
# Spectral analysis of the Laplacian
# =========================================================================== #
def spectral_analysis(U: nx.Graph, k: int = 8, weight: Optional[str] = None) -> Dict:
    """Laplacian eigenvalues, algebraic connectivity and spectral bisection.

    L = D - A is positive semi-definite, so 0 = lambda_1 <= lambda_2 <= ...
    The multiplicity of the zero eigenvalue counts connected components, and
    lambda_2 - the algebraic connectivity - measures how strongly the network
    resists being cut in two. Its eigenvector, the Fiedler vector, says *where*
    that cut would fall: sorting nodes by their Fiedler value and splitting at
    zero gives a near-minimum bisection.

    For charger siting this is a warning system. A small lambda_2 means the
    area is nearly two separate districts joined by a thin bridge, and a plan
    that treats it as one pool of demand will strand whichever side loses.
    """
    from scipy.sparse.linalg import eigsh

    nodes = sorted(U.nodes())
    L = nx.laplacian_matrix(U, nodelist=nodes, weight=weight).astype(float).tocsc()
    n = L.shape[0]
    k = int(min(k, n - 2))

    try:
        # Shift-invert around zero converges far faster on the small end.
        vals, vecs = eigsh(L, k=k, sigma=1e-8, which="LM")
    except Exception:
        vals, vecs = eigsh(L, k=k, which="SM", maxiter=5000)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    fiedler = vecs[:, 1]
    side_a = [nodes[i] for i in np.where(fiedler <= 0)[0]]
    side_b = [nodes[i] for i in np.where(fiedler > 0)[0]]

    cut = sum(1 for u, v in U.edges()
              if (u in set(side_a)) != (v in set(side_a)))
    vol = min(len(side_a), len(side_b))
    conductance = float(cut / max(vol, 1))

    return {
        "eigenvalues": vals.tolist(),
        "lambda_1": float(vals[0]),
        "algebraic_connectivity_lambda2": float(vals[1]),
        "spectral_gap_lambda3_minus_lambda2": float(vals[2] - vals[1]),
        "n_zero_eigenvalues": int(np.sum(vals < 1e-9)),
        "fiedler_vector": pd.Series(fiedler, index=nodes, name="fiedler"),
        "bisection_sizes": [len(side_a), len(side_b)],
        "bisection_cut_edges": int(cut),
        "bisection_conductance": conductance,
        "interpretation": (
            f"lambda_2 = {vals[1]:.4f}: the network is "
            + ("tightly connected; no single corridor controls it"
               if vals[1] > 0.05 else
               "weakly connected - it is close to splitting into separate "
               "districts, so coverage must be checked on each side of the cut")
        ),
    }


# =========================================================================== #
# Matrix perturbation theory
# =========================================================================== #
def perturbation_analysis(U: nx.Graph, edges: Sequence[Tuple], scale: float = 1.0,
                          n_modes: int = 12) -> Dict:
    """Predict the centrality shift from a road capacity change, then check it.

    For a symmetric A with a simple dominant eigenpair (lambda_1, x_1), standard
    first-order perturbation theory gives, for a perturbation dA,

        d_lambda_1  ~  x_1' dA x_1            (x_1 normalised)
        d_x_1       ~  sum_{k>1}  (v_k' dA x_1) / (lambda_1 - lambda_k) * v_k

    Both come from the *unperturbed* spectrum alone, so screening a what-if
    costs one matrix-vector product rather than a full re-solve. The exact
    answer is recomputed anyway and the two compared, because a prediction
    nobody checks is not worth making.

    `scale` sets the severity: 1.0 removes the road entirely, 0.1 models a
    capacity cut of about a tenth, as a lane closure would.

    Level crossings are detected and reported. If the perturbation pushes the
    original dominant mode below another one, the "dominant eigenvector" after
    the change is a *different* vector, not a shifted version of the original,
    and first-order theory does not apply at all - the denominator
    lambda_1 - lambda_k it divides by has passed through zero. On a network
    with near-degenerate localised modes that happens readily, so the check is
    not academic.
    """
    from scipy.sparse.linalg import eigsh

    nodes = sorted(U.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(U, nodelist=nodes, weight=None, format="csr").astype(float)

    k = int(min(n_modes, A.shape[0] - 2))
    vals, vecs = eigsh(A, k=k, which="LA")
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    lam1, x1 = float(vals[0]), vecs[:, 0]
    if x1.sum() < 0:
        x1, vecs = -x1, -vecs

    # dA: weaken (or remove) each undirected edge, symmetrically.
    rows, cols, data = [], [], []
    applied = 0
    for u, v in edges:
        if u in idx and v in idx and U.has_edge(u, v):
            rows += [idx[u], idx[v]]
            cols += [idx[v], idx[u]]
            data += [-float(scale), -float(scale)]
            applied += 1
    dA = sp.csr_matrix((data, (rows, cols)), shape=A.shape)

    dlam_pred = float(x1 @ (dA @ x1))

    dx = np.zeros_like(x1)
    for j in range(1, k):
        denom = lam1 - vals[j]
        if abs(denom) < 1e-12:
            continue
        dx += (vecs[:, j] @ (dA @ x1)) / denom * vecs[:, j]

    # Exact recomputation for comparison.
    A2 = (A + dA).tocsr()
    vals2, vecs2 = eigsh(A2, k=min(3, A.shape[0] - 2), which="LA")
    o2 = np.argsort(vals2)[::-1]
    lam1_new = float(vals2[o2][0])
    x1_new = vecs2[:, o2][:, 0]
    if x1_new.sum() < 0:
        x1_new = -x1_new

    dlam_true = lam1_new - lam1
    x1_pred = x1 + dx
    x1_pred /= np.linalg.norm(x1_pred) or 1.0

    cos_pred = float(abs(x1_pred @ x1_new))
    cos_naive = float(abs(x1 @ x1_new))
    crossed = bool(cos_naive < 0.5)

    shift = pd.Series(np.abs(x1_new) - np.abs(x1), index=nodes, name="centrality_shift")
    return {
        "n_edges_perturbed": applied,
        "scale": float(scale),
        "lambda1_before": lam1,
        "lambda1_after_exact": lam1_new,
        "delta_lambda_predicted": dlam_pred,
        "delta_lambda_exact": dlam_true,
        "delta_lambda_abs_error": float(abs(dlam_pred - dlam_true)),
        "delta_lambda_rel_error": float(abs(dlam_pred - dlam_true) / abs(dlam_true))
        if abs(dlam_true) > 1e-12 else np.nan,
        "eigenvector_cosine_first_order": cos_pred,
        "eigenvector_cosine_no_correction": cos_naive,
        "first_order_improves_alignment": bool(cos_pred > cos_naive),
        "level_crossing": crossed,
        "theory_applicable": not crossed,
        "note": ("the dominant mode was overtaken by a different one - a level "
                 "crossing, where first-order theory does not apply"
                 if crossed else "single dominant mode preserved; theory applies"),
        "biggest_gainers": shift.nlargest(5).to_dict(),
        "biggest_losers": shift.nsmallest(5).to_dict(),
    }


def top_betweenness_edges(U: nx.Graph, cent: pd.DataFrame, n: int = 3) -> List[Tuple]:
    """The busiest links - the closure a traffic engineer would actually worry about."""
    edges = []
    for v in cent["betweenness"].nlargest(n * 5).index:
        if v not in U:
            continue
        for w in list(U.neighbors(v))[:1]:
            edges.append((v, w))
        if len(edges) >= n:
            break
    return edges[:n]


def top_eigenvector_edges(U: nx.Graph, cent: pd.DataFrame, n: int = 3) -> List[Tuple]:
    """Links where the dominant eigenvector actually carries weight.

    Needed because on this network the two are not the same place at all -
    see `eigenvector_localisation`. Perturbing the spectrum requires
    perturbing where the spectrum lives.
    """
    edges = []
    for v in cent["eigenvector"].nlargest(n * 5).index:
        if v not in U:
            continue
        nbrs = sorted(U.neighbors(v), key=lambda w: -cent["eigenvector"].get(w, 0.0))
        for w in nbrs[:1]:
            if (w, v) not in edges:
                edges.append((v, w))
        if len(edges) >= n:
            break
    return edges[:n]


def eigenvector_localisation(cent: pd.DataFrame, column: str = "eigenvector") -> Dict:
    """How concentrated is the dominant eigenvector? (inverse participation ratio)

        IPR = sum(x_i^4) / (sum(x_i^2))^2,     effective support = 1 / IPR

    A vector spread evenly over N nodes has IPR = 1/N and effective support N.
    A vector living on a handful of nodes has effective support of that order,
    regardless of how large the network is.

    This matters for the project, not just for the mathematics. If the
    dominant eigenvector is localised on one dense cluster, then eigenvector
    centrality is numerically meaningless for the rest of the network - the
    values there are at the level of floating-point noise, so their *ranking*
    is arbitrary. That is the justification for giving eigenvector centrality
    a modest weight in the combined siting score rather than trusting it.
    """
    x = np.abs(cent[column].values.astype(float))
    x = x / (np.linalg.norm(x) or 1.0)
    ipr = float(np.sum(x ** 4) / (np.sum(x ** 2) ** 2))
    n = len(x)
    eff = float(1.0 / ipr) if ipr > 0 else np.nan
    return {
        "n_nodes": int(n),
        "inverse_participation_ratio": ipr,
        "effective_support_nodes": eff,
        "effective_support_share": float(eff / n),
        "share_above_1e_3": float((cent[column] > 1e-3).mean()),
        "median_value": float(cent[column].median()),
        "is_localised": bool(eff < 0.05 * n),
        "interpretation": (
            f"the dominant eigenvector effectively lives on {eff:.0f} of {n} "
            f"junctions"
            + ("; eigenvector centrality is therefore informative only on that "
               "cluster and should not be read as a ranking elsewhere"
               if eff < 0.05 * n else "; it is well spread across the network")),
    }


def perturbation_sweep(U: nx.Graph, edges: Sequence[Tuple],
                       scales: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
                       n_modes: int = 12) -> pd.DataFrame:
    """Accuracy of the first-order prediction as the perturbation gets harsher.

    First-order theory is exact in the limit of a vanishing perturbation, so
    the question is never "is it right" but "how far can it be pushed before
    it stops being right". Sweeping the severity traces that boundary, and the
    error should grow roughly quadratically until a level crossing ends the
    theory's validity outright.
    """
    rows = []
    for sc in scales:
        r = perturbation_analysis(U, edges, scale=float(sc), n_modes=n_modes)
        rows.append({
            "scale": float(sc),
            "delta_lambda_predicted": r["delta_lambda_predicted"],
            "delta_lambda_exact": r["delta_lambda_exact"],
            "rel_error_pct": 100.0 * r["delta_lambda_rel_error"]
            if np.isfinite(r["delta_lambda_rel_error"]) else np.nan,
            "cosine_first_order": r["eigenvector_cosine_first_order"],
            "cosine_no_correction": r["eigenvector_cosine_no_correction"],
            "level_crossing": r["level_crossing"],
        })
    return pd.DataFrame(rows)


# =========================================================================== #
# Temporal networks
# =========================================================================== #
WINDOWS = {
    "morning_peak": list(range(7, 11)),
    "midday": list(range(11, 16)),
    "evening_peak": list(range(17, 22)),
    "night": [22, 23, 0, 1, 2, 3, 4, 5],
}


def temporal_activity(cfg, panel: Dict, svd: Dict, nodes: Sequence,
                      dem: pd.DataFrame) -> pd.DataFrame:
    """Per-node traffic activity in each time window, from the denoised panel.

    Sensors cover only a subset of junctions, so window activity is carried to
    the rest by inverse-distance interpolation over the sensor cloud. That is
    stated rather than hidden: the interpolation is the weakest link in the
    temporal chain, and a real deployment would put out more sensors instead.
    """
    from scipy.spatial import cKDTree

    X = svd["X_denoised"]
    hours = np.array([t.hour for t in panel["time_index"]])
    sensors = panel["sensor_nodes"]

    act = {}
    for name, hrs in WINDOWS.items():
        sel = np.isin(hours, hrs)
        act[name] = X[:, sel].mean(axis=1)
    S = pd.DataFrame(act, index=sensors)

    sxy = dem.reindex(sensors)[["x", "y"]].values
    axy = dem.reindex(nodes)[["x", "y"]].values
    tree = cKDTree(sxy)
    d, i = tree.query(axy, k=min(6, len(sensors)))
    d = np.maximum(d, 1.0)
    w = 1.0 / d
    w /= w.sum(axis=1, keepdims=True)

    out = {}
    for name in WINDOWS:
        v = S[name].values
        out[name] = (v[i] * w).sum(axis=1)
    A = pd.DataFrame(out, index=pd.Index(nodes, name="node"))
    return A.div(A.mean(axis=0), axis=1)          # 1.0 = an average hour


def temporal_optimisation(cfg, D: np.ndarray, nodes: Sequence, dem: pd.DataFrame,
                          activity: pd.DataFrame, candidates: Sequence,
                          precovered: np.ndarray, p: Optional[int] = None) -> Dict:
    """Re-solve the siting problem separately for each time window.

    Demand is rescaled by that window's traffic activity, so the morning peak
    problem genuinely weights commuter corridors more heavily. Sites chosen in
    every window are the robust core of the programme; sites chosen in only
    one are the ones worth a second look.
    """
    from .optimise import build_coverage_matrix, greedy_mclp

    p = p or cfg.n_new_stations
    A = build_coverage_matrix(D, nodes, candidates, cfg.coverage_radius_m)
    base = dem["demand"].reindex(nodes).fillna(0.0).values

    picks, rows = {}, []
    for name in cfg.temporal_windows:
        if name not in activity.columns:
            continue
        w = activity[name].reindex(nodes).fillna(1.0).values
        d_t = base * w
        r = greedy_mclp(A, d_t, p, precovered)
        picks[name] = set(candidates[j] for j in r["selected"])
        rows.append({
            "window": name,
            "demand_covered_pct": r["objective_pct"],
            # Activity is normalised to mean 1.0 per window, so the mean says
            # nothing; the spread and the hot tail are what differ.
            "activity_spread_sd": float(w.std()),
            "top_decile_activity": float(np.percentile(w, 90)),
            "demand_shift_pct": float(100.0 * (d_t.sum() / base.sum() - 1.0)),
        })

    names = list(picks)
    jac = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            inter = len(picks[a] & picks[b])
            union = len(picks[a] | picks[b])
            jac.loc[a, b] = inter / union if union else np.nan

    always = set.intersection(*picks.values()) if picks else set()
    ever = set.union(*picks.values()) if picks else set()

    return {
        "per_window": pd.DataFrame(rows),
        "selections": {k: sorted(v) for k, v in picks.items()},
        "jaccard": jac,
        "robust_sites": sorted(always),
        "n_robust_sites": len(always),
        "n_sites_ever_chosen": len(ever),
        "stability": float(len(always) / len(ever)) if ever else np.nan,
        "interpretation": (
            f"{len(always)} of {len(ever)} sites are optimal in every time "
            f"window; the remainder are peak-specific and should be phased "
            f"or sized differently"),
    }


# =========================================================================== #
# Resilience
# =========================================================================== #
def resilience_analysis(cfg, D: np.ndarray, nodes: Sequence, dem: pd.DataFrame,
                        station_nodes: Sequence, max_removals: Optional[int] = None,
                        n_random_trials: int = 20) -> Dict:
    """Coverage decay as stations are taken out of service.

    Two attack models. Targeted removal takes the station currently serving
    the most demand, which is the worst case and also what a badly timed
    maintenance schedule looks like. Random removal averages over trials and
    stands in for ordinary faults. The gap between the curves is the network's
    exposure to losing its single most important site.
    """
    from .coverage import coverage_metrics, distance_to_nearest

    demand = dem["demand"].reindex(nodes).fillna(0.0).values
    stations = list(station_nodes)
    max_removals = max_removals or min(cfg.resilience_removals, len(stations) - 1)
    rng = np.random.default_rng(cfg.seed)

    def cov(sts):
        d, _ = distance_to_nearest(D, nodes, sts)
        return coverage_metrics(d, demand, cfg.coverage_radius_m)["demand_covered_pct"]

    rows = []
    # --- targeted ---------------------------------------------------------
    remaining = stations.copy()
    for k in range(max_removals + 1):
        c = cov(remaining)
        rows.append({"removed": k, "mode": "targeted", "demand_covered_pct": c})
        if k == max_removals or len(remaining) <= 1:
            break
        d, which = distance_to_nearest(D, nodes, remaining)
        inr = d <= cfg.coverage_radius_m
        loads = np.array([demand[(which == i) & inr].sum() for i in range(len(remaining))])
        remaining.pop(int(np.argmax(loads)))

    # --- random -----------------------------------------------------------
    for k in range(max_removals + 1):
        vals = []
        for _ in range(n_random_trials if k else 1):
            keep = list(rng.choice(stations, size=len(stations) - k, replace=False)) if k else stations
            vals.append(cov(keep))
        rows.append({"removed": k, "mode": "random",
                     "demand_covered_pct": float(np.mean(vals)),
                     "std": float(np.std(vals))})

    df = pd.DataFrame(rows)
    full = df[(df["removed"] == 0)]["demand_covered_pct"].max()
    t_end = df[(df["mode"] == "targeted")]["demand_covered_pct"].iloc[-1]
    r_end = df[(df["mode"] == "random")]["demand_covered_pct"].iloc[-1]

    return {
        "curve": df,
        "full_coverage_pct": float(full),
        "targeted_loss_pp": float(full - t_end),
        "random_loss_pp": float(full - r_end),
        "fragility_ratio": float((full - t_end) / max(full - r_end, 1e-9)),
        "n_removed": int(max_removals),
        "interpretation": (
            f"losing the {max_removals} busiest stations costs "
            f"{full - t_end:.1f} points of coverage, against "
            f"{full - r_end:.1f} for {max_removals} random failures"),
    }
