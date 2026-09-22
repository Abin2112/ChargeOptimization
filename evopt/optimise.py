"""Deliverables 5 and 6 - the optimisation model and the recommendations.

The siting question is posed as the Maximal Covering Location Problem (MCLP):

    choose p sites from a candidate set so as to maximise the charging demand
    that lies within `coverage_radius_m` of driving distance from a station,
    counting demand already covered by stations that exist today.

Written as an integer program, with y_j = 1 if candidate j is built and z_i
the covered-indicator of demand point i:

    maximise    sum_i  d_i * z_i
    subject to  z_i <= sum_{j in N_i} y_j      for every demand point i
                sum_j y_j = p
                y_j in {0,1},  0 <= z_i <= 1

N_i is the set of candidates covering point i, read straight off the
network-distance matrix. Two solvers are provided:

  * greedy   - add the site with the largest marginal gain, repeatedly.
               Coverage is a monotone submodular set function, so this is
               guaranteed to reach at least (1 - 1/e) ~ 63.2% of the optimum,
               and in practice lands far closer.
  * ILP      - branch and bound through CBC, which proves optimality.

Both are run, and the gap between them is reported rather than assumed: that
comparison is the evidence the recommendation is actually good.

The connection back to Unit 1 is the coverage matrix itself. It is a linear
operator: given a 0/1 site-selection vector y, the product A y counts how many
chosen stations cover each demand point, and the whole objective is the linear
functional d' min(A y, 1).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("evopt.optimise")


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
def select_candidates(cfg, cent: pd.DataFrame, dem: pd.DataFrame,
                      existing: Optional[Sequence] = None) -> List:
    """Shortlist plausible sites before optimising.

    Optimising over all 2,480 junctions is possible but wasteful and produces
    absurd recommendations (a charger in the middle of a cul-de-sac). Sites
    are scored on centrality and demand together, then thinned so that no two
    candidates sit within `candidate_min_separation_m` of each other - without
    that, the whole shortlist collapses onto one busy corridor and the
    optimiser has no real geographic choice to make.
    """
    from scipy.spatial import cKDTree

    score = (0.55 * cent["priority_score"].rank(pct=True)
             + 0.45 * dem["demand"].reindex(cent.index).rank(pct=True))
    ranked = score.sort_values(ascending=False)

    xy = dem.reindex(ranked.index)[["x", "y"]].values
    chosen_idx: List[int] = []
    chosen_xy: List[np.ndarray] = []
    sep = cfg.candidate_min_separation_m

    for i in range(len(ranked)):
        if len(chosen_idx) >= cfg.n_candidates:
            break
        p = xy[i]
        if chosen_xy:
            d = np.linalg.norm(np.asarray(chosen_xy) - p, axis=1)
            if d.min() < sep:
                continue
        chosen_idx.append(i)
        chosen_xy.append(p)

    cands = list(ranked.index[chosen_idx])
    log.info("candidates: %d sites, min separation %.0fm, "
             "priority score range %.1f-%.1f",
             len(cands), sep,
             cent.loc[cands, "priority_score"].min(),
             cent.loc[cands, "priority_score"].max())
    return cands


def build_coverage_matrix(D: np.ndarray, nodes: Sequence, candidates: Sequence,
                          radius: float) -> np.ndarray:
    """A[i, j] = 1 if candidate j is within `radius` driving distance of i.

    Direction: D[i, j] is i -> j, so this asks whether a driver at demand
    point i can reach candidate j, which is the correct orientation.
    """
    pos = {n: i for i, n in enumerate(nodes)}
    cols = [pos[c] for c in candidates]
    return (D[:, cols] <= radius)


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def greedy_mclp(A: np.ndarray, demand: np.ndarray, p: int,
                precovered: Optional[np.ndarray] = None) -> Dict:
    """Greedy maximal covering, with the submodular guarantee.

    At each step pick the candidate adding the most *currently uncovered*
    demand. Diminishing returns are exactly submodularity, which is what
    makes the 1 - 1/e bound hold.
    """
    n_dem, n_cand = A.shape
    covered = np.zeros(n_dem, dtype=bool) if precovered is None else precovered.copy()
    selected, gains = [], []

    for _ in range(min(p, n_cand)):
        uncovered = ~covered
        # Marginal gain of every candidate at once.
        marginal = (A[uncovered].T * demand[uncovered]).sum(axis=1)
        marginal[selected] = -1.0
        j = int(np.argmax(marginal))
        if marginal[j] <= 0:
            break
        selected.append(j)
        gains.append(float(marginal[j]))
        covered |= A[:, j]

    total = float(demand.sum())
    return {
        "selected": selected,
        "marginal_gains": gains,
        "covered_mask": covered,
        "objective": float(demand[covered].sum()),
        "objective_pct": float(100.0 * demand[covered].sum() / total) if total else 0.0,
        "solver": "greedy",
        "guarantee": ">= (1 - 1/e) = 63.2% of optimum",
    }


def ilp_mclp(A: np.ndarray, demand: np.ndarray, p: int,
             precovered: Optional[np.ndarray] = None,
             time_limit: int = 180, msg: bool = False) -> Dict:
    """Exact MCLP through CBC branch and bound."""
    import pulp

    n_dem, n_cand = A.shape
    pre = np.zeros(n_dem, dtype=bool) if precovered is None else precovered
    # Points already covered, or that no candidate can reach, are not decisions.
    reachable = A.any(axis=1)
    active = np.where((~pre) & reachable & (demand > 0))[0]

    prob = pulp.LpProblem("MCLP", pulp.LpMaximize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_cand)]
    z = {i: pulp.LpVariable(f"z_{i}", lowBound=0, upBound=1) for i in active}

    prob += pulp.lpSum(demand[i] * z[i] for i in active)
    prob += pulp.lpSum(y) == min(p, n_cand)
    for i in active:
        js = np.flatnonzero(A[i])
        prob += z[i] <= pulp.lpSum(y[j] for j in js)

    t0 = time.time()
    status = prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
    elapsed = time.time() - t0

    selected = [j for j in range(n_cand) if y[j].value() and y[j].value() > 0.5]
    covered = pre.copy()
    for j in selected:
        covered |= A[:, j]

    total = float(demand.sum())
    return {
        "selected": selected,
        "covered_mask": covered,
        "objective": float(demand[covered].sum()),
        "objective_pct": float(100.0 * demand[covered].sum() / total) if total else 0.0,
        "solver": "ilp_cbc",
        "status": pulp.LpStatus[status],
        "proven_optimal": pulp.LpStatus[status] == "Optimal",
        "solve_seconds": elapsed,
        "n_binary_vars": n_cand,
        "n_constraints": len(active) + 1,
    }


def p_median_local_search(D: np.ndarray, nodes: Sequence, candidates: Sequence,
                          demand: np.ndarray, p: int, init: Optional[List] = None,
                          max_passes: int = 25, seed: int = 42) -> Dict:
    """p-median by Teitz-Bart vertex substitution.

    MCLP asks "how many people are within range"; p-median asks "how far does
    the average driver travel". The two disagree, and seeing where is part of
    the analysis: covering rules stop rewarding you once a user is inside the
    radius, so MCLP will happily leave someone at 1,999 m.
    """
    rng = np.random.default_rng(seed)
    pos = {n: i for i, n in enumerate(nodes)}
    cols = np.array([pos[c] for c in candidates])
    Dc = D[:, cols]                     # demand points x candidates
    total = demand.sum()

    def cost(sel):
        return float((demand * Dc[:, sel].min(axis=1)).sum() / total)

    cur = list(init) if init else list(rng.choice(len(candidates), size=p, replace=False))
    cur = cur[:p]
    best = cost(cur)

    for _ in range(max_passes):
        improved = False
        for out_i in range(len(cur)):
            for cand in range(len(candidates)):
                if cand in cur:
                    continue
                trial = cur.copy()
                trial[out_i] = cand
                c = cost(trial)
                if c < best - 1e-9:
                    cur, best, improved = trial, c, True
        if not improved:
            break

    return {
        "selected": cur,
        "mean_weighted_distance_m": best,
        "solver": "p_median_teitz_bart",
        "objective": best,
    }


# --------------------------------------------------------------------------- #
# Cost model and multi-objective view
# --------------------------------------------------------------------------- #
def site_costs(cfg, cent: pd.DataFrame, dem: pd.DataFrame,
               candidates: Sequence) -> pd.Series:
    """Capital cost per site, in lakh rupees.

    Land near a busy, high-demand junction is more expensive, which is the
    central tension of the problem: the best sites for coverage are the
    priciest to acquire. Modelled as a base cost plus a premium rising with
    the site's demand and centrality percentile.
    """
    d = dem["demand"].reindex(candidates).rank(pct=True)
    c = cent["priority_score"].reindex(candidates).rank(pct=True)
    premium = cfg.land_premium_lakh * (0.6 * d + 0.4 * c)
    return (cfg.base_station_cost_lakh + premium).rename("cost_lakh")


def pareto_front(cfg, A: np.ndarray, demand: np.ndarray, costs: np.ndarray,
                 precovered: np.ndarray, p_values: Sequence[int]) -> pd.DataFrame:
    """Coverage against budget, for a range of programme sizes.

    The point of the sweep is the shape, not any single row: submodularity
    guarantees the curve is concave, so there is a knee beyond which each new
    station buys noticeably less coverage. That knee is the honest
    recommendation to a planner with a flexible budget.
    """
    base = float(demand[precovered].sum() / demand.sum() * 100.0)
    rows = []
    for p in p_values:
        r = greedy_mclp(A, demand, int(p), precovered)
        sel = r["selected"]
        rows.append({
            "n_stations": int(p),
            "demand_covered_pct": r["objective_pct"],
            "incremental_coverage_pct": r["objective_pct"] - base,
            "total_cost_lakh": float(costs[sel].sum()),
            "coverage_per_crore": (r["objective_pct"] - base) / max(costs[sel].sum() / 100.0, 1e-9),
        })
    df = pd.DataFrame(rows)
    df["marginal_coverage_pct"] = df["demand_covered_pct"].diff().fillna(
        df["demand_covered_pct"] - base)
    df["marginal_cost_lakh"] = df["total_cost_lakh"].diff().fillna(df["total_cost_lakh"])
    df["marginal_pct_per_lakh"] = df["marginal_coverage_pct"] / df["marginal_cost_lakh"].replace(0, np.nan)
    return df


def knee_point(pareto: pd.DataFrame, x: str = "total_cost_lakh",
               y: str = "demand_covered_pct") -> Dict:
    """Knee of the trade-off curve: the point furthest from the chord.

    Standard 'elbow' geometry - join the cheapest and richest options with a
    straight line and take the option with the greatest perpendicular
    distance above it.
    """
    X = pareto[x].values.astype(float)
    Y = pareto[y].values.astype(float)
    if len(X) < 3:
        return {"n_stations": int(pareto["n_stations"].iloc[-1])}
    p0, p1 = np.array([X[0], Y[0]]), np.array([X[-1], Y[-1]])
    v = p1 - p0
    v = v / (np.linalg.norm(v) or 1.0)
    dists = []
    for xi, yi in zip(X, Y):
        w = np.array([xi, yi]) - p0
        dists.append(np.linalg.norm(w - (w @ v) * v))
    i = int(np.argmax(dists))
    return {
        "n_stations": int(pareto["n_stations"].iloc[i]),
        "demand_covered_pct": float(pareto["demand_covered_pct"].iloc[i]),
        "total_cost_lakh": float(pareto["total_cost_lakh"].iloc[i]),
        "distance_from_chord": float(dists[i]),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def optimise_placement(cfg, D: np.ndarray, nodes: Sequence, cent: pd.DataFrame,
                       dem: pd.DataFrame, existing_nodes: Sequence,
                       candidates: Optional[List] = None) -> Dict:
    """Run the full siting model and return ranked recommendations."""
    candidates = candidates or select_candidates(cfg, cent, dem, existing_nodes)
    demand = dem["demand"].reindex(nodes).fillna(0.0).values
    A = build_coverage_matrix(D, nodes, candidates, cfg.coverage_radius_m)

    # Demand already within reach of a charger that exists today.
    pre = np.zeros(len(nodes), dtype=bool)
    if len(existing_nodes):
        E = build_coverage_matrix(D, nodes, list(existing_nodes), cfg.coverage_radius_m)
        pre = E.any(axis=1)

    costs = site_costs(cfg, cent, dem, candidates).values

    g = greedy_mclp(A, demand, cfg.n_new_stations, pre)
    result = dict(g)

    if cfg.use_ilp:
        try:
            ilp = ilp_mclp(A, demand, cfg.n_new_stations, pre, cfg.ilp_time_limit_s)
            gap = ilp["objective_pct"] - g["objective_pct"]
            log.info("MCLP p=%d: greedy %.2f%% | ILP %.2f%% (%s, %.1fs) | gap %+.2f pp",
                     cfg.n_new_stations, g["objective_pct"], ilp["objective_pct"],
                     ilp["status"], ilp["solve_seconds"], gap)
            if ilp["objective"] >= g["objective"]:
                # Keep the better solution, but never lose the comparison:
                # the greedy-vs-optimal gap is the evidence, not a side note.
                result = dict(ilp)
            result["ilp_objective_pct"] = ilp["objective_pct"]
            result["greedy_vs_ilp_gap_pp"] = gap
            result["greedy_ratio_of_optimum"] = (
                g["objective_pct"] / ilp["objective_pct"] if ilp["objective_pct"] else np.nan)
            result["ilp_status"] = ilp["status"]
            result["ilp_solve_seconds"] = ilp["solve_seconds"]
        except Exception as exc:  # pragma: no cover - solver availability
            log.warning("ILP unavailable (%s); keeping the greedy solution", exc)
            result["ilp_status"] = f"unavailable: {exc}"

    # Recorded after any solver swap above, so it always survives.
    result["greedy_objective_pct"] = g["objective_pct"]

    sel = result["selected"]
    chosen_nodes = [candidates[j] for j in sel]

    # Attribute a marginal contribution to each chosen site, in the order the
    # greedy solver added them, so the list can be built in priority order.
    order_gains = _attribute_marginals(A, demand, sel, pre)

    recs = pd.DataFrame({
        "candidate_index": sel,
        "node": chosen_nodes,
        "lat": dem["lat"].reindex(chosen_nodes).values,
        "lon": dem["lon"].reindex(chosen_nodes).values,
        "priority_score": cent["priority_score"].reindex(chosen_nodes).values,
        "betweenness": cent["betweenness"].reindex(chosen_nodes).values,
        "local_demand": dem["demand"].reindex(chosen_nodes).values,
        "cost_lakh": costs[sel],
        "marginal_demand_covered": [order_gains[j] for j in sel],
    })
    recs["marginal_coverage_pct"] = 100.0 * recs["marginal_demand_covered"] / demand.sum()
    recs = recs.sort_values("marginal_demand_covered", ascending=False).reset_index(drop=True)
    recs.insert(0, "rank", np.arange(1, len(recs) + 1))

    result.update({
        "candidates": candidates,
        "coverage_matrix": A,
        "precovered_mask": pre,
        "costs": costs,
        "recommendations": recs,
        "chosen_nodes": chosen_nodes,
        "baseline_covered_pct": float(100.0 * demand[pre].sum() / demand.sum()),
        "total_cost_lakh": float(costs[sel].sum()),
    })
    return result


def _attribute_marginals(A: np.ndarray, demand: np.ndarray, selected: List[int],
                         pre: np.ndarray) -> Dict[int, float]:
    """Greedy-order marginal contribution of each selected site.

    Coverage overlaps, so a site's standalone value and its value given the
    others differ. Sequencing the chosen set greedily gives each site a
    contribution that sums exactly to the total gain - the ordering a
    phased rollout should follow.
    """
    covered = pre.copy()
    remaining = list(selected)
    out: Dict[int, float] = {}
    while remaining:
        best_j, best_gain = None, -1.0
        for j in remaining:
            gain = float(demand[A[:, j] & ~covered].sum())
            if gain > best_gain:
                best_j, best_gain = j, gain
        out[best_j] = best_gain
        covered |= A[:, best_j]
        remaining.remove(best_j)
    return out
