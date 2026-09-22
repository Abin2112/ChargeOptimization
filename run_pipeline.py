#!/usr/bin/env python3
"""End-to-end run: road network -> mathematics -> recommended charging sites.

    python run_pipeline.py                     # full study area
    python run_pipeline.py --quick             # small, fast smoke test
    python run_pipeline.py --place "Pune" --center-lat 18.52 --center-lon 73.85
    python run_pipeline.py --n-new-stations 20 --coverage-radius-m 1500

Outputs land in `outputs/`: an interactive map, a figure report, CSV tables and
a machine-readable results.json holding every metric printed below.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from evopt import advanced as AD
from evopt import centrality as CN
from evopt import communities as CM
from evopt import coverage as CV
from evopt import dashboard as DB
from evopt import demand as DM
from evopt import linalg as LA
from evopt import network as NW
from evopt import optimise as OP
from evopt import traffic as TR
from evopt.config import Config

log = logging.getLogger("evopt")


# --------------------------------------------------------------------------- #
def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Series, pd.DataFrame)):
        return json.loads(o.to_json(orient="split"))
    if isinstance(o, pd.Index):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def banner(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * len(title))


def kv(label: str, value, unit: str = "") -> None:
    if isinstance(value, float):
        value = f"{value:,.3f}".rstrip("0").rstrip(".") if abs(value) < 1e4 else f"{value:,.0f}"
    print(f"  {label:<46} {value}{unit}")


# --------------------------------------------------------------------------- #
def run(cfg: Config) -> dict:
    t_start = time.time()
    art: dict = {}
    results: dict = {"config": cfg.to_dict()}

    # ===================================================================== #
    banner("1. Road network from OpenStreetMap  (Unit 4.1)")
    G = NW.build_road_network(cfg)
    H = NW.simple_digraph(G)
    U = NW.undirected_view(H)
    summary = NW.graph_summary(G, H)
    results["network"] = summary
    for k in ("n_nodes", "n_edges_directed", "total_road_km", "mean_degree_undirected",
              "share_oneway", "is_strongly_connected"):
        kv(k.replace("_", " "), summary[k])

    nodes = sorted(H.nodes())
    coords = NW.node_frame(H)
    existing = NW.load_existing_stations(cfg, H)
    art["existing"] = existing
    kv("charging stations already mapped", len(existing))

    # ===================================================================== #
    banner("2. Charging demand from land use")
    dem = DM.build_demand_surface(cfg, H)
    dsum = DM.demand_summary(dem)
    results["demand"] = dsum
    kv("demand units distributed", dsum["total_demand_units"])
    kv("concentration (Gini)", dsum["gini"])
    kv("share of demand in the busiest 10% of junctions",
       f"{100*dsum['top10pct_share']:.1f}", "%")

    # ===================================================================== #
    banner("3. Linear algebra checkpoints  (Units 1.1, 1.2)")
    inc = LA.verify_incidence_laplacian(U)
    results["incidence"] = inc
    kv("incidence matrix B", f"{inc['shape_B'][0]} x {inc['shape_B'][1]}")
    kv("B B' reproduces the Laplacian", inc["identity_holds"])
    kv("max abs error", inc["max_abs_error_BBt_vs_L"])
    kv("rank of B (n - components)", inc["rank_B_expected"])

    import networkx as nx
    A_und = nx.to_scipy_sparse_array(U, nodelist=nodes, weight=None, format="csr").astype(float)
    pv = LA.validate_power_iteration(A_und)
    results["power_iteration"] = pv
    kv("power iteration lambda_1", pv["lambda_power_iteration"])
    kv("agrees with reference eigensolver", pv["agrees"])
    kv("eigenvector cosine similarity", pv["eigenvector_cosine_similarity"])
    kv("iterations to converge", pv["iterations"])

    prop = LA.propagate(A_und.toarray(), np.ones(len(nodes)), steps=10)
    kv("flow iteration x -> Ax, final direction drift",
       f"{prop['angle_drift_deg'][-1]:.3f}", " deg")

    # ===================================================================== #
    banner("4. Traffic panel and SVD denoising  (Unit 1.3)")
    panel = TR.simulate_traffic_panel(cfg, H, dem)
    svd = LA.svd_denoise(panel["X_obs"], rank=cfg.svd_rank, energy=cfg.svd_energy,
                         X_true=panel["X_true"], rule=cfg.svd_rank_rule)
    art["panel"], art["svd"] = panel, svd
    art["rank_sweep"] = LA.rank_sweep(panel["X_obs"], panel["X_true"],
                                      max_rank=min(12, cfg.n_sensors - 1))
    results["svd"] = {k: v for k, v in svd.items()
                      if isinstance(v, (int, float, bool, str))}
    kv("panel", f"{panel['X_true'].shape[0]} sensors x {panel['X_true'].shape[1]} hours")
    kv("latent regimes injected", cfg.true_rank)
    kv("rank selected (Gavish-Donoho threshold)", svd["rank_gavish_donoho"])
    kv("rank by fixed energy rule", svd["rank_energy_rule"])
    kv("error before denoising", f"{svd['relative_error_before']*100:.2f}", "%")
    kv("error after denoising", f"{svd['relative_error_after']*100:.2f}", "%")
    kv("noise removed", f"{svd['noise_removed_pct']:.1f}", "%")
    kv("signal-to-noise gain",
       f"{svd['snr_before_db']:.1f} -> {svd['snr_after_db']:.1f}", " dB")
    kv("compression ratio", f"{svd['compression_ratio']:.0f}", "x")

    # ===================================================================== #
    banner("5. Centrality measures  (Unit 4.4)")
    D, nodes = CN.distance_matrix(H, nodes=nodes)
    cent = CN.compute_centralities(cfg, H, U, D, nodes)
    cent = CN.combined_score(cfg, cent)
    ev_check = CN.validate_eigenvector(cent, U)
    csum = CN.centrality_summary(cent)
    results["centrality"] = csum
    results["centrality"]["eigenvector_validation"] = ev_check
    kv("eigenvector centrality matches NetworkX", ev_check["validated"])
    kv("mean agreement between measures (Spearman)",
       csum["mean_spearman_between_measures"])
    kv("most similar pair",
       f"{csum['max_spearman_pair']['pair']} rho={csum['max_spearman_pair']['rho']:.2f}")
    kv("least similar pair",
       f"{csum['min_spearman_pair']['pair']} rho={csum['min_spearman_pair']['rho']:.2f}")
    kv("betweenness held by the top 10% of junctions",
       f"{100*csum['betweenness_top10pct_share']:.1f}", "%")

    loc = AD.eigenvector_localisation(cent)
    results["eigenvector_localisation"] = loc
    kv("dominant eigenvector effective support",
       f"{loc['effective_support_nodes']:.0f} of {loc['n_nodes']} junctions")

    # ===================================================================== #
    banner("6. Feature independence  (Unit 1.4)")
    sensors = panel["sensor_nodes"]
    mean_flow = TR.interpolate_to_nodes(svd["row_means"], sensors, nodes, dem)
    arterial = TR.arterial_score(H).reindex(nodes)

    # Ordered most-wanted first: Gram-Schmidt keeps earlier columns and flags
    # later ones as redundant. `demand_share_pct` is demand times a constant -
    # the deliberate stand-in for the syllabus's "population density vs number
    # of residents" case, included to show the check catching it.
    F = pd.DataFrame({
        "betweenness": cent["betweenness"].reindex(nodes),
        "demand": dem["demand"].reindex(nodes),
        "closeness": cent["closeness"].reindex(nodes),
        "harmonic": cent["harmonic"].reindex(nodes),
        "degree": cent["degree"].reindex(nodes),
        "eigenvector": cent["eigenvector"].reindex(nodes),
        "arterial_class": arterial,
        "mean_traffic_flow": mean_flow.reindex(nodes),
        "demand_share_pct": dem["demand_share_pct"].reindex(nodes),
    }).fillna(0.0)

    fi = LA.feature_independence(F, redundancy_threshold=0.15)
    art["feature_independence"] = fi
    art["redundancy_threshold"] = 0.15
    results["feature_independence"] = {
        "residual_ratio": fi["residual_ratio"].round(4).to_dict(),
        "redundant_features": fi["redundant_features"],
        "retained_features": fi["retained_features"],
        "condition_number": fi["condition_number"],
        "vif": {k: round(v, 2) for k, v in fi["vif"].items()},
        "effective_rank": fi["effective_rank"],
        "orthogonality_error": fi["orthogonality_error"],
    }
    kv("features tested", len(F.columns))
    kv("flagged as redundant", ", ".join(fi["redundant_features"]) or "none")
    kv("condition number of the feature matrix", f"{fi['condition_number']:.1f}")
    kv("orthogonality error after Gram-Schmidt", fi["orthogonality_error"])
    for name, r in fi["residual_ratio"].sort_values().items():
        flag = "  <- redundant" if name in fi["redundant_features"] else ""
        print(f"      {name:<22} independent share {r:5.3f}  VIF {fi['vif'][name]:7.1f}{flag}")

    # ===================================================================== #
    banner("7. Communities, small worlds and cores  (Units 4.2, 4.5, 4.6)")
    comm = CM.detect_communities(cfg, U)
    core = CM.core_decomposition(U)
    profile = CM.community_profile(comm, dem, cent, existing["node"].tolist()
                                   if len(existing) else [])
    comm_sum = CM.community_summary(comm, U, profile)
    sw = CM.small_world_analysis(U, seed=cfg.seed)
    hubs = CM.identify_hubs(cent, core, dem, top_n=25)
    results["communities"] = comm_sum
    results["small_world"] = sw
    kv("communities found", comm_sum["n_communities"])
    kv("modularity", comm_sum["modularity"])
    kv("communities with no charging station", comm_sum["communities_without_a_station"])
    kv("demand sitting in those communities",
       f"{comm_sum['demand_in_unserved_communities_pct']:.1f}", "%")
    kv("small-world regime (omega)", f"{sw['regime']} (omega={sw['omega']:+.3f})")
    kv("paths vs a random graph", f"{sw['path_length_ratio_vs_random']:.2f}", "x longer")
    kv("maximum k-core", int(core.max()))

    # ===================================================================== #
    banner("8. Coverage today  (Unit 4.3)")
    demand_vec = dem["demand"].reindex(nodes).fillna(0.0).values
    ex_nodes = existing["node"].tolist() if len(existing) else []
    d_before, _ = CV.distance_to_nearest(D, nodes, ex_nodes)
    before = CV.coverage_metrics(d_before, demand_vec, cfg.coverage_radius_m, "today")
    results["coverage_before"] = before
    kv(f"demand within {cfg.coverage_radius_m/1000:.1f} km",
       f"{before['demand_covered_pct']:.1f}", "%")
    kv("average drive to a charger", f"{before['mean_distance_m']/1000:.2f}", " km")
    kv("90th percentile drive", f"{before['p90_distance_m']/1000:.2f}", " km")

    # ===================================================================== #
    banner("9. Optimisation  (Deliverables 5 and 6)")
    res = OP.optimise_placement(cfg, D, nodes, cent, dem, ex_nodes)
    art["result"] = res
    kv("candidate sites considered", len(res["candidates"]))
    kv("solver used", res["solver"])
    if "ilp_status" in res:
        kv("ILP status", res["ilp_status"])
    if "greedy_vs_ilp_gap_pp" in res:
        kv("greedy coverage", f"{res['greedy_objective_pct']:.2f}", "%")
        kv("ILP coverage", f"{res['ilp_objective_pct']:.2f}", "%")
        kv("greedy share of the proven optimum",
           f"{100*res['greedy_ratio_of_optimum']:.1f}", "%  (guarantee: 63.2%)")
    kv("total capital cost", f"Rs {res['total_cost_lakh']:,.0f}", " lakh")

    costs = res["costs"]
    pareto = OP.pareto_front(cfg, res["coverage_matrix"], demand_vec, costs,
                             res["precovered_mask"], cfg.pareto_p_values)
    knee = OP.knee_point(pareto)
    art["pareto"], art["knee"] = pareto, knee
    results["pareto"] = json.loads(pareto.to_json(orient="records"))
    results["knee"] = knee
    kv("best value for money", f"{knee['n_stations']} stations for "
       f"Rs {knee['total_cost_lakh']:.0f} lakh -> {knee['demand_covered_pct']:.1f}% covered")

    pm = OP.p_median_local_search(D, nodes, res["candidates"], demand_vec,
                                 cfg.n_new_stations,
                                 init=res["selected"], seed=cfg.seed)
    results["p_median"] = {"mean_weighted_distance_m": pm["mean_weighted_distance_m"]}
    kv("p-median alternative: mean drive",
       f"{pm['mean_weighted_distance_m']/1000:.2f}", " km")

    # ===================================================================== #
    banner("10. Coverage after the recommended build  (Unit 4.3)")
    all_stations = list(ex_nodes) + res["chosen_nodes"]
    d_after, which = CV.distance_to_nearest(D, nodes, all_stations)
    after = CV.coverage_metrics(d_after, demand_vec, cfg.coverage_radius_m, "after build")
    results["coverage_after"] = after
    art["coverage_before"], art["coverage_after"] = before, after
    art["n_existing"] = len(existing)
    art["coverage_curves"] = {
        "Today": CV.coverage_curve(d_before, demand_vec),
        "After recommended build": CV.coverage_curve(d_after, demand_vec),
    }
    art["community_coverage"] = CV.coverage_by_community(
        d_after, demand_vec, nodes, comm, cfg.coverage_radius_m)
    art["station_load"] = CV.station_load(d_after, which, demand_vec, all_stations,
                                          cfg.coverage_radius_m, cfg.station_capacity)
    art["centrality_corr"] = CN.centrality_correlations(cent)

    print(CV.compare_scenarios({"today": before, "after build": after})
          .round(2).to_string())
    kv("coverage gained",
       f"+{after['demand_covered_pct'] - before['demand_covered_pct']:.1f}", " points")
    kv("average drive reduced by",
       f"{100*(1 - after['mean_distance_m']/before['mean_distance_m']):.0f}", "%")
    kv("access inequality (Gini)",
       f"{before['access_gini']:.3f} -> {after['access_gini']:.3f}")
    cap = CV.capacity_check(art["station_load"], after["demand_covered_pct"] / 100.0
                            * demand_vec.sum(), cfg.station_capacity, len(all_stations))
    results["capacity"] = cap
    kv("mean station utilisation", f"{cap['mean_utilisation_pct']:.0f}", "% of capacity")
    kv("stations over capacity",
       f"{cap['stations_over_capacity']} of {cap['stations_built']}")
    kv("binding constraint", cap["binding_constraint"])
    print(f"      {cap['recommendation']}")

    # ===================================================================== #
    if cfg.run_advanced:
        banner("11. Advanced analyses  (bonus)")
        spec = AD.spectral_analysis(U)
        results["spectral"] = {k: v for k, v in spec.items() if k != "fiedler_vector"}
        kv("algebraic connectivity lambda_2", spec["algebraic_connectivity_lambda2"])
        kv("spectral bisection", f"{spec['bisection_sizes']} split by "
           f"{spec['bisection_cut_edges']} roads")
        print(f"      {spec['interpretation']}")

        ev_edges = AD.top_eigenvector_edges(U, cent, n=4)
        sweep = AD.perturbation_sweep(U, ev_edges)
        ok = sweep[~sweep["level_crossing"]]
        results["perturbation"] = json.loads(sweep.to_json(orient="records"))
        kv("perturbation theory: severity tested",
           f"{sweep['scale'].min():.2f} to {sweep['scale'].max():.2f}")
        if len(ok):
            kv("first-order error at the mildest perturbation",
               f"{ok['rel_error_pct'].iloc[0]:.2f}", "%")
            kv("still valid up to severity", f"{ok['scale'].max():.2f}")
        if sweep["level_crossing"].any():
            kv("theory breaks down at severity",
               f"{sweep[sweep['level_crossing']]['scale'].min():.2f}",
               "  (eigenvalue level crossing)")

        activity = AD.temporal_activity(cfg, panel, svd, nodes, dem)
        tmp = AD.temporal_optimisation(cfg, D, nodes, dem, activity, res["candidates"],
                                       res["precovered_mask"])
        results["temporal"] = {
            "per_window": json.loads(tmp["per_window"].to_json(orient="records")),
            "n_robust_sites": tmp["n_robust_sites"],
            "n_sites_ever_chosen": tmp["n_sites_ever_chosen"],
            "stability": tmp["stability"],
        }
        kv("sites optimal in every time window",
           f"{tmp['n_robust_sites']} of {tmp['n_sites_ever_chosen']}")
        kv("temporal stability", f"{tmp['stability']:.2f}")

        rl = AD.resilience_analysis(cfg, D, nodes, dem, all_stations)
        art["resilience"] = rl
        results["resilience"] = {
            "targeted_loss_pp": rl["targeted_loss_pp"],
            "random_loss_pp": rl["random_loss_pp"],
            "fragility_ratio": rl["fragility_ratio"],
            "curve": json.loads(rl["curve"].to_json(orient="records")),
        }
        kv("coverage lost to targeted outages",
           f"{rl['targeted_loss_pp']:.1f}", " points")
        kv("coverage lost to random outages", f"{rl['random_loss_pp']:.1f}", " points")
        kv("fragility ratio", rl["fragility_ratio"])

    # ===================================================================== #
    banner("12. Outputs  (Deliverable 8)")
    node_view = coords.copy()
    node_view["demand"] = dem["demand"].reindex(node_view.index)
    node_view["priority_score"] = cent["priority_score"].reindex(node_view.index)
    node_view["betweenness"] = cent["betweenness"].reindex(node_view.index)
    node_view["community"] = comm.reindex(node_view.index)
    node_view["core_number"] = core.reindex(node_view.index)
    node_view["dist_before"] = pd.Series(d_before, index=nodes).reindex(node_view.index)
    node_view["dist_after"] = pd.Series(d_after, index=nodes).reindex(node_view.index)
    art["node_view"] = node_view

    out = cfg.out_path
    res["recommendations"].to_csv(out / "recommended_stations.csv", index=False)
    node_view.to_csv(out / "node_analysis.csv")
    profile.to_csv(out / "community_profile.csv")
    hubs.to_csv(out / "hub_junctions.csv")
    pareto.to_csv(out / "pareto_front.csv", index=False)
    art["station_load"].to_csv(out / "station_load.csv", index=False)
    art["community_coverage"].to_csv(out / "community_coverage.csv")
    cfg.save()

    DB.build_report(cfg, art)
    DB.build_map(cfg, art)

    results["runtime_seconds"] = time.time() - t_start
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_jsonable))

    for f in sorted(out.iterdir()):
        kv(f.name, f"{f.stat().st_size/1024:,.0f} KB")

    banner("Recommended sites, in build order")
    show = res["recommendations"][["rank", "node", "lat", "lon", "priority_score",
                                   "marginal_coverage_pct", "cost_lakh"]]
    print(show.round({"lat": 5, "lon": 5, "priority_score": 1,
                      "marginal_coverage_pct": 2, "cost_lakh": 1}).to_string(index=False))

    print(f"\n\033[1mDone in {results['runtime_seconds']:.0f}s.\033[0m  "
          f"Coverage {before['demand_covered_pct']:.0f}% -> "
          f"{after['demand_covered_pct']:.0f}% of charging demand within "
          f"{cfg.coverage_radius_m/1000:.0f} km.")
    print(f"Open {out/'map.html'} and {out/'report.html'}")
    return results


# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        description="EV charging station placement optimiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--place", dest="area_name", help="label for the study area")
    p.add_argument("--center-lat", dest="center_lat", type=float)
    p.add_argument("--center-lon", dest="center_lon", type=float)
    p.add_argument("--radius-m", dest="radius_m", type=int)
    p.add_argument("--network-type", dest="network_type",
                   choices=["drive", "drive_service", "all"])
    p.add_argument("--n-new-stations", dest="n_new_stations", type=int)
    p.add_argument("--coverage-radius-m", dest="coverage_radius_m", type=float)
    p.add_argument("--n-candidates", dest="n_candidates", type=int)
    p.add_argument("--svd-rank", dest="svd_rank", type=int)
    p.add_argument("--svd-rank-rule", dest="svd_rank_rule",
                   choices=["gavish_donoho", "energy"])
    p.add_argument("--no-ilp", dest="use_ilp", action="store_false", default=None)
    p.add_argument("--no-advanced", dest="run_advanced", action="store_false", default=None)
    p.add_argument("--quick", action="store_true", default=None,
                   help="small fast run for testing")
    p.add_argument("--seed", type=int)
    p.add_argument("--outdir")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="\033[2m%(name)-18s %(message)s\033[0m", stream=sys.stdout)
    return Config.from_namespace(args)


if __name__ == "__main__":
    cfg = parse_args()
    print(f"\n\033[1mEV charging station optimisation\033[0m — {cfg.area_name} "
          f"({cfg.radius_m/1000:.0f} km radius)")
    run(cfg)
