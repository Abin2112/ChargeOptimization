"""Unit 4.1 - Network elements and definitions.

Turns OpenStreetMap road data into the object every later stage works on:
a *directed, weighted* graph whose nodes are real intersections and whose
edges are road segments carrying length, speed and travel-time weights.

Three modelling decisions are made here, and they matter:

1. Intersection consolidation.  Raw OSM represents a single junction with
   several vertices (one per mapped way). Left alone, that inflates degree
   centrality and invents intersections that do not exist. Vertices within
   `consolidate_tolerance_m` are merged into one node.

2. Largest strongly connected component.  A drive network is directed, so a
   node you can enter but never leave breaks closeness and shortest-path
   coverage. Restricting to the largest SCC guarantees every pair of nodes
   is mutually reachable, which is what makes the distance metrics well
   defined.

3. Weights.  `length` (metres) drives distance work; `travel_time` (seconds)
   derived from OSM `maxspeed` drives anything where congestion matters.
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

log = logging.getLogger("evopt.network")


def _configure_osmnx(cfg) -> "module":  # type: ignore[valid-type]
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(Path(cfg.cache_dir) / "osmnx")
    ox.settings.log_console = False
    ox.settings.timeout = 300
    return ox


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_road_network(cfg, force: bool = False) -> nx.MultiDiGraph:
    """Download (or load from cache) the study-area road network.

    Returns a graph projected to a metric CRS, with consolidated
    intersections, restricted to the largest strongly connected component.
    """
    ox = _configure_osmnx(cfg)
    cache_file = Path(cfg.cache_dir) / f"network_{cfg.slug()}_{cfg.consolidate_tolerance_m:.0f}.graphml"

    if cache_file.exists() and not force:
        log.info("loading cached network from %s", cache_file)
        G = ox.load_graphml(cache_file)
        # graphml round-trips numbers as strings; restore the numeric weights
        for _, _, d in G.edges(data=True):
            for key in ("length", "travel_time", "speed_kph"):
                if key in d:
                    d[key] = float(d[key])
        for _, d in G.nodes(data=True):
            for key in ("x", "y", "lat", "lon"):
                if key in d:
                    d[key] = float(d[key])
        return G

    log.info("downloading OSM road network: %s r=%dm type=%s",
             cfg.area_name, cfg.radius_m, cfg.network_type)
    G_raw = ox.graph_from_point(cfg.center, dist=cfg.radius_m,
                                network_type=cfg.network_type, simplify=True)
    log.info("raw graph: %d nodes, %d edges", G_raw.number_of_nodes(), G_raw.number_of_edges())

    G_proj = ox.project_graph(G_raw)
    G = ox.consolidate_intersections(
        G_proj, tolerance=cfg.consolidate_tolerance_m, rebuild_graph=True,
        dead_ends=False, reconnect_edges=True,
    )
    log.info("consolidated: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    G = largest_strongly_connected(G)
    log.info("largest SCC: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    G = _attach_speeds(G, ox)
    G = _attach_latlon(G)

    ox.save_graphml(G, cache_file)
    log.info("cached network to %s", cache_file)
    return G


def largest_strongly_connected(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Keep the mutually reachable core of a directed road network."""
    if not nx.is_directed(G):
        comp = max(nx.connected_components(G), key=len)
    else:
        comp = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(comp).copy()


def _attach_speeds(G: nx.MultiDiGraph, ox) -> nx.MultiDiGraph:
    """Add speed_kph / travel_time, falling back to road-class defaults."""
    # Typical free-flow speeds for dense Indian urban roads (km/h).
    defaults = {
        "motorway": 70, "motorway_link": 45, "trunk": 55, "trunk_link": 35,
        "primary": 45, "primary_link": 30, "secondary": 38, "secondary_link": 28,
        "tertiary": 32, "tertiary_link": 25, "residential": 22,
        "living_street": 12, "unclassified": 25, "service": 15, "road": 25,
    }
    try:
        G = ox.add_edge_speeds(G, hwy_speeds=defaults, fallback=25)
        G = ox.add_edge_travel_times(G)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("osmnx speed enrichment failed (%s); using class defaults", exc)
        for _, _, d in G.edges(data=True):
            hwy = d.get("highway", "residential")
            if isinstance(hwy, (list, tuple)):
                hwy = hwy[0]
            kph = defaults.get(hwy, 25)
            d["speed_kph"] = float(kph)
            d["travel_time"] = float(d.get("length", 0.0)) / (kph * 1000.0 / 3600.0)

    for _, _, d in G.edges(data=True):
        d["length"] = float(d.get("length", 1.0))
        d["travel_time"] = float(d.get("travel_time", d["length"] / 6.94))
        d["speed_kph"] = float(d.get("speed_kph", 25.0))
    return G


def _attach_latlon(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Consolidation drops geographic coordinates; recover them from x/y."""
    import geopandas as gpd
    from shapely.geometry import Point

    crs = G.graph.get("crs")
    nodes = list(G.nodes)
    pts = [Point(float(G.nodes[n]["x"]), float(G.nodes[n]["y"])) for n in nodes]
    gs = gpd.GeoSeries(pts, crs=crs).to_crs("EPSG:4326")
    for n, pt in zip(nodes, gs):
        G.nodes[n]["lon"] = float(pt.x)
        G.nodes[n]["lat"] = float(pt.y)
    return G


# --------------------------------------------------------------------------- #
# Derived views
# --------------------------------------------------------------------------- #
def simple_digraph(G: nx.MultiDiGraph, weight: str = "length") -> nx.DiGraph:
    """Collapse parallel edges to the cheapest one.

    Centrality routines run far faster on a simple graph, and for shortest
    paths only the cheapest parallel edge can ever be used anyway.
    """
    H = nx.DiGraph()
    H.add_nodes_from((n, d.copy()) for n, d in G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = float(d.get(weight, 1.0))
        tt = float(d.get("travel_time", w / 6.94))
        if H.has_edge(u, v):
            if w < H[u][v]["length"]:
                H[u][v].update(length=w, travel_time=tt,
                               highway=d.get("highway"), speed_kph=d.get("speed_kph", 25.0))
        else:
            H.add_edge(u, v, length=w, travel_time=tt,
                       highway=d.get("highway"), speed_kph=d.get("speed_kph", 25.0))
    H.graph.update(G.graph)
    return H


def undirected_view(G: nx.DiGraph) -> nx.Graph:
    """Undirected projection used by modularity, k-core and spectral analysis."""
    U = nx.Graph()
    U.add_nodes_from((n, d.copy()) for n, d in G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = float(d["length"])
        if U.has_edge(u, v):
            U[u][v]["length"] = min(U[u][v]["length"], w)
        else:
            U.add_edge(u, v, length=w)
    # Modularity and spectral methods need "closeness = strong tie", so the
    # affinity weight is the inverse of road length (in km, clipped).
    for _, _, d in U.edges(data=True):
        d["affinity"] = 1.0 / max(d["length"] / 1000.0, 0.02)
    return U


def node_frame(G: nx.Graph) -> pd.DataFrame:
    """Tabular node index: the join key for every later analysis."""
    rows = []
    for n, d in G.nodes(data=True):
        rows.append({
            "node": n,
            "x": float(d.get("x", np.nan)),
            "y": float(d.get("y", np.nan)),
            "lat": float(d.get("lat", np.nan)),
            "lon": float(d.get("lon", np.nan)),
        })
    df = pd.DataFrame(rows).set_index("node")
    return df.sort_index()


def graph_summary(G: nx.MultiDiGraph, H: Optional[nx.DiGraph] = None) -> Dict:
    """Headline structural statistics (Unit 4.1 checkpoint)."""
    H = H if H is not None else simple_digraph(G)
    U = undirected_view(H)
    n, m = H.number_of_nodes(), H.number_of_edges()
    lengths = np.array([d["length"] for _, _, d in H.edges(data=True)])
    degs = np.array([d for _, d in U.degree()])
    return {
        "n_nodes": int(n),
        "n_edges_directed": int(m),
        "n_edges_undirected": int(U.number_of_edges()),
        "density": float(nx.density(H)),
        "mean_degree_undirected": float(degs.mean()),
        "max_degree": int(degs.max()),
        "total_road_km": float(lengths.sum() / 1000.0),
        "mean_edge_length_m": float(lengths.mean()),
        "median_edge_length_m": float(np.median(lengths)),
        "n_oneway_edges": int(sum(1 for u, v in H.edges() if not H.has_edge(v, u))),
        "share_oneway": float(sum(1 for u, v in H.edges() if not H.has_edge(v, u)) / max(m, 1)),
        "is_strongly_connected": bool(nx.is_strongly_connected(H)),
        "avg_clustering_undirected": float(nx.average_clustering(U)),
        "crs": str(G.graph.get("crs")),
    }


# --------------------------------------------------------------------------- #
# Existing infrastructure
# --------------------------------------------------------------------------- #
def load_existing_stations(cfg, G: nx.Graph) -> pd.DataFrame:
    """Charging stations already mapped in OSM, snapped to network nodes."""
    ox = _configure_osmnx(cfg)
    cache_file = Path(cfg.cache_dir) / f"stations_{cfg.slug()}.csv"

    if cache_file.exists():
        pois = pd.read_csv(cache_file)
    else:
        try:
            gdf = ox.features_from_point(
                cfg.center, tags={"amenity": "charging_station"}, dist=cfg.radius_m)
            if len(gdf):
                cent = gdf.to_crs("EPSG:4326").geometry.centroid
                pois = pd.DataFrame({
                    "name": gdf.get("name", pd.Series(index=gdf.index, dtype=object)).values,
                    "lat": cent.y.values,
                    "lon": cent.x.values,
                    "capacity": gdf.get("capacity", pd.Series(index=gdf.index, dtype=object)).values,
                })
            else:
                pois = pd.DataFrame(columns=["name", "lat", "lon", "capacity"])
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("could not fetch existing stations (%s)", exc)
            pois = pd.DataFrame(columns=["name", "lat", "lon", "capacity"])
        pois.to_csv(cache_file, index=False)

    if not len(pois):
        return pd.DataFrame(columns=["name", "lat", "lon", "capacity", "node"])

    pois = pois.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    pois["node"] = snap_to_nodes(G, pois["lat"].values, pois["lon"].values)
    pois["name"] = pois["name"].fillna("unnamed charging point")
    return pois.drop_duplicates(subset=["node"]).reset_index(drop=True)


def snap_to_nodes(G: nx.Graph, lats, lons) -> np.ndarray:
    """Nearest network node for each lat/lon, using projected coordinates.

    Done with a KD-tree on the metric CRS rather than osmnx's helper so that
    the same routine works on the consolidated graph, whose node ids no
    longer match the original OSM ids.
    """
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.geometry import Point

    nf = node_frame(G)
    tree = cKDTree(nf[["x", "y"]].values)

    pts = gpd.GeoSeries([Point(float(lo), float(la)) for la, lo in zip(lats, lons)],
                        crs="EPSG:4326").to_crs(G.graph.get("crs"))
    query = np.column_stack([pts.x.values, pts.y.values])
    _, idx = tree.query(query, k=1)
    return nf.index.values[idx]
