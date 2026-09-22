"""
evopt - EV Charging Station Placement Optimisation
==================================================

A mathematical-optimisation pipeline that places electric-vehicle charging
stations on a real road network, built around two blocks of theory:

    Unit 1  Linear Algebra   -> incidence matrices, linear transformations,
                                eigenvectors/eigenvalues, SVD, Gram-Schmidt
    Unit 4  Network Science  -> graph models, shortest paths, centrality,
                                communities, hubs and core decomposition

Module map
----------
config      run parameters
network     Unit 4.1  road network graph from OpenStreetMap
demand      charging-demand surface from OSM points of interest
traffic     synthetic-but-grounded traffic sensor panel (input to the SVD)
linalg      Unit 1.1-1.4  incidence, power iteration, SVD, Gram-Schmidt
centrality  Unit 4.4  five centrality measures + combined priority score
communities Unit 4.2/4.5/4.6  Louvain communities, small-world, k-core hubs
coverage    Unit 4.3  shortest-path coverage metrics
optimise    facility-location models (greedy, ILP, p-median, Pareto front)
advanced    spectral analysis, perturbation theory, temporal, resilience
dashboard   interactive Folium map + Plotly report
"""

__version__ = "1.0.0"
__all__ = [
    "config", "network", "demand", "traffic", "linalg", "centrality",
    "communities", "coverage", "optimise", "advanced", "dashboard",
]
