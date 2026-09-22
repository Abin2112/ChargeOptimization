# Optimal Placement of EV Charging Stations

**Mathematics for Data Science — MSc Data Science (Div A), 2026–27**
Mathematical optimisation and network analysis applied to a real road network.

A working application, not a notebook of snippets: it downloads a real city's
road network, builds a charging-demand surface from real land use, and solves
the siting problem with the linear algebra of Unit 1 and the network science of
Unit 4. Every mathematical checkpoint in the brief is implemented from first
principles and then *checked against* an established implementation, so the code
demonstrates the mathematics and proves it was done correctly.

---

## Headline result — Andheri–Jogeshwari, Mumbai

| | Today | After the recommended build |
|---|---|---|
| Charging demand within 2 km of a station | **20.7 %** | **88.0 %** |
| Average drive to a charger | 6.62 km | **1.36 km** (−79 %) |
| 90th-percentile drive | 11.95 km | **2.09 km** |
| Access inequality (Gini) | 0.317 | **0.251** |
| Stations | 2 mapped in OSM | +12 new, ₹364 lakh |

The 12 sites are proven optimal for this candidate set: CBC's branch and bound
closed the problem in 1.4 s. The greedy solver reached 96.6 % of that optimum,
well above its guaranteed 63.2 %.

Whole pipeline: **~35 s** on a laptop once the network is cached.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python run_pipeline.py            # full study area
.venv/bin/python run_pipeline.py --quick    # small, fast smoke test
.venv/bin/python run_pipeline.py -v         # with progress logging
```

A different city needs no code change:

```bash
.venv/bin/python run_pipeline.py --place "Pune" \
    --center-lat 18.5204 --center-lon 73.8567 --radius-m 6000 \
    --n-new-stations 20 --coverage-radius-m 1500
```

The first run downloads OpenStreetMap data (a few minutes) and caches it in
`.cache/`; later runs reuse it.

### Outputs (`outputs/`)

| File | What it is |
|---|---|
| `map.html` | Interactive map: demand, priority, coverage before/after, ranked sites |
| `report.html` | Every figure, with a light/dark toggle and a data table |
| `recommended_stations.csv` | The recommendation, in build order |
| `node_analysis.csv` | Per-junction centrality, demand, community, coverage |
| `community_profile.csv`, `community_coverage.csv` | Neighbourhood-level equity |
| `hub_junctions.csv` | Hub ranking from centrality × core depth × demand |
| `pareto_front.csv`, `station_load.csv` | Budget trade-off, projected utilisation |
| `results.json` | Every metric printed to the console, machine-readable |

---

## How the syllabus maps onto the code

### Unit 1 — Linear algebra (`evopt/linalg.py`)

| § | Checkpoint | Implementation | How it is verified |
|---|---|---|---|
| 1.1 | Incidence matrix of the road network | `incidence_matrix` builds the oriented **B** (2480 × 3728) | `verify_incidence_laplacian` confirms **B Bᵀ = L** exactly (max error 0.0) and rank **n − c** |
| 1.1 | Linear transformation `flow = A × traffic` | `propagate` iterates **x → Ax** | Direction converges to the dominant eigenvector (drift → 4.1°), as Perron–Frobenius predicts |
| 1.2 | Eigenvector centrality | `power_iteration`, written from scratch with a Rayleigh quotient | `validate_power_iteration`: λ₁ matches SciPy's `eigsh` to **7 × 10⁻¹⁵**, eigenvector cosine **1.000000** |
| 1.3 | SVD traffic cleaning | `svd_denoise` + `soft_impute` for missing readings | Ground truth is known by construction: error **9.3 % → 1.6 %**, SNR **20.6 → 36.1 dB**, **83 % of noise removed**, 46× compression |
| 1.4 | Gram–Schmidt feature validation | `gram_schmidt` (modified, for stability) and `feature_independence` | Orthogonality error **0.0**; correctly flags the planted redundant feature |

### Unit 4 — Network science

| § | Checkpoint | Implementation |
|---|---|---|
| 4.1 | Road network from OSM | `network.py` — directed, weighted, **intersection-consolidated**, largest strongly connected component |
| 4.2 / 4.5 | Communities, small-world | `communities.py` — Louvain (modularity **0.94**, 43 neighbourhoods), σ and ω indices |
| 4.3 | Shortest-path coverage | `coverage.py` — all-pairs network distance, demand-weighted coverage, equity Gini |
| 4.4 | **All five** centrality measures + combined score | `centrality.py` — betweenness, closeness, degree, eigenvector, harmonic → 0–100 priority score |
| 4.6 | Hubs and core decomposition | `communities.py` — k-core numbers, hub score |

### Deliverables 5–8

`optimise.py` poses the Maximal Covering Location Problem and solves it two
ways — greedy (with its submodular guarantee) and exact ILP — then sweeps the
programme size to trace the cost/coverage frontier. `dashboard.py` renders the
map and the figure report.

### Bonus work

Spectral analysis of the Laplacian, matrix perturbation theory, temporal
networks, network resilience, integer linear programming and multi-objective
optimisation are all implemented (`advanced.py`, `optimise.py`). Multilayer
road + transit networks are the one bonus item **not** attempted.

---

## Findings worth defending in a viva

These are the places where the analysis pushed back on the obvious approach.

**1. Centring the traffic matrix decides whether SVD helps or hurts.**
Traffic counts are strictly positive with a large mean, so on the raw matrix
σ₁ ≈ 9 σ₂ and the first component encodes nothing but "how busy is this junction
on average". A 90 % energy rule then selects **rank 1** and denoising makes the
signal *worse* (−42 %). Removing each sensor's own mean first makes the SVD model
deviations, where both the regimes and the noise live. Same data, opposite
conclusion.

**2. A fixed energy threshold is the wrong rank rule here; Gavish–Donoho is not.**
Even after centring, a 0.90 energy rule still picks rank 1. The Gavish–Donoho
optimal hard threshold — derived from the noise floor itself rather than from an
arbitrary percentage — picks **rank 4**. Sweeping reconstruction error against the
known clean signal puts the true optimum at rank 3 (1.52 %), with rank 4 within
noise of it (1.55 %), while rank 1 gives 12.1 %. Gavish–Donoho lands on the
optimum without being told what fraction of variance to keep, which is why it is
the default; the energy rule is still computed and reported alongside it.

**3. Only 3 of the 5 injected traffic regimes are recoverable.**
The two weakest regimes sit below the noise floor at σ = 0.18 and cannot be
recovered at any rank. The honest report is "3–4 of 5", not "5 of 5".

**4. Eigenvector centrality is nearly meaningless on this network.**
Its inverse participation ratio gives an effective support of **11 junctions out
of 2,480** — the dominant eigenvector localises on one dense cluster, and its
median value across the network is ~10⁻¹⁴. The *ranking* of the other 2,469
junctions is floating-point noise. This is why it carries a modest weight in the
combined score, and it is a measured result, not an assumption.

**5. σ says this is a small world; σ is not trustworthy here.**
On a sparse graph an Erdős–Rényi null model has almost no triangles
(C_rand = 8.4 × 10⁻⁴), so C/C_rand explodes and σ = 42 would call almost any
clustered sparse network a small world. The bounded ω index is used for the
verdict instead (ω = −0.02), and both are reported.

**6. First-order perturbation theory works — within a stated range.**
Predicting the centrality shift from a road capacity cut is accurate to **0.22 %**
at 1 % severity and **1.1 %** at 5 %, then fails outright at 10 % because a
different eigenvalue overtakes the first. That **level crossing is detected and
flagged** rather than silently producing a wrong answer.

**7. Coverage and capacity are different constraints.**
12 sites meet the 2 km coverage standard, but 5 of the 14 stations then run above
100 % of assumed capacity. The fix is more bays at those sites, not more sites —
a distinction a coverage-only model cannot make.

**8. Gram–Schmidt and VIF disagree, and both are right.**
Given two perfectly collinear features, Gram–Schmidt keeps the first and flags
the second (order matters, by design), while VIF flags *both* at ~10⁹ because
each is predictable from the other. Ordering the feature list is therefore a
modelling decision, and the most-wanted features are listed first.

---

## Data and honesty about it

**Real, measured data**

- Road network — OpenStreetMap via OSMnx (2,480 intersections, 1,274 km)
- Land use — 12,439 OSM points of interest (amenities, shops, offices,
  buildings, land-use polygons) driving the demand surface
- Existing charging stations — OSM `amenity=charging_station`

**Simulated, and labelled as such everywhere it is used**

The hourly traffic panel (600 sensors × 336 hours) is **simulated**. No public
hourly detector feed exists for this area. It is generated from a known low-rank
structure — commute, errand, freight and weekend regimes mixed per junction by
road class and surrounding land use, then corrupted with sensor noise and 4 %
dropouts. Because the clean signal is known by construction, the SVD stage can be
*validated* rather than merely performed. Swapping in a real feed means replacing
`traffic.simulate_traffic_panel`; nothing downstream changes.

**Stated modelling assumptions** — POI demand weights (`demand.WEIGHTS`),
station capacity, land-cost premium and the centrality weights are all assumptions
collected in `config.py` and `demand.py` so they can be challenged and changed.

---

## Layout

```
evopt/
  config.py       every tunable parameter, serialised beside the results
  network.py      4.1  OSM -> directed weighted graph of real intersections
  demand.py       charging demand from OSM land use
  traffic.py      the simulated sensor panel (input to the SVD)
  linalg.py       1.1-1.4  incidence, power iteration, SVD, Gram-Schmidt
  centrality.py   4.4  five centrality measures + combined priority score
  communities.py  4.2/4.5/4.6  Louvain, small-world indices, k-core hubs
  coverage.py     4.3  shortest-path coverage, equity, capacity
  optimise.py     MCLP greedy + ILP, p-median, cost model, Pareto front
  advanced.py     spectral, perturbation, temporal, resilience
  dashboard.py    the map and the figure report
  viz_theme.py    the validated colour palette
run_pipeline.py   the CLI that runs all of it
```

### Performance notes

Closeness, harmonic and betweenness all need all-pairs shortest paths. Rather
than three Python-level Dijkstra sweeps, the distance matrix is computed once
through SciPy's compiled `csgraph.dijkstra` (**0.4 s** for 2,480 × 2,480) and the
measures are read off it; exact weighted betweenness then takes 10 s.
Intersection consolidation cuts 8,480 raw OSM vertices to 2,480 real junctions,
which is both a better model and a 3× speed-up.

### Charts

The palette is validated rather than eyeballed: OKLab ΔE with Machado (2009) CVD
simulation gives worst adjacent-pair ΔE **9.1** (colour-blind) and **19.6**
(normal vision) against floors of 8 and 15. Status colours always ship with a
text label, every chart has a hover layer, the recommendations also appear as a
table, and dark mode is a selected set of steps rather than an inverted palette.
