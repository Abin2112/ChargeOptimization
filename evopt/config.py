"""Run configuration.

Every tunable number in the pipeline lives here so that an experiment is
described by one object that can be serialised next to its results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Study area (Unit 4.1 - network elements)
    # ------------------------------------------------------------------ #
    area_name: str = "Andheri-Jogeshwari, Mumbai"
    center_lat: float = 19.1136
    center_lon: float = 72.8697
    radius_m: int = 5000
    network_type: str = "drive"          # drive | drive_service | all
    # Raw OSM encodes one junction as several nodes; merging them keeps the
    # graph an honest model of *intersections* rather than of GPS vertices.
    consolidate_tolerance_m: float = 30.0

    # ------------------------------------------------------------------ #
    # Traffic sensor panel (input matrix for Unit 1.3 - SVD)
    # ------------------------------------------------------------------ #
    n_sensors: int = 600                 # monitoring points on the network
    n_timesteps: int = 336               # 14 days x 24 hourly readings
    true_rank: int = 5                   # latent traffic regimes in the ground truth
    noise_sigma: float = 0.18            # sensor noise, fraction of signal scale
    dropout_rate: float = 0.04           # share of readings lost by the sensors
    svd_rank: Optional[int] = None       # set to force a specific truncation rank
    # "gavish_donoho" derives the cut-off from the noise floor itself and is
    # the default; "energy" uses the fixed share below, which on strongly
    # mean-dominated traffic data tends to under-select.
    svd_rank_rule: str = "gavish_donoho"  # gavish_donoho | energy
    svd_energy: float = 0.96             # only used by the energy rule

    # ------------------------------------------------------------------ #
    # Centrality (Unit 4.4)
    # ------------------------------------------------------------------ #
    betweenness_k: Optional[int] = None  # None -> exact when the graph is small
    exact_betweenness_max_nodes: int = 3500
    centrality_weights: Tuple[float, ...] = (0.30, 0.20, 0.15, 0.15, 0.20)
    #                        betweenness, closeness, degree, eigenvector, harmonic

    # ------------------------------------------------------------------ #
    # Demand surface
    # ------------------------------------------------------------------ #
    demand_smoothing_m: float = 400.0    # kernel bandwidth over the node cloud
    ev_adoption_rate: float = 0.06       # share of vehicles that are electric

    # ------------------------------------------------------------------ #
    # Facility location
    # ------------------------------------------------------------------ #
    n_new_stations: int = 12
    coverage_radius_m: float = 2000.0    # network distance a driver accepts
    n_candidates: int = 250              # shortlist drawn from the priority score
    candidate_min_separation_m: float = 350.0
    # Demand units one site can absorb per day. Demand totals 10,000 units
    # across the study area, so 650 represents a site of roughly six fast
    # chargers taking about 6.5% of area-wide daily demand. Coverage and
    # capacity are separate constraints and the run reports which one binds.
    station_capacity: float = 650.0
    use_ilp: bool = True                 # exact MCLP via CBC, greedy otherwise
    ilp_time_limit_s: int = 180
    pareto_p_values: Tuple[int, ...] = (2, 4, 6, 8, 10, 12, 16, 20, 25, 30)

    # Cost model: a site on a busy, high-demand junction costs more to acquire.
    base_station_cost_lakh: float = 18.0
    land_premium_lakh: float = 22.0      # maximum premium on top of the base cost

    # ------------------------------------------------------------------ #
    # Bonus analyses
    # ------------------------------------------------------------------ #
    run_advanced: bool = True
    resilience_removals: int = 6
    temporal_windows: Tuple[str, ...] = ("morning_peak", "midday", "evening_peak", "night")

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #
    seed: int = 42
    outdir: str = "outputs"
    cache_dir: str = ".cache"
    quick: bool = False                  # small, fast run for smoke-testing

    # ------------------------------------------------------------------ #
    def apply_quick_mode(self) -> "Config":
        """Shrink the run so the whole pipeline finishes in a couple of minutes."""
        if not self.quick:
            return self
        self.radius_m = min(self.radius_m, 2000)
        self.n_sensors = 150
        self.n_timesteps = 168
        self.betweenness_k = 120
        self.n_candidates = 80
        self.n_new_stations = 6
        self.pareto_p_values = (2, 4, 6, 8, 12)
        self.resilience_removals = 3
        self.ilp_time_limit_s = 45
        return self

    # ------------------------------------------------------------------ #
    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_lat, self.center_lon)

    @property
    def out_path(self) -> Path:
        p = Path(self.outdir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def slug(self) -> str:
        """Short identifier used for cache filenames."""
        lat = f"{self.center_lat:.4f}".replace(".", "p").replace("-", "m")
        lon = f"{self.center_lon:.4f}".replace(".", "p").replace("-", "m")
        return f"{lat}_{lon}_{self.radius_m}_{self.network_type}"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else self.out_path / "run_config.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def from_namespace(cls, ns) -> "Config":
        """Build a Config from an argparse namespace, ignoring unset values."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(ns).items() if k in known and v is not None}
        return cls(**kwargs).apply_quick_mode()
