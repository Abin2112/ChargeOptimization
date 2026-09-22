"""Unit 1 - Linear algebra, implemented rather than imported.

Every routine here that the syllabus names as a checkpoint is written from
first principles and then *checked against* an established implementation, so
the code both demonstrates the mathematics and proves it was done correctly.

  1.1  Linear transformations and systems  ->  incidence_matrix, propagate
  1.2  Eigenvectors and eigenvalues        ->  power_iteration
  1.3  Singular value decomposition        ->  svd_denoise, soft_impute
  1.4  Gram-Schmidt orthogonalisation      ->  gram_schmidt, feature_independence
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("evopt.linalg")


# =========================================================================== #
# 1.1  Linear transformations and systems
# =========================================================================== #
def incidence_matrix(U: nx.Graph, weighted: bool = False) -> Tuple[sp.csr_matrix, List, List]:
    """Oriented incidence matrix B of an undirected graph.

    For each edge e = {i, j} an orientation is fixed arbitrarily and

        B[i, e] = -sqrt(w_e)     (tail)
        B[j, e] = +sqrt(w_e)     (head)

    The defining identity is B @ B.T == L, the graph Laplacian. That identity
    is what makes the incidence matrix useful: it factors the Laplacian, so
    every quadratic form x'Lx is a sum of squared differences across edges,
    which is exactly "how unevenly is this quantity spread over the network".

    Returns (B, nodes, edges).
    """
    nodes = sorted(U.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    edges = list(U.edges())

    rows, cols, vals = [], [], []
    for e, (u, v) in enumerate(edges):
        w = float(U[u][v].get("affinity", 1.0)) if weighted else 1.0
        s = np.sqrt(w)
        rows += [index[u], index[v]]
        cols += [e, e]
        vals += [-s, +s]

    B = sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(edges)))
    return B, nodes, edges


def verify_incidence_laplacian(U: nx.Graph, weighted: bool = False) -> Dict:
    """Checkpoint 1.1: confirm B @ B.T reproduces the Laplacian."""
    B, nodes, edges = incidence_matrix(U, weighted=weighted)
    L_from_B = (B @ B.T).toarray()
    weight = "affinity" if weighted else None
    L_ref = nx.laplacian_matrix(U, nodelist=nodes, weight=weight).toarray().astype(float)

    err = float(np.abs(L_from_B - L_ref).max())
    # Rank of B is n - c, where c is the number of connected components.
    n_comp = nx.number_connected_components(U)
    return {
        "shape_B": tuple(B.shape),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "max_abs_error_BBt_vs_L": err,
        "identity_holds": bool(err < 1e-8),
        "rank_B_expected": len(nodes) - n_comp,
        "n_connected_components": int(n_comp),
        "row_sums_zero": bool(np.abs(B.sum(axis=0)).max() < 1e-12),
    }


def propagate(A: np.ndarray, x0: np.ndarray, steps: int = 12,
              normalise: bool = True) -> Dict:
    """Apply the linear transformation x -> A x repeatedly.

    Physically: if x holds vehicles waiting at each intersection and A encodes
    where traffic from one intersection goes next, then A @ x is the flow one
    step later. Iterating shows the network's long-run loading, which by the
    Perron-Frobenius theorem aligns with the dominant eigenvector - the same
    object eigenvector centrality measures.
    """
    x = np.asarray(x0, dtype=float).copy()
    if normalise:
        x /= np.linalg.norm(x) or 1.0
    history, ratios = [x.copy()], []
    for _ in range(steps):
        y = A @ x
        nrm = np.linalg.norm(y)
        if nrm == 0:
            break
        ratios.append(float(nrm / (np.linalg.norm(x) or 1.0)))
        x = y / nrm if normalise else y
        history.append(x.copy())
    H = np.array(history)
    # Angle between successive iterates: how fast the direction settles.
    drift = [float(np.degrees(np.arccos(np.clip(
        H[i] @ H[i + 1] / ((np.linalg.norm(H[i]) * np.linalg.norm(H[i + 1])) or 1.0), -1, 1))))
        for i in range(len(H) - 1)]
    return {"history": H, "final": x, "growth_ratios": ratios, "angle_drift_deg": drift}


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> Dict:
    """Least-squares solve of A x = b, reporting how trustworthy it is."""
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    cond = float(sv.max() / sv.min()) if sv.min() > 0 else np.inf
    return {
        "x": x,
        "rank": int(rank),
        "condition_number": cond,
        "residual_norm": float(np.linalg.norm(A @ x - b)),
        "is_full_rank": bool(rank == min(A.shape)),
    }


# =========================================================================== #
# 1.2  Eigenvectors and eigenvalues
# =========================================================================== #
def power_iteration(A, tol: float = 1e-10, max_iter: int = 5000,
                    x0: Optional[np.ndarray] = None) -> Dict:
    """Dominant eigenpair by power iteration (Unit 1.2, from scratch).

    Convergence rate is governed by the spectral gap |lambda_2 / lambda_1|:
    the closer the second eigenvalue, the slower the iteration settles.
    """
    n = A.shape[0]
    x = np.ones(n) / np.sqrt(n) if x0 is None else np.asarray(x0, float) / np.linalg.norm(x0)

    lam, deltas, lams = 0.0, [], []
    it = 0
    for it in range(1, max_iter + 1):
        y = A @ x
        nrm = np.linalg.norm(y)
        if nrm == 0:
            break
        x_new = y / nrm
        # Rayleigh quotient converges quadratically for symmetric A.
        lam = float(x_new @ (A @ x_new))
        delta = float(np.linalg.norm(x_new - x))
        deltas.append(delta)
        lams.append(lam)
        x = x_new
        if delta < tol:
            break

    # Perron-Frobenius: for a connected, non-negative A the dominant vector
    # can be taken entrywise positive. Fix the sign so results are comparable.
    if x.sum() < 0:
        x = -x
    return {
        "eigenvalue": lam,
        "eigenvector": x,
        "iterations": it,
        "converged": bool(deltas and deltas[-1] < tol),
        "delta_history": deltas,
        "eigenvalue_history": lams,
    }


def validate_power_iteration(A_sparse) -> Dict:
    """Checkpoint 1.2: our power iteration vs a library eigensolver."""
    from scipy.sparse.linalg import eigsh

    A = A_sparse.toarray() if sp.issparse(A_sparse) else np.asarray(A_sparse)
    mine = power_iteration(A)

    k = min(2, A.shape[0] - 1)
    vals, vecs = eigsh(sp.csr_matrix(A).astype(float), k=k, which="LA")
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ref_vec = vecs[:, 0]
    if ref_vec.sum() < 0:
        ref_vec = -ref_vec

    cos = float(np.abs(mine["eigenvector"] @ ref_vec) /
                (np.linalg.norm(mine["eigenvector"]) * np.linalg.norm(ref_vec)))
    gap = float(abs(vals[1] / vals[0])) if k > 1 and vals[0] != 0 else np.nan
    return {
        "lambda_power_iteration": mine["eigenvalue"],
        "lambda_reference": float(vals[0]),
        "lambda_abs_error": float(abs(mine["eigenvalue"] - vals[0])),
        "eigenvector_cosine_similarity": cos,
        "iterations": mine["iterations"],
        "converged": mine["converged"],
        "spectral_gap_ratio": gap,
        "agrees": bool(abs(mine["eigenvalue"] - vals[0]) < 1e-6 and cos > 1 - 1e-6),
    }


# =========================================================================== #
# 1.3  Singular value decomposition
# =========================================================================== #
def rank_by_energy(s: np.ndarray, energy: float = 0.90) -> int:
    """Smallest k whose components carry `energy` of the total variance."""
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(e, energy) + 1)


def gavish_donoho_threshold(X: np.ndarray, s: np.ndarray) -> float:
    """The singular-value cut-off itself (see `gavish_donoho_rank`)."""
    m, n = X.shape
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return float(omega * np.median(s))


def gavish_donoho_rank(X: np.ndarray, s: np.ndarray) -> int:
    """Optimal hard threshold for singular values, unknown noise level.

    Gavish & Donoho (2014), "The Optimal Hard Threshold for Singular Values
    is 4/sqrt(3)". For an m x n matrix with aspect ratio beta = m/n <= 1 the
    threshold is omega(beta) * median(s), with the published approximation
    omega(beta) ~ 0.56 beta^3 - 0.95 beta^2 + 1.82 beta + 1.43.

    Unlike an energy rule this needs no arbitrary cut-off: it estimates where
    the singular values stop being signal and start being noise.
    """
    m, n = X.shape
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    tau = omega * np.median(s)
    return int(max(1, np.sum(s > tau)))


def soft_impute(X: np.ndarray, rank: int, max_iter: int = 60,
                tol: float = 1e-5) -> np.ndarray:
    """Fill missing readings using the matrix's own low-rank structure.

    Start from row means, then alternate: factor with SVD, truncate to `rank`,
    write the reconstruction back into the missing cells only. Because the
    panel really is low rank, the gaps get filled from the regimes visible in
    the sensors that *were* reporting.
    """
    missing = np.isnan(X)
    if not missing.any():
        return X.copy()

    row_mean = np.nanmean(X, axis=1)
    row_mean = np.where(np.isfinite(row_mean), row_mean, np.nanmean(X))
    Z = np.where(missing, row_mean[:, None], X)

    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        k = min(rank, len(s))
        Z_new = (U[:, :k] * s[:k]) @ Vt[:k]
        Z_next = np.where(missing, Z_new, X)
        change = np.linalg.norm(Z_next - Z) / (np.linalg.norm(Z) or 1.0)
        Z = Z_next
        if change < tol:
            break
    return Z


def svd_denoise(X: np.ndarray, rank: Optional[int] = None, energy: float = 0.96,
                X_true: Optional[np.ndarray] = None, center: str = "rows",
                rule: str = "gavish_donoho") -> Dict:
    """Checkpoint 1.3: SVD-based traffic cleaning.

    Decomposes X = U S V', keeps the leading k components and rebuilds. The
    discarded tail is where independent sensor noise lives: it spreads evenly
    over all directions, while the real signal concentrates in a few.

    Centring is not cosmetic here. Traffic counts are strictly positive and
    every sensor has a large mean level, so on the raw matrix the first
    singular component simply encodes "how busy is this junction on average"
    and swamps everything else - on this study area sigma_1 is roughly nine
    times sigma_2, and an energy rule then selects rank 1 and discards the
    regimes we actually want. Removing each sensor's own mean first makes the
    SVD model *deviations from normal*, which is where both the traffic
    regimes and the sensor noise live. The means are added back on
    reconstruction, so no information is lost.
    """
    had_missing = bool(np.isnan(X).any())
    frac_missing = float(np.isnan(X).mean())

    # A provisional rank is needed before imputation; refine it after.
    if had_missing:
        prelim = np.where(np.isnan(X), np.nanmean(X, axis=1, keepdims=True), X)
        s0 = np.linalg.svd(prelim - prelim.mean(axis=1, keepdims=True), compute_uv=False)
        Xf = soft_impute(X, rank or rank_by_energy(s0, energy))
    else:
        Xf = X.copy()

    if center == "rows":
        mu = Xf.mean(axis=1, keepdims=True)
    elif center == "global":
        mu = np.full((Xf.shape[0], 1), Xf.mean())
    else:
        mu = np.zeros((Xf.shape[0], 1))
    Xc = Xf - mu

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Sign convention. An SVD is only determined up to a joint sign flip of
    # each (u_i, v_i) pair, so a component describing the morning commute is
    # just as likely to come back as a dip as a peak. Flipping each pair so
    # that the largest-magnitude entry of v_i is positive costs nothing
    # mathematically - the product u_i s_i v_i' is unchanged - and makes the
    # components readable as traffic profiles.
    flip = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
    flip[flip == 0] = 1.0
    U = U * flip
    Vt = Vt * flip[:, None]

    k_energy = rank_by_energy(s, energy)
    k_gd = gavish_donoho_rank(Xc, s)
    if rank:
        k = int(rank)
    else:
        k = int(k_gd if rule == "gavish_donoho" else k_energy)

    X_hat = mu + (U[:, :k] * s[:k]) @ Vt[:k]
    explained = (s ** 2) / np.sum(s ** 2)

    # Storage: k singular triplets plus one mean per sensor.
    stored = k * (Xc.shape[0] + Xc.shape[1] + 1) + Xc.shape[0]

    out = {
        "U": U, "s": s, "Vt": Vt, "row_means": mu.ravel(),
        "X_filled": Xf, "X_centered": Xc, "X_denoised": X_hat,
        "centering": center,
        "rank_used": k,
        "rank_rule": "fixed" if rank else rule,
        "rank_energy_rule": int(k_energy),
        "rank_gavish_donoho": int(k_gd),
        "gd_threshold": gavish_donoho_threshold(Xc, s),
        "energy_threshold": energy,
        "explained_variance_ratio": explained,
        "cumulative_energy": np.cumsum(explained),
        "energy_captured_at_k": float(np.cumsum(explained)[k - 1]),
        "had_missing": had_missing,
        "fraction_missing": frac_missing,
        "compression_ratio": float(X.size / max(stored, 1)),
    }

    # When the clean signal is known, quantify how much noise was removed.
    if X_true is not None:
        nt = np.linalg.norm(X_true)
        obs = np.where(np.isnan(X), Xf, X)
        err_before = float(np.linalg.norm(obs - X_true) / nt)
        err_after = float(np.linalg.norm(X_hat - X_true) / nt)
        out.update({
            "relative_error_before": err_before,
            "relative_error_after": err_after,
            "noise_removed_pct": float(100.0 * (1 - err_after / err_before)) if err_before else 0.0,
            "snr_before_db": float(20 * np.log10(nt / (np.linalg.norm(obs - X_true) or 1e-12))),
            "snr_after_db": float(20 * np.log10(nt / (np.linalg.norm(X_hat - X_true) or 1e-12))),
        })
    return out


def rank_sweep(X: np.ndarray, X_true: np.ndarray, max_rank: int = 15,
               center: str = "rows") -> pd.DataFrame:
    """Reconstruction error as a function of retained rank.

    The U-shape this traces out is the bias-variance trade-off made visible:
    too few components and real structure is thrown away, too many and the
    noise is faithfully reproduced.
    """
    rows = []
    for k in range(1, max_rank + 1):
        r = svd_denoise(X, rank=k, X_true=X_true, center=center)
        rows.append({"rank": k,
                     "relative_error": r["relative_error_after"],
                     "cumulative_energy": float(r["cumulative_energy"][k - 1]),
                     "snr_db": r["snr_after_db"]})
    return pd.DataFrame(rows)


def svd_sensor_features(svd: Dict, k: Optional[int] = None) -> np.ndarray:
    """Per-sensor scores on the leading regimes: the U*S block.

    This is the dimensionality reduction the syllabus asks for - hundreds of
    hourly columns replaced by a handful of coordinates per monitoring point.
    """
    k = k or svd["rank_used"]
    return svd["U"][:, :k] * svd["s"][:k]


# =========================================================================== #
# 1.4  Gram-Schmidt orthogonalisation
# =========================================================================== #
def gram_schmidt(M: np.ndarray, modified: bool = True,
                 tol: float = 1e-10) -> Dict:
    """QR by (modified) Gram-Schmidt, keeping the diagnostics.

    R[j, j] is the length of the part of column j that is *not* explained by
    the columns before it. Divided by the column's own norm it becomes a
    direct, scale-free measure of how much new information that feature adds.
    Classical Gram-Schmidt is offered for comparison but loses orthogonality
    on ill-conditioned input; modified GS is the default for that reason.
    """
    A = np.asarray(M, dtype=float)
    n, p = A.shape
    Q = np.zeros((n, p))
    R = np.zeros((p, p))
    residual_ratio, dependent = [], []

    for j in range(p):
        v = A[:, j].copy()
        col_norm = np.linalg.norm(A[:, j])
        if modified:
            for i in range(j):
                R[i, j] = Q[:, i] @ v          # project against the updated vector
                v -= R[i, j] * Q[:, i]
        else:
            for i in range(j):
                R[i, j] = Q[:, i] @ A[:, j]    # project against the original
                v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        ratio = float(R[j, j] / col_norm) if col_norm > 0 else 0.0
        residual_ratio.append(ratio)
        if ratio < 1e-6:
            dependent.append(j)
            Q[:, j] = 0.0
        else:
            Q[:, j] = v / R[j, j]

    Qc = Q[:, [j for j in range(p) if j not in dependent]]
    orth_err = float(np.abs(Qc.T @ Qc - np.eye(Qc.shape[1])).max()) if Qc.size else 0.0
    recon_err = float(np.linalg.norm(Q @ R - A) / (np.linalg.norm(A) or 1.0))
    return {
        "Q": Q, "R": R,
        "residual_ratio": np.array(residual_ratio),
        "dependent_columns": dependent,
        "rank": int(p - len(dependent)),
        "orthogonality_error": orth_err,
        "reconstruction_error": recon_err,
    }


def feature_independence(F: pd.DataFrame, redundancy_threshold: float = 0.15) -> Dict:
    """Checkpoint 1.4: validate and orthogonalise the feature set.

    Features are standardised first so the comparison is about direction, not
    units. A feature whose Gram-Schmidt residual falls below the threshold is
    almost entirely reconstructible from the features already accepted -
    exactly the "population density vs number of residents" case - and is
    flagged for removal.

    Feature order matters: columns are processed left to right, so the caller
    should list the features it most wants to keep first.
    """
    names = list(F.columns)
    X = F.values.astype(float)
    X = X - X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = X / sd

    gs = gram_schmidt(Z, modified=True)
    ratios = gs["residual_ratio"]

    redundant = [names[j] for j, r in enumerate(ratios) if r < redundancy_threshold]
    keep = [nm for nm in names if nm not in redundant]

    corr = pd.DataFrame(np.corrcoef(Z, rowvar=False), index=names, columns=names)
    sv = np.linalg.svd(Z, compute_uv=False)
    cond = float(sv.max() / sv.min()) if sv.min() > 1e-12 else float("inf")

    # Variance inflation factor: 1 / (1 - R^2) of each feature on the others.
    vif = {}
    for j, nm in enumerate(names):
        others = np.delete(Z, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        r2 = 1.0 - np.sum((Z[:, j] - others @ beta) ** 2) / np.sum(Z[:, j] ** 2)
        vif[nm] = float(1.0 / max(1.0 - r2, 1e-9))

    Q_keep = pd.DataFrame(
        gram_schmidt(Z[:, [names.index(nm) for nm in keep]])["Q"],
        index=F.index, columns=[f"{nm}_orth" for nm in keep])

    return {
        "features": names,
        "residual_ratio": pd.Series(ratios, index=names),
        "redundant_features": redundant,
        "retained_features": keep,
        "correlation": corr,
        "condition_number": cond,
        "vif": vif,
        "orthogonal_features": Q_keep,
        "orthogonality_error": gs["orthogonality_error"],
        "reconstruction_error": gs["reconstruction_error"],
        "effective_rank": gs["rank"],
    }
