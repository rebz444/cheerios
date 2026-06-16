import numpy as np
from typing import Dict, List, Tuple, Optional

# ================================================================
# ===  Utility functions  ========================================
# ================================================================

def make_time_bins(trial_start: float, trial_end: float, dt: float) -> np.ndarray:
    """Return bin edges from start to end with step dt."""
    return np.arange(trial_start, trial_end + 1e-9, dt)


def boxcar_mask(bin_edges: np.ndarray, start: float, end: float) -> np.ndarray:
    """Binary mask active between start and end times."""
    return ((bin_edges[:-1] >= start) & (bin_edges[:-1] < end)).astype(float)


def delta_regressor(bin_edges: np.ndarray, event_time: float) -> np.ndarray:
    """Single-bin regressor at event_time."""
    idx = np.searchsorted(bin_edges, event_time, side="right") - 1
    x = np.zeros(bin_edges.size - 1, dtype=float)
    if 0 <= idx < x.size:
        x[idx] = 1.0
    return x


def raised_cosine_basis(n_basis: int, duration: float, dt: float, eps: float = 1e-8) -> np.ndarray:
    """Return T x n_basis raised cosine basis spanning [0, duration]."""
    t = np.arange(0, duration + eps, dt)
    centers = np.linspace(0, duration, n_basis)
    width = duration / (n_basis - 1) if n_basis > 1 else duration
    B = []
    for c in centers:
        x = (t - c) / (width + eps) * np.pi
        b = (np.cos(np.clip(x, -np.pi, np.pi)) + 1.0) / 2.0
        b[(t < c - width) | (t > c + width)] = 0.0
        B.append(b)
    return np.stack(B, axis=1)  # T x n_basis


def orthogonalize(target: np.ndarray, design_cols: np.ndarray) -> np.ndarray:
    """Remove the subspace of design_cols from target (Gram–Schmidt projection)."""
    if design_cols.ndim == 1:
        A = design_cols.reshape(-1, 1)
    else:
        A = design_cols
    beta, *_ = np.linalg.lstsq(A, target, rcond=None)
    proj = A @ beta
    return target - proj


# ================================================================
# ===  Epoch and basis builders  ================================
# ================================================================

def build_epoch_masks(bin_edges: np.ndarray, bg_start: float, bg_end: float,
                      wait_end: float, cons_duration: float = 3.0) -> Dict[str, np.ndarray]:
    """Return binary masks for BG, WAIT, and CONSUMPTION epochs."""
    BG = boxcar_mask(bin_edges, bg_start, bg_end)
    WAIT = boxcar_mask(bin_edges, bg_end, wait_end)
    CONS = boxcar_mask(bin_edges, wait_end, wait_end + cons_duration)
    return {"BG": BG, "WAIT": WAIT, "CONS": CONS}


def build_time_bases(bin_edges: np.ndarray, masks: Dict[str, np.ndarray],
                     n_basis_bg: int = 8, n_basis_wait: int = 8, dt: float = 0.02,
                     absolute_time: bool = True,
                     t_max_bg: float = 5.0, t_max_wait: float = 10.0) -> Dict[str, np.ndarray]:
    """Raised-cosine bases for BG and WAIT epochs.

    If absolute_time is True (default), the basis is built once on a FIXED
    grid [0, t_max] per epoch and evaluated at each bin's local time since
    epoch onset. This means basis_j(t) takes the SAME value across trials
    for the same absolute t — bins past t_max get clamped to the last grid
    point (rare; covers essentially all waits with t_max_wait=10s).

    If absolute_time is False, the legacy behavior is used: the basis is
    built per-trial on [0, epoch_duration_i] and stretched to fit. This
    makes basis_j(t) depend on the trial's total epoch duration —
    "duration-warped" — which lets prev_X × basis_j interactions soak up
    duration-shape variation and corrupted the M1_vs_M2 LRT calibration on
    high-pR² units (see 4c_shuffle_control on MOp6a u224 and the stratified
    diagnostic in 4c_wait_quartile_stratified for evidence).
    """
    T = bin_edges.size - 1
    Xbg = np.zeros((T, n_basis_bg))
    Xw = np.zeros((T, n_basis_wait))

    for epoch, n_basis, X, t_max in [
        ("BG", n_basis_bg, Xbg, t_max_bg),
        ("WAIT", n_basis_wait, Xw, t_max_wait),
    ]:
        idx = np.where(masks[epoch] > 0)[0]
        if len(idx) == 0:
            continue
        if absolute_time:
            # Build basis ONCE on the fixed [0, t_max] grid.
            basis = raised_cosine_basis(n_basis, t_max, dt)  # (~t_max/dt, n_basis)
            last_grid = basis.shape[0] - 1
            for i, k in enumerate(idx):
                t = i * dt          # time since epoch onset (absolute)
                grid_i = min(int(round(t / dt)), last_grid)
                X[k, :] = basis[grid_i, :]
        else:
            # Legacy: per-trial duration-warped basis (kept for comparison)
            local = np.arange(len(idx)) * dt
            basis = raised_cosine_basis(n_basis, local[-1] + dt, dt)
            for i, k in enumerate(idx):
                X[k, :] = basis[min(i, basis.shape[0] - 1), :]
        if epoch == "BG":
            Xbg = X
        else:
            Xw = X

    return {"B_BG": Xbg, "B_WAIT": Xw}


def build_bgwait_basis(bin_edges: np.ndarray, masks: Dict[str, np.ndarray],
                       n_basis_bgwait: int = 0, dt: float = 0.02,
                       B_bg: Optional[np.ndarray] = None,
                       B_wait: Optional[np.ndarray] = None) -> np.ndarray:
    """Raised-cosine basis spanning BG ∪ WAIT, orthogonalized to BG and WAIT bases.

    Returns T x n_basis_bgwait. If n_basis_bgwait == 0, returns (T x 0) empty array.
    """
    T = bin_edges.size - 1
    if n_basis_bgwait <= 0:
        return np.zeros((T, 0))

    mask_bgwait = ((masks["BG"] + masks["WAIT"]) > 0).astype(bool)
    idx = np.where(mask_bgwait)[0]
    Xbw = np.zeros((T, n_basis_bgwait))
    if len(idx) == 0:
        return Xbw

    # Local time within BG∪WAIT
    local = np.arange(len(idx)) * dt
    basis = raised_cosine_basis(n_basis_bgwait, local[-1] + dt, dt)  # (L x n)
    for i, k in enumerate(idx):
        Xbw[k, :] = basis[min(i, basis.shape[0]-1), :]

    # Zero-mean within BG∪WAIT to reduce intercept overlap
    Xbw[idx, :] -= Xbw[idx, :].mean(axis=0, keepdims=True)

    # Orthogonalize to existing BG and WAIT bases if provided
    if B_bg is not None and B_bg.size > 0:
        A = B_bg
    else:
        A = np.zeros((T, 0))
    if B_wait is not None and B_wait.size > 0:
        A = np.column_stack([A, B_wait]) if A.size else B_wait
    if A.size:
        # Column-wise projection removal
        for j in range(Xbw.shape[1]):
            Xbw[:, j] = orthogonalize(Xbw[:, j], A)

    return Xbw


def build_hazard(bin_edges: np.ndarray, masks: Dict[str, np.ndarray], tau: float = 3.0) -> np.ndarray:
    """Hazard function starting at WAIT onset."""
    T = bin_edges.size - 1
    idx_w = np.where(masks["WAIT"] > 0)[0]
    h = np.zeros(T)
    if len(idx_w) > 0:
        local_t = np.arange(len(idx_w)) * (bin_edges[1] - bin_edges[0])
        h_wait = 1.0 - np.exp(-local_t / tau)
        h[idx_w] = h_wait
    return h


def build_cue_terms(bin_edges: np.ndarray, bg_start: float, bg_end: float) -> Dict[str, np.ndarray]:
    """Cue-on, sustained, and cue-off regressors."""
    cue_on = delta_regressor(bin_edges, bg_start)
    cue_box = boxcar_mask(bin_edges, bg_start, bg_end)
    cue_off = delta_regressor(bin_edges, bg_end)
    return {"cue_on": cue_on, "cue_box": cue_box, "cue_off": cue_off}


def build_event_kernel(bin_edges: np.ndarray, event_times: List[float], kernel_ms: float = 120.0) -> np.ndarray:
    """Causal event kernel (short smoothing window)."""
    dt = (bin_edges[1] - bin_edges[0])
    T = bin_edges.size - 1
    x = np.zeros(T)
    for t in event_times:
        x += delta_regressor(bin_edges, t)
    taps = int(np.round((kernel_ms / 1000.0) / dt))
    taps = max(taps, 1)
    k = np.ones(taps) / taps
    xk = np.convolve(x, k, mode="full")[:T]
    return xk


def build_spike_history(spike_counts: np.ndarray, ms: float = 100.0,
                        dt: float = 0.02, n_taps: int = 5) -> np.ndarray:
    """Lagged spike history columns."""
    taps = int(np.round((ms / 1000.0) / dt))
    taps = max(taps, n_taps)
    cols = []
    for lag in np.linspace(1, taps, n_taps, dtype=int):
        col = np.roll(spike_counts, lag)
        col[:lag] = 0.0
        cols.append(col)
    return np.stack(cols, axis=1)  # T x n_taps


# ================================================================
# ===  Main design assembler  ===================================
# ================================================================

def assemble_design(
    bin_edges: np.ndarray,
    bg_start: float,
    bg_end: float,
    wait_end: float,
    lick_times_bg: List[float],
    lick_time_decision: Optional[float],
    lick_times_cons: List[float],
    outcome_rewarded: int,
    spike_counts_for_history: Optional[np.ndarray] = None,
    dt: float = 0.02,
    n_basis_bg: int = 8,
    n_basis_wait: int = 8,
    n_basis_bgwait: int = 0,
    hazard_tau: float = 3.0,
    include_spike_history: bool = True,
    drop_cue_box: bool = True,
    absolute_time_basis: bool = True,
    t_max_bg: float = 5.0,
    t_max_wait: float = 10.0,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Build full design matrix and return (X, column_names, debug_info)."""

    masks = build_epoch_masks(bin_edges, bg_start, bg_end, wait_end)
    bases = build_time_bases(bin_edges, masks, n_basis_bg, n_basis_wait, dt=dt,
                             absolute_time=absolute_time_basis,
                             t_max_bg=t_max_bg, t_max_wait=t_max_wait)
    B_bgwait = build_bgwait_basis(bin_edges, masks, n_basis_bgwait, dt=dt,
                                  B_bg=bases["B_BG"], B_wait=bases["B_WAIT"]) if n_basis_bgwait > 0 else np.zeros((bin_edges.size-1, 0))
    hazard = build_hazard(bin_edges, masks, tau=hazard_tau)

    # Orthogonalize hazard vs. WAIT bases
    h_ortho = orthogonalize(hazard, bases["B_WAIT"]) if np.any(bases["B_WAIT"]) else hazard.copy()

    cue = build_cue_terms(bin_edges, bg_start, bg_end)

    # Lick kernels
    x_lick_bg = build_event_kernel(bin_edges, lick_times_bg, kernel_ms=120.0) if lick_times_bg else np.zeros(bin_edges.size-1)
    x_lick_dec = build_event_kernel(bin_edges, [lick_time_decision], kernel_ms=120.0) if lick_time_decision is not None else np.zeros(bin_edges.size-1)
    x_lick_cons = build_event_kernel(bin_edges, lick_times_cons, kernel_ms=120.0) if lick_times_cons else np.zeros(bin_edges.size-1)

    # Outcome (during consumption)
    outcome = (masks["CONS"] * (1 if outcome_rewarded else 0)).astype(float)

    # Spike history
    if include_spike_history and spike_counts_for_history is not None:
        X_hist = build_spike_history(spike_counts_for_history, ms=100.0, dt=dt, n_taps=5)
        hist_names = [f"hist_{i+1}" for i in range(X_hist.shape[1])]
    else:
        X_hist = np.zeros((bin_edges.size-1, 0))
        hist_names = []

    # Stack all columns
    cols = [("cue_on", cue["cue_on"]), ("cue_off", cue["cue_off"]) ] if drop_cue_box else [
        ("cue_on", cue["cue_on"]),
        ("cue_box", cue["cue_box"]),
        ("cue_off", cue["cue_off"]),
    ]

    for j in range(bases["B_BG"].shape[1]):
        cols.append((f"bg_time_b{j+1}", bases["B_BG"][:, j]))
    for j in range(bases["B_WAIT"].shape[1]):
        cols.append((f"wait_time_b{j+1}", bases["B_WAIT"][:, j]))

    # BG+WAIT basis
    for j in range(B_bgwait.shape[1]):
        cols.append((f"bgwait_time_b{j+1}", B_bgwait[:, j]))

    cols.append(("hazard_ortho", h_ortho))
    cols += [
        ("lick_bg", x_lick_bg),
        ("lick_decision", x_lick_dec),
        ("lick_cons", x_lick_cons),
        ("outcome_rewarded", outcome),
    ]

    for j, name in enumerate(hist_names):
        cols.append((name, X_hist[:, j]))

    X = np.column_stack([c[1] for c in cols])
    names = [c[0] for c in cols]

    debug = {"masks": masks, "hazard_raw": hazard, "hazard_ortho": h_ortho,
             "B_BG": bases["B_BG"], "B_WAIT": bases["B_WAIT"], "B_BGWAIT": B_bgwait}
    return X, names, debug