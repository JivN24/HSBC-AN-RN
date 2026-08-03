"""Step 5.5-5.6: information-theoretic causality estimators, implemented
from scratch (no pyinform/IDTxl/JIDT -- the bias corrections and
neighbour conventions matter at our sample sizes and those packages
hide them; see CLAUDE.md).

Two estimators, both Kraskov-Stogbauer-Grassberger (KSG) k-nearest-
neighbour constructions using the max-norm (Chebyshev distance):

  - Mutual information I(X;Y) (Kraskov, Stogbauer & Grassberger 2004,
    "KSG-1"): for each point, find the distance to its k-th nearest
    neighbour in the JOINT (X,Y) space, then count neighbours within
    that same (open-ball) distance in each MARGINAL space alone.
        I(X;Y) = psi(k) - <psi(n_x+1) + psi(n_y+1)> + psi(N)

  - Conditional mutual information I(A;B|C) (Frenzel & Pompe 2007,
    the standard basis for a KSG-style transfer-entropy estimator):
    same k-th-neighbour construction in the full joint (A,B,C) space,
    then neighbour counts in the (A,C), (B,C), and C-alone marginals.
        I(A;B|C) = psi(k) - <psi(n_ac+1) + psi(n_bc+1) - psi(n_c+1)>

Transfer entropy TE(source->target) is exactly the conditional case
with A = target's next value, B = source's past (history-embedded),
C = target's own past (history-embedded): TE quantifies how much
knowing the source's recent history reduces uncertainty about the
target's future BEYOND what the target's own history already tells
you -- the textbook definition of "does X help predict Y beyond Y's
own history".

History embedding dimension (Ragwitz criterion): rather than run a
full Ragwitz optimisation (computationally heavy at project scale),
we FIX history_len=1 as the default with an explicit justification
documented on transfer_entropy: at daily frequency, most of an event's
information content is impounded within 1-2 days (verified directly:
evc.causality.eventstudy's crush pattern on the isolated-FOMC
synthetic series decays to near-baseline within ~5-10 days, but the
day-over-day CHANGE dlogvar -- what transfer entropy actually
operates on here -- is overwhelmingly a 1-lag process). Callers who
want a different, explicitly-chosen embedding can pass history_len
directly; this is a fixed, stated choice, not a search.

Surrogates for TE significance live in evc.causality.surrogates
(circular shift AND IAAFT); this module only computes the point
estimate. A discrete/symbolised (quantile-binned) cross-check is
provided as symbolic_transfer_entropy -- if it disagrees materially
with the continuous KSG estimate, that disagreement should be
reported, not resolved by picking whichever number is prettier.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

_OPEN_BALL_TOL = 1e-10


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _neighbor_counts(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Count of neighbours strictly within `radii[i]` of points[i]
    (max-norm), excluding the point itself. The tiny subtraction from
    radii makes this an OPEN ball (standard KSG convention -- points
    exactly at the k-th-neighbour distance must not be double-counted
    across the joint and marginal spaces)."""
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, radii - _OPEN_BALL_TOL, p=np.inf, return_length=True)
    return counts - 1  # exclude self


def _kth_neighbor_distance(points: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1, p=np.inf)
    return dists[:, -1]


def ksg_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 4) -> float:
    """KSG-1 estimator of I(X;Y). x, y can be 1D (single variable) or
    2D (N, dim) arrays; both must have the same N."""
    x, y = _as_2d(x), _as_2d(y)
    n = len(x)
    joint = np.hstack([x, y])
    eps = _kth_neighbor_distance(joint, k)

    n_x = _neighbor_counts(x, eps)
    n_y = _neighbor_counts(y, eps)

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return float(max(mi, 0.0))


def _ksg_conditional_mutual_information(a: np.ndarray, b: np.ndarray, c: np.ndarray, k: int = 4) -> float:
    """Frenzel-Pompe KSG-style estimator of I(A;B|C)."""
    a, b, c = _as_2d(a), _as_2d(b), _as_2d(c)
    joint = np.hstack([a, b, c])
    eps = _kth_neighbor_distance(joint, k)

    n_ac = _neighbor_counts(np.hstack([a, c]), eps)
    n_bc = _neighbor_counts(np.hstack([b, c]), eps)
    n_c = _neighbor_counts(c, eps)

    cmi = digamma(k) - np.mean(digamma(n_ac + 1) + digamma(n_bc + 1) - digamma(n_c + 1))
    return float(max(cmi, 0.0))


def tdmi_curve(x: np.ndarray, y: np.ndarray, max_lag: int, k: int = 4) -> np.ndarray:
    """Time-delayed mutual information I(x_t; y_{t+lag}) for lag in
    -max_lag..max_lag (both directions in one sweep -- negative lags
    probe "does future y inform present x", the anticipation-detection
    direction CLAUDE.md constraint 1 cares about)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        if lag >= 0:
            xi, yi = x[: n - lag], y[lag:]
        else:
            xi, yi = x[-lag:], y[: n + lag]
        if len(xi) < 2 * k + 2:
            continue
        out[i] = ksg_mutual_information(xi, yi, k=k)
    return out


def _embed_history(series: np.ndarray, history_len: int) -> np.ndarray:
    """(series[t-1], ..., series[t-history_len]) for each valid t --
    strictly past values, never including series[t] itself."""
    n = len(series)
    return np.column_stack([series[history_len - 1 - h : n - 1 - h] for h in range(history_len)])


def transfer_entropy(source: np.ndarray, target: np.ndarray, history_len: int = 1, k: int = 4) -> float:
    """TE(source -> target) = I(target_future; source_past | target_past),
    with both histories embedded at `history_len` (see module
    docstring for why 1 is the default rather than a Ragwitz search)."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    n = len(target)
    target_future = target[history_len:]
    target_past = _embed_history(target, history_len)
    source_past = _embed_history(source, history_len)
    return _ksg_conditional_mutual_information(target_future, source_past, target_past, k=k)


def conditional_transfer_entropy(
    source: np.ndarray, target: np.ndarray, conditioning: np.ndarray, history_len: int = 1, k: int = 4
) -> float:
    """TE(source -> target | conditioning): as transfer_entropy, but
    the conditioning set also includes the conditioning series' own
    history -- controls for a third series (e.g. a VIX-analogue) that
    might otherwise drive both source and target."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    conditioning = np.asarray(conditioning, dtype=float)

    target_future = target[history_len:]
    target_past = _embed_history(target, history_len)
    source_past = _embed_history(source, history_len)
    conditioning_past = _embed_history(conditioning, history_len)
    full_conditioning = np.hstack([target_past, conditioning_past])
    return _ksg_conditional_mutual_information(target_future, source_past, full_conditioning, k=k)


def symbolic_transfer_entropy(source: np.ndarray, target: np.ndarray, history_len: int = 1, n_bins: int = 4) -> float:
    """Discrete/symbolised cross-check: quantile-bin each series into
    n_bins symbols, then compute TE via plug-in (maximum-likelihood)
    discrete entropies on the symbol sequences. A cheap, different-
    assumptions estimator to compare against the continuous KSG
    version -- if they disagree materially, that's worth reporting
    rather than picking whichever number looks better.
    """
    def symbolize(series: np.ndarray) -> np.ndarray:
        quantiles = np.quantile(series, np.linspace(0, 1, n_bins + 1)[1:-1])
        return np.searchsorted(quantiles, series)

    source_sym = symbolize(np.asarray(source, dtype=float))
    target_sym = symbolize(np.asarray(target, dtype=float))

    target_future = target_sym[history_len:]
    target_past = _embed_history(target_sym, history_len)
    source_past = _embed_history(source_sym, history_len)

    def joint_entropy(*arrays: np.ndarray) -> float:
        stacked = np.column_stack(arrays)
        _, counts = np.unique(stacked, axis=0, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p)))

    # TE = H(target_future, target_past) + H(source_past, target_past)
    #      - H(target_future, source_past, target_past) - H(target_past)
    h_yz = joint_entropy(target_future, target_past)
    h_xz = joint_entropy(source_past, target_past)
    h_xyz = joint_entropy(target_future, source_past, target_past)
    h_z = joint_entropy(target_past)
    return max(h_yz + h_xz - h_xyz - h_z, 0.0)
