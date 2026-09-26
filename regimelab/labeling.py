"""Regime labelling with economically anchored, permutation-invariant labels.

The problem
-----------
An unsupervised mixture model has no notion of which component is "state 0".
Component indices are an artefact of initialisation, so re-fitting after one
extra day of data can permute every label.  The original notebook worked
around this by hand-mapping component numbers to economic names, and the
mapping silently went stale (its stress mask selects components ``[0, 2]``
while the comment beside it says ``1, 3``).

The fix
-------
Labels are never taken from the estimator's own indexing.  After fitting, each
component is scored on a **risk-appetite functional with signs fixed a
priori** -- equity trend and breadth positive, volatility negative, credit
spreads negative, yen carry positive -- computed from that component's mean in
standardised feature space.  Components are then re-indexed in ascending score
order, so canonical label ``0`` is always the most risk-averse state and
``K-1`` always the most risk-seeking one, for any seed, any K and any data
vintage.  The binary risk-on/risk-off split falls out of the sign of the same
score rather than a hand-maintained lookup table.

Causality
---------
``predict_proba`` returns *smoothed* (forward-backward) posteriors, which
condition on the whole sample.  That is appropriate for describing history but
is look-ahead if fed to a forecaster, so ``filtered_proba`` implements the
forward-only recursion and is what the forecasting pipeline consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Fixed economic sign convention
# --------------------------------------------------------------------------
# +1 : a higher standardised value of this feature means MORE risk appetite.
# -1 : a higher standardised value means LESS risk appetite.
# Features whose risk interpretation is genuinely ambiguous (curve slope,
# the level change in yields, the stock/bond correlation) are deliberately
# excluded from the score and used only to describe regimes afterwards.
RISK_BLOCKS: dict[str, dict[str, int]] = {
    "equity": {
        "SPX_rmom21": +1,
        "SPX_rmom63": +1,
        "SPX_dd252": +1,
        "RUT_rel_SPX_63": +1,
        "Nikkei_rmom63": +1,
    },
    "volatility": {
        "SPX_logvol21": -1,
        "SPX_volratio": -1,
        "VIX_log": -1,
    },
    "credit": {
        "Baa_log": -1,
        "Baa_chg63": -1,
    },
    "carry": {
        "USDJPY_rmom63": +1,
    },
}

CANONICAL_NAMES: dict[int, list[str]] = {
    2: ["Risk-off", "Risk-on"],
    3: ["Crisis / deleveraging", "Transition", "Risk-on expansion"],
    4: [
        "Crisis / deleveraging",
        "Stress / drawdown",
        "Recovery / mixed",
        "Risk-on expansion",
    ],
    5: [
        "Crisis / deleveraging",
        "Stress / drawdown",
        "Neutral / transition",
        "Recovery",
        "Risk-on expansion",
    ],
}


def risk_appetite_score(z: pd.DataFrame | np.ndarray, columns: list[str]) -> np.ndarray:
    """Block-equal-weighted signed average of standardised features.

    Blocks (equity, volatility, credit, carry) are weighted equally so that the
    score is not dominated by whichever block happens to contribute the most
    columns.  Zero means "average conditions over the estimation sample".
    """
    if isinstance(z, pd.DataFrame):
        arr, columns = z.values, list(z.columns)
    else:
        arr = np.asarray(z)
    idx = {c: i for i, c in enumerate(columns)}

    block_vals, used = [], []
    for block, spec in RISK_BLOCKS.items():
        present = [(idx[c], s) for c, s in spec.items() if c in idx]
        if not present:
            continue
        contrib = np.mean([s * arr[:, i] for i, s in present], axis=0)
        block_vals.append(contrib)
        used.append(block)
    if not block_vals:
        raise ValueError("no risk-appetite features present in the feature matrix")
    return np.mean(block_vals, axis=0)


@dataclass
class RegimeLabeler:
    """Fit-once regime labeller with canonical, economically ordered labels.

    Parameters
    ----------
    n_regimes
        Number of latent states.
    method
        ``"hmm"``   Gaussian HMM -- models persistence explicitly and yields
                    both smoothed and filtered state probabilities.
        ``"gmm"``   Gaussian mixture -- the original notebook's estimator,
                    retained for comparison.
        ``"score"`` Deterministic quantile buckets of the risk-appetite score.
                    A transparent baseline: if the latent-variable models do
                    not beat this, they are not earning their complexity.
    n_pca
        If not None, PCA is applied after standardisation.  ``None`` keeps the
        compact labelling features as-is, which is usually more interpretable.
    """

    n_regimes: int = 4
    method: str = "hmm"
    n_pca: float | int | None = None
    random_state: int = 42
    covariance_type: str = "full"
    n_init: int = 20
    init: str = "score"          # "score" (deterministic) | "random"
    risk_off_share_bounds: tuple[float, float] = (0.15, 0.55)

    scaler_: StandardScaler = field(init=False, default=None)
    pca_: PCA | None = field(init=False, default=None)
    model_: object = field(init=False, default=None)
    columns_: list[str] = field(init=False, default_factory=list)
    order_: np.ndarray = field(init=False, default=None)     # raw -> rank
    inverse_order_: np.ndarray = field(init=False, default=None)
    regime_scores_: np.ndarray = field(init=False, default=None)  # canonical order
    score_cuts_: np.ndarray = field(init=False, default=None)
    regime_shares_: np.ndarray = field(init=False, default=None)
    risk_off_mask_: np.ndarray = field(init=False, default=None)

    # -- internals ---------------------------------------------------------
    def _design(self, X: pd.DataFrame) -> np.ndarray:
        z = self.scaler_.transform(X[self.columns_].values)
        return self.pca_.transform(z) if self.pca_ is not None else z

    def _canonicalise(self, Xz: np.ndarray, raw_labels: np.ndarray) -> None:
        """Order components by risk appetite computed in standardised space."""
        k = self.n_regimes
        means = np.full((k, Xz.shape[1]), np.nan)
        for j in range(k):
            m = raw_labels == j
            if m.any():
                means[j] = Xz[m].mean(axis=0)
        # An empty component gets the worst score so it sorts to the bottom
        # deterministically rather than propagating NaN through the argsort.
        scores = np.where(
            np.isnan(means).any(axis=1),
            -np.inf,
            risk_appetite_score(np.nan_to_num(means), self.columns_),
        )
        self.order_ = np.argsort(np.argsort(scores))          # raw -> canonical
        self.inverse_order_ = np.argsort(self.order_)          # canonical -> raw
        self.regime_scores_ = scores[self.inverse_order_]

        canon = self.order_[raw_labels]
        self.regime_shares_ = np.array(
            [float((canon == j).mean()) for j in range(k)]
        )
        self.risk_off_mask_ = self._choose_risk_off_split()

    def _choose_risk_off_split(self) -> np.ndarray:
        """Pick where on the ordered score axis risk-off ends and risk-on begins.

        States are already sorted by risk appetite, so the split is a prefix.
        Among prefixes whose day-share falls inside ``risk_off_share_bounds``
        the one at the widest score gap is taken: that is the most natural
        economic break rather than an arbitrary threshold, and because it
        depends only on the ordered scores and shares it reproduces exactly
        across seeds and vintages.
        """
        k = self.n_regimes
        scores, shares = self.regime_scores_, self.regime_shares_
        lo, hi = self.risk_off_share_bounds
        cum = np.cumsum(shares)

        best_j, best_gap = None, -np.inf
        for j in range(k - 1):                     # risk-off = states 0..j
            share = cum[j]
            if not (lo <= share <= hi):
                continue
            gap = scores[j + 1] - scores[j]
            if gap > best_gap:
                best_j, best_gap = j, gap
        if best_j is None:
            # No admissible gap: fall back to the smallest prefix reaching ``lo``.
            best_j = int(np.argmax(cum >= lo))
        mask = np.zeros(k, dtype=bool)
        mask[: best_j + 1] = True
        return mask

    # -- deterministic initialisation --------------------------------------
    def _score_buckets(self, Xz: np.ndarray) -> np.ndarray:
        """Quantile buckets of the risk-appetite score: a seed-free starting
        partition that already respects the economic ordering."""
        s = risk_appetite_score(Xz, self.columns_)
        qs = np.linspace(0, 1, self.n_regimes + 1)[1:-1]
        return np.digitize(s, np.quantile(s, qs))

    def _fit_hmm_from_init(self, design: np.ndarray, seed_labels: np.ndarray):
        """EM started from the score partition instead of a random draw.

        The Gaussian HMM likelihood on strongly autocorrelated financial data
        is badly multi-modal: random restarts converge to different partitions
        on different data vintages, which is *itself* a source of label
        instability that anchoring cannot repair.  Seeding EM with the
        deterministic score partition removes the lottery, and because the
        starting point is a monotone function of the economic score the basin
        EM lands in is the economically ordered one.
        """
        from hmmlearn.hmm import GaussianHMM

        k, d = self.n_regimes, design.shape[1]
        means = np.vstack([design[seed_labels == j].mean(axis=0) for j in range(k)])
        if self.covariance_type == "full":
            covars = np.stack([
                np.cov(design[seed_labels == j].T) + np.eye(d) * 1e-3
                for j in range(k)
            ])
        else:
            covars = np.vstack([
                design[seed_labels == j].var(axis=0) + 1e-3 for j in range(k)
            ])

        trans = np.ones((k, k)) * 1e-6
        for a, b in zip(seed_labels[:-1], seed_labels[1:]):
            trans[a, b] += 1.0
        trans /= trans.sum(axis=1, keepdims=True)

        m = GaussianHMM(
            n_components=k,
            covariance_type=self.covariance_type,
            n_iter=300,
            tol=1e-4,
            min_covar=1e-3,
            random_state=self.random_state,
            init_params="",          # nothing is drawn at random
            params="stmc",
        )
        m.startprob_ = np.bincount(seed_labels, minlength=k).astype(float) / len(seed_labels)
        m.transmat_ = trans
        m.means_ = means
        m.covars_ = covars
        m.fit(design)
        return m

    def _fit_hmm_random(self, design: np.ndarray):
        from hmmlearn.hmm import GaussianHMM

        best, best_ll = None, -np.inf
        for seed in range(self.n_init):
            m = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=300, tol=1e-4, min_covar=1e-3,
                random_state=self.random_state + seed,
            )
            try:
                m.fit(design)
                ll = m.score(design)
            except Exception:
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        if best is None:
            raise RuntimeError("no HMM fit converged")
        return best

    # -- API ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "RegimeLabeler":
        self.columns_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X.values)
        Xz = self.scaler_.transform(X.values)

        if self.n_pca is not None:
            self.pca_ = PCA(n_components=self.n_pca, random_state=self.random_state).fit(Xz)
            design = self.pca_.transform(Xz)
        else:
            design = Xz

        seed_labels = self._score_buckets(Xz) if self.init == "score" else None

        if self.method == "gmm":
            kw = {}
            if seed_labels is not None:
                kw["means_init"] = np.vstack(
                    [design[seed_labels == j].mean(axis=0) for j in range(self.n_regimes)]
                )
                kw["n_init"] = 1
            else:
                kw["n_init"] = self.n_init
            self.model_ = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                reg_covar=1e-4,
                random_state=self.random_state,
                **kw,
            ).fit(design)
            raw = self.model_.predict(design)

        elif self.method == "hmm":
            if seed_labels is not None:
                self.model_ = self._fit_hmm_from_init(design, seed_labels)
            else:
                self.model_ = self._fit_hmm_random(design)
            raw = self.model_.predict(design)

        elif self.method == "score":
            s = risk_appetite_score(Xz, self.columns_)
            qs = np.linspace(0, 1, self.n_regimes + 1)[1:-1]
            self.score_cuts_ = np.quantile(s, qs)
            raw = np.digitize(s, self.score_cuts_)
            self.model_ = None
        else:
            raise ValueError(f"unknown method {self.method!r}")

        self._canonicalise(Xz, raw)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Hard canonical labels (Viterbi path for the HMM)."""
        if self.method == "score":
            s = risk_appetite_score(self.scaler_.transform(X[self.columns_].values), self.columns_)
            raw = np.digitize(s, self.score_cuts_)
        else:
            raw = self.model_.predict(self._design(X))
        return pd.Series(self.order_[raw], index=X.index, name="regime")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Smoothed posteriors.  Uses the whole of ``X`` -- descriptive only."""
        if self.method == "score":
            hard = self.predict(X).values
            p = np.eye(self.n_regimes)[hard]
        else:
            p = self.model_.predict_proba(self._design(X))[:, self.inverse_order_]
        return pd.DataFrame(p, index=X.index, columns=range(self.n_regimes))

    def filtered_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Forward-only (online) posteriors: P(state_t | observations <= t).

        This is the causal counterpart of ``predict_proba`` and is the only
        state estimate that may be used as a forecasting feature.
        """
        if self.method == "score":
            return self.predict_proba(X)

        design = self._design(X)
        if self.method == "gmm":
            # A mixture has no time dependence, so its posterior is already
            # a function of the current observation only.
            p = self.model_.predict_proba(design)[:, self.inverse_order_]
            return pd.DataFrame(p, index=X.index, columns=range(self.n_regimes))

        log_b = self.model_._compute_log_likelihood(design)
        log_pi = np.log(np.clip(self.model_.startprob_, 1e-300, None))
        log_A = np.log(np.clip(self.model_.transmat_, 1e-300, None))

        n, k = log_b.shape
        alpha = np.empty((n, k))
        a = log_pi + log_b[0]
        alpha[0] = a - _logsumexp(a)
        for t in range(1, n):
            prev = alpha[t - 1]
            a = log_b[t] + _logsumexp_rows(prev[:, None] + log_A)
            alpha[t] = a - _logsumexp(a)
        p = np.exp(alpha)[:, self.inverse_order_]
        return pd.DataFrame(p, index=X.index, columns=range(self.n_regimes))

    # -- derived quantities ------------------------------------------------
    def risk_off_mask(self) -> np.ndarray:
        """Boolean array over canonical labels: True where the state is risk-off."""
        return self.risk_off_mask_

    def binary_labels(self, labels: pd.Series) -> pd.Series:
        mask = self.risk_off_mask()
        return pd.Series(mask[labels.values].astype(int), index=labels.index, name="risk_off")

    def names(self) -> list[str]:
        base = CANONICAL_NAMES.get(self.n_regimes)
        if base is None:
            base = [f"State {i}" for i in range(self.n_regimes)]
        tag = self.risk_off_mask()
        return [f"{nm} ({'off' if t else 'on'})" for nm, t in zip(base, tag)]

    def transition_matrix(self, labels: pd.Series, horizon: int = 1) -> pd.DataFrame:
        return empirical_transition_matrix(labels, horizon, self.n_regimes)


def _logsumexp(a: np.ndarray) -> float:
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def _logsumexp_rows(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=0)
    return m + np.log(np.exp(a - m).sum(axis=0))


def empirical_transition_matrix(
    labels: pd.Series, horizon: int = 1, n_regimes: int | None = None
) -> pd.DataFrame:
    """Row-normalised P(state_{t+h} | state_t) with Laplace smoothing."""
    v = np.asarray(labels)
    k = n_regimes or int(v.max() + 1)
    counts = np.ones((k, k)) * 1e-9
    for a, b in zip(v[:-horizon], v[horizon:]):
        counts[a, b] += 1
    return pd.DataFrame(counts / counts.sum(axis=1, keepdims=True),
                        index=range(k), columns=range(k))


def regime_characteristics(
    panel: pd.DataFrame, labels: pd.Series, n_regimes: int
) -> pd.DataFrame:
    """Economic description of each canonical regime, in native units."""
    idx = labels.index
    out = {}
    px = panel.loc[idx]

    fwd_1d = np.log(panel["SPX"]).diff().shift(-1).loc[idx]

    for j in range(n_regimes):
        m = labels.values == j
        if not m.any():
            out[j] = {}
            continue
        row = {
            "share_of_days": float(m.mean()),
            "SPX_ann_return": float(fwd_1d[m].mean() * 252),
            "SPX_ann_vol": float(fwd_1d[m].std() * np.sqrt(252)),
            "SPX_hit_rate": float((fwd_1d[m] > 0).mean()),
        }
        row["SPX_sharpe"] = row["SPX_ann_return"] / row["SPX_ann_vol"] if row["SPX_ann_vol"] else np.nan
        for col, label in [
            ("VIX", "VIX_level"),
            ("BaaSpread", "Baa_spread"),
            ("Slope_10Y2Y", "Curve_10y2y"),
            ("DGS10", "US10y"),
        ]:
            if col in px.columns:
                row[label] = float(px[col][m].mean())
        out[j] = row
    df = pd.DataFrame(out).T
    df.index.name = "regime"
    return df


def regime_asset_returns(
    panel: pd.DataFrame, labels: pd.Series, n_regimes: int
) -> pd.DataFrame:
    """Annualised next-day returns per asset, conditional on today's regime.

    This is the check that the risk-on/risk-off reading means what it says:
    defensive assets should improve, and cyclical ones deteriorate, as the
    canonical index falls.  Rate series are reported as changes in basis
    points rather than returns.
    """
    idx = labels.index
    price_cols = [c for c in ["SPX", "NDX", "RUT", "Nikkei", "Gold", "Oil", "DXY", "USDJPY"]
                  if c in panel.columns]
    rate_cols = [c for c in ["DGS10", "DGS2", "BaaSpread", "Slope_10Y2Y"]
                 if c in panel.columns]

    fwd = {c: np.log(panel[c]).diff().shift(-1).reindex(idx) for c in price_cols}
    fwd.update({c: panel[c].diff().shift(-1).reindex(idx) for c in rate_cols})

    rows = {}
    for j in range(n_regimes):
        m = labels.values == j
        if not m.any():
            continue
        row = {}
        for c in price_cols:
            row[f"{c} (ann. %)"] = float(fwd[c][m].mean() * 252 * 100)
        for c in rate_cols:
            row[f"{c} (bp/yr)"] = float(fwd[c][m].mean() * 252 * 100)
        rows[j] = row
    out = pd.DataFrame(rows).T
    out.index.name = "regime"
    return out


def episode_table(labels: pd.Series, min_days: int = 5) -> pd.DataFrame:
    """Contiguous runs of a single regime, for inspecting history."""
    v = labels.values
    starts = [0] + list(np.where(v[1:] != v[:-1])[0] + 1)
    ends = starts[1:] + [len(v)]
    rows = []
    for s, e in zip(starts, ends):
        if e - s < min_days:
            continue
        rows.append(
            {
                "regime": int(v[s]),
                "start": labels.index[s],
                "end": labels.index[e - 1],
                "days": e - s,
            }
        )
    return pd.DataFrame(rows)
