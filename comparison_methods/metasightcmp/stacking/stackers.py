"""The six supervised stacking methods.

Every method has the same interface:

    m = make(name, seed, **configuration)
    m.fit(X, y, n_classes)
    P = m.predict_proba(X)          # (N, n_classes), rows sum to 1

Two families:
  feature stackers  ridge, elasticnet, logistic, gbm: a classifier on a design matrix built from the foundation-model
                    (FM) probabilities (stacking/fit.py builds it).
  weight stackers   nnls, bma: a convex combination of the FMs' predictive distributions; `fit` and `predict_proba`
                    take the (K, N, C) FM stack.

GRIDS holds each method's hyperparameter grid. stacking/fit.py draws a fixed number of configurations from it and
selects among them by inner cross-validation on the training data.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

# Some BLAS builds raise RuntimeWarnings (divide by zero, overflow, invalid value) inside scikit-learn although every
# value is finite. They are silenced unless METASIGHTCMP_SHOW_WARNINGS=1. A prediction that is not finite is an error
# (stacking/fit.py checks every output).
if os.environ.get("METASIGHTCMP_SHOW_WARNINGS", "") != "1":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(all="ignore")
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
DEFAULT_BUDGET = 300              # configurations per method and training set


def _norm(P: np.ndarray) -> np.ndarray:
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    return P / P.sum(axis=1, keepdims=True)


def _align_columns(P: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """scikit-learn drops a class that the training data lack; put it back with a probability near 0."""
    out = np.full((P.shape[0], n_classes), EPS, dtype=float)
    for j, c in enumerate(classes):
        out[:, int(c)] = P[:, j]
    return _norm(out)


# --------------------------------------------------------------------------- feature stackers
class _SkStacker:
    """A scikit-learn classifier. `needs_imputation` says whether the design matrix must have its missing entries
    filled (histogram gradient boosting takes NaN as it is)."""

    needs_imputation = True

    def __init__(self, est, seed: int):
        self.est, self.seed, self.n_classes_ = est, seed, None
        self.degenerate_ = None

    def fit(self, X, y, n_classes):
        self.n_classes_ = n_classes
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:          # training data with one class
            self.degenerate_ = int(np.unique(y)[0])
            return self
        self.est.fit(np.asarray(X, dtype=float), y)
        return self

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        if self.degenerate_ is not None:
            out = np.full((n, self.n_classes_), EPS)
            out[:, self.degenerate_] = 1.0
            return _norm(out)
        P = self.est.predict_proba(np.asarray(X, dtype=float))
        return _align_columns(P, self.est.classes_, self.n_classes_)


def _ridge(seed, C=1.0):
    return _SkStacker(LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                         max_iter=5000, random_state=seed), seed)


def _elasticnet(seed, C=1.0, l1_ratio=0.5):
    return _SkStacker(LogisticRegression(penalty="elasticnet", C=C, l1_ratio=l1_ratio,
                                         solver="saga", max_iter=8000, random_state=seed), seed)


def _logistic(seed, C=1e6):
    # Unpenalized logistic stacking; the very large C keeps lbfgs stable.
    return _SkStacker(LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                         max_iter=8000, random_state=seed), seed)


def _gbm(seed, max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, l2=0.0):
    # Early stopping on an internal split of the training data, so max_iter is an upper bound.
    m = _SkStacker(HistGradientBoostingClassifier(
        max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2,
        early_stopping=True, n_iter_no_change=10, validation_fraction=0.15,
        random_state=seed), seed)
    m.needs_imputation = False
    return m


# --------------------------------------------------------------------------- weight stackers
class _WeightStacker:
    """A convex combination of the FMs' predictive distributions.

    fit(stack, y, n_classes) with stack (K, N, C). NaN means that FM k does not cover unit i; the weights are
    renormalized per unit over the FMs that cover it."""

    needs_imputation = False
    kind = "weight"

    def __init__(self, seed):
        self.seed, self.w_, self.n_classes_ = seed, None, None

    @staticmethod
    def _stack_to_probs(stack, n_classes):
        S = np.asarray(stack, dtype=float)
        if S.shape[-1] == 1:                                   # Model_1 -> (K, N, 2)
            S = np.stack([1.0 - S[:, :, 0], S[:, :, 0]], axis=-1)
        return np.clip(S, EPS, 1.0)

    def _blend(self, S, w):
        cov = ~np.isnan(S[:, :, 0])                            # (K, N)
        W = np.where(cov, w[:, None], 0.0)
        den = W.sum(axis=0, keepdims=True)
        W = np.divide(W, den, out=np.zeros_like(W), where=den > 0)
        P = np.nansum(np.nan_to_num(S, nan=0.0) * W[:, :, None], axis=0)
        bad = P.sum(axis=1) <= 0
        if bad.any():
            P[bad] = 1.0 / P.shape[1]
        return _norm(P)

    def predict_proba(self, stack):
        S = self._stack_to_probs(stack, self.n_classes_)
        return self._blend(S, self.w_)


class NNLS(_WeightStacker):
    """Nonnegative weights on the simplex that minimize the log loss on the training data."""

    def fit(self, stack, y, n_classes):
        self.n_classes_ = n_classes
        S = self._stack_to_probs(stack, n_classes)
        y = np.asarray(y).astype(int)
        K = S.shape[0]

        def nll(w):
            P = self._blend(S, np.abs(w))
            return -np.mean(np.log(P[np.arange(len(y)), y]))

        w0 = np.full(K, 1.0 / K)
        res = minimize(nll, w0, method="SLSQP",
                       bounds=[(0.0, 1.0)] * K,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                       options={"maxiter": 300, "ftol": 1e-9})
        w = np.abs(res.x) if res.x is not None else w0
        self.w_ = w / w.sum() if w.sum() > 0 else w0
        return self


class PseudoBMA(_WeightStacker):
    """Pseudo Bayesian model averaging (Yao et al. 2018): w_k proportional to exp(tau * elpd_k), where elpd_k is
    FM k's summed log predictive density on the training data. tau = 1 is the standard form."""

    def __init__(self, seed, tau=1.0):
        super().__init__(seed)
        self.tau = tau

    def fit(self, stack, y, n_classes):
        self.n_classes_ = n_classes
        S = self._stack_to_probs(stack, n_classes)
        y = np.asarray(y).astype(int)
        idx = np.arange(len(y))
        elpd = np.array([np.nansum(np.log(S[k][idx, y])) for k in range(S.shape[0])])
        elpd = np.where(np.isfinite(elpd), elpd, -np.inf)
        z = self.tau * (elpd - np.max(elpd))
        w = np.exp(np.clip(z, -700, 0))
        self.w_ = w / w.sum() if w.sum() > 0 else np.full(S.shape[0], 1.0 / S.shape[0])
        return self


# --------------------------------------------------------------------------- grids and factory
GRIDS = {
    "ridge":      {"C": list(np.logspace(-4, 4, 81))},
    "elasticnet": {"C": list(np.logspace(-4, 4, 33)),
                   "l1_ratio": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]},
    "logistic":   {"C": [1e6]},                       # one configuration
    "gbm":        {"max_depth": [2, 3, 4, 6, None],
                   "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                   "max_iter": [100, 200, 400],
                   "min_samples_leaf": [5, 10, 20, 50],
                   "l2_regularization": [0.0, 0.1, 1.0]},
    "nnls":       {},                                  # one configuration
    "bma":        {"tau": [1.0]},                      # one configuration
}

WEIGHT_FAMILIES = {"nnls", "bma"}
ALL = ["ridge", "elasticnet", "logistic", "gbm", "nnls", "bma"]
NAMES = {"ridge": "Ridge", "elasticnet": "Elastic net", "logistic": "Logistic", "gbm": "GBM",
         "nnls": "NNLS (simplex)", "bma": "Pseudo-BMA"}


def make(name: str, seed: int, **params):
    if name == "ridge":
        return _ridge(seed, **params)
    if name == "elasticnet":
        return _elasticnet(seed, **params)
    if name == "logistic":
        return _logistic(seed, **params)
    if name == "gbm":
        return _gbm(seed, **{("l2" if k == "l2_regularization" else k): v
                             for k, v in params.items()})
    if name == "nnls":
        return NNLS(seed)
    if name == "bma":
        return PseudoBMA(seed, **params)
    raise ValueError(f"unknown stacking method: {name}")
