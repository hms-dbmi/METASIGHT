"""Fitting helpers that turn a program's fitted nodes into numbers: the learner fits and the calibration fits.

Every fit uses the rows it is given (the search's TCGA training cohort, or one cancer type of it) with fixed seeds.
The numbers are stored in the program, which is then applied unchanged to data the search never saw.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from metasightcmp.search import interp

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

_I = interp._ns
EPS = 1e-6
CS = (0.3, 1.0, 3.0)
RFF_CFG = ((128, 0.5), (128, 1.0), (256, 1.0))
TEMP_GRID = np.round(np.concatenate([np.linspace(0.4, 1.0, 5), np.linspace(1.15, 3.0, 12)]), 3)


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def pc_auroc_m1(y, p):
    m = np.isfinite(p)
    if m.sum() < 5 or len(set(y[m].astype(int).tolist())) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y[m].astype(int), p[m]))
    except Exception:
        return np.nan


def pc_auroc_m2(y, M):
    m = np.all(np.isfinite(M), axis=1)
    if m.sum() < 5:
        return np.nan
    aucs = []
    for c in range(3):
        yb = (y[m] == c).astype(int)
        if len(set(yb.tolist())) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yb, M[m, c]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else np.nan


# ----------------------------------------------------------------------------- learners, Model_1
def fit_lr_m1(P, y, Cs=CS):
    X = _I["_logit"](np.where(np.isfinite(P), np.clip(P, EPS, 1 - EPS), 0.5)).T  # (N,K)
    yy = y.astype(int)
    if len(set(yy.tolist())) < 2:
        return None
    best = None
    for C in Cs:
        try:
            lr = LogisticRegression(C=C, max_iter=700).fit(X, yy)
        except Exception:
            continue
        node = {"t": "lr", "coef": lr.coef_.ravel().tolist(), "b": float(lr.intercept_[0])}
        a = pc_auroc_m1(y, _I["_lr_m1"](P, node["coef"], node["b"]))
        if best is None or (a is not None and not np.isnan(a) and a > best[1]):
            best = (node, a if a is not None else -1)
    return best[0] if best else None


def fit_poly_m1(P, y, degree=2, Cs=(0.3, 1.0)):
    X = _I["_poly_feats"](_I["_logits_NK"](P), degree)
    yy = y.astype(int)
    if len(set(yy.tolist())) < 2:
        return None
    best = None
    for C in Cs:
        try:
            lr = LogisticRegression(C=C, max_iter=800).fit(X, yy)
        except Exception:
            continue
        node = {"t": "poly", "degree": degree, "coef": lr.coef_.ravel().tolist(), "b": float(lr.intercept_[0])}
        a = pc_auroc_m1(y, _I["_poly_m1"](P, node))
        if best is None or (not np.isnan(a) and a > best[1]):
            best = (node, a if not np.isnan(a) else -1)
    return best[0] if best else None


def fit_rff_m1(P, y, cfgs=RFF_CFG, seed=0, C=1.0):
    L = _I["_logits_NK"](P); mean = L.mean(0); std = L.std(0) + 1e-9; Ls = (L - mean) / std
    yy = y.astype(int)
    if len(set(yy.tolist())) < 2:
        return None
    best = None
    for dim, gamma in cfgs:
        rng = np.random.default_rng(seed)
        W = rng.normal(0, np.sqrt(2 * gamma), (dim, L.shape[1])); bb = rng.uniform(0, 2 * np.pi, dim)
        X = np.cos(Ls @ W.T + bb)
        try:
            lr = LogisticRegression(C=C, max_iter=800).fit(X, yy)
        except Exception:
            continue
        node = {"t": "rff", "mean": mean.tolist(), "std": std.tolist(), "W": W.tolist(),
                "b": bb.tolist(), "coef": lr.coef_.ravel().tolist(), "int": float(lr.intercept_[0])}
        a = pc_auroc_m1(y, _I["_rff_m1"](P, node))
        if best is None or (not np.isnan(a) and a > best[1]):
            best = (node, a if not np.isnan(a) else -1)
    return best[0] if best else None


# ----------------------------------------------------------------------------- learners, Model_2
def fit_lr3_m2(P, y, C=1.0):
    yy = y.astype(int)
    coef, b = [], []
    for c in range(3):
        Xc = _I["_logit"](np.where(np.isfinite(P[:, :, c]), np.clip(P[:, :, c], EPS, 1 - EPS), 0.5)).T
        yb = (yy == c).astype(int)
        if len(set(yb.tolist())) < 2:
            coef.append([0.0] * P.shape[0]); b.append(0.0); continue
        try:
            lr = LogisticRegression(C=C, max_iter=700).fit(Xc, yb)
            coef.append(lr.coef_.ravel().tolist()); b.append(float(lr.intercept_[0]))
        except Exception:
            coef.append([0.0] * P.shape[0]); b.append(0.0)
    return {"t": "lr3", "coef": coef, "b": b}


def fit_poly3_m2(P, y, degree=2, C=1.0):
    coef, b = [], []
    for c in range(3):
        L = _I["_logit"](np.where(np.isfinite(P[:, :, c]), np.clip(P[:, :, c], EPS, 1 - EPS), 0.5)).T
        X = _I["_poly_feats"](L, degree); yb = (y.astype(int) == c).astype(int)
        if len(set(yb.tolist())) < 2:
            coef.append([0.0] * X.shape[1]); b.append(0.0); continue
        try:
            lr = LogisticRegression(C=C, max_iter=800).fit(X, yb)
            coef.append(lr.coef_.ravel().tolist()); b.append(float(lr.intercept_[0]))
        except Exception:
            coef.append([0.0] * X.shape[1]); b.append(0.0)
    return {"t": "poly3", "degree": degree, "coef": coef, "b": b}


def fit_rff3_m2(P, y, dim=128, gamma=1.0, seed=0, C=1.0):
    means, stds, Ws, bs, coef, inter = [], [], [], [], [], []
    for c in range(3):
        L = _I["_logit"](np.where(np.isfinite(P[:, :, c]), np.clip(P[:, :, c], EPS, 1 - EPS), 0.5)).T
        mean = L.mean(0); std = L.std(0) + 1e-9; Ls = (L - mean) / std
        rng = np.random.default_rng(seed)
        W = rng.normal(0, np.sqrt(2 * gamma), (dim, L.shape[1])); bb = rng.uniform(0, 2 * np.pi, dim)
        X = np.cos(Ls @ W.T + bb); yb = (y.astype(int) == c).astype(int)
        means.append(mean.tolist()); stds.append(std.tolist()); Ws.append(W.tolist()); bs.append(bb.tolist())
        if len(set(yb.tolist())) < 2:
            coef.append([0.0] * dim); inter.append(0.0); continue
        try:
            lr = LogisticRegression(C=C, max_iter=800).fit(X, yb)
            coef.append(lr.coef_.ravel().tolist()); inter.append(float(lr.intercept_[0]))
        except Exception:
            coef.append([0.0] * dim); inter.append(0.0)
    return {"t": "rff3", "mean": means, "std": stds, "W": Ws, "b": bs, "coef": coef, "int": inter}


# ----------------------------------------------------------------------------- calibration
def fit_calibration(p, y):
    """(t, b) for p -> sigmoid(t*logit(p)+b) minimizing the Brier score on the training rows, over a grid."""
    m = np.isfinite(p)
    lg = _logit(p[m]); yy = y[m]
    best = (1.0, 0.0, 1e9)
    b_grid = np.linspace(-1.5, 1.5, 16)
    for t in TEMP_GRID:
        z = t * lg
        for b in b_grid:
            q = _sig(z + b)
            br = float(np.mean((q - yy) ** 2))
            if br < best[2]:
                best = (float(t), float(b), br)
    return best[0], best[1]


def calib_global(blend, y, model):
    """Global monotone calibration of a base output. Returns (calib, renorm)."""
    if model == "Model_1":
        t, b = fit_calibration(np.asarray(blend, float).reshape(-1), y.astype(float))
        return [float(t), float(b)], None
    cal = []
    for c in range(3):
        t, b = fit_calibration(blend[:, c], (y == c).astype(float))
        cal.append([float(t), float(b)])
    return cal, True


def calib_per_cancer(base, y, ct, model):
    """Per-cancer calibration with a global default for small cancers."""
    ctv = np.asarray(ct, dtype=object)
    if model == "Model_1":
        gt, gb = fit_calibration(base.reshape(-1), y.astype(float)); by = {}
        for c in sorted(set(ctv.tolist())):
            m = ctv == c
            if m.sum() >= 20 and len(set(y[m].astype(int).tolist())) > 1:
                t, b = fit_calibration(base.reshape(-1)[m], y[m].astype(float)); by[c] = [float(t), float(b)]
        return {"by_cancer": by, "default": [float(gt), float(gb)]}, None
    gd = [list(map(float, fit_calibration(base[:, k], (y == k).astype(float)))) for k in range(3)]
    by = {}
    for c in sorted(set(ctv.tolist())):
        m = ctv == c
        if m.sum() >= 20:
            tbs = []
            for k in range(3):
                yb = (y[m] == k).astype(float)
                tbs.append(list(map(float, fit_calibration(base[m, k], yb))) if len(set(yb.tolist())) > 1 else gd[k])
            by[c] = tbs
    return {"by_cancer": by, "default": gd}, True
