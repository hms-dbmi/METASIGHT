"""The SPEC interpreter of the two-stage package, shared verbatim by Stage 2 and every program it emits.

The node {"t": "stage1", "idx": i} is the output of Stage 1 candidate i, so a Stage 2 program can refine a
Stage 1 program (route it to some cancers, mix it with other bases, calibrate it). `run_blend` takes the Stage 1
candidates in `stage1`; routed cells index the full-input output by row.

SPEC = {"model","fm_order","base",<"calib">,<"renorm">}; node types:
    sim, logitmean, wmean, rankmean, bestfm, blend, lr, poly, rff, routed, mix, protect   (Model_1)
    sim, rankmean3, wmean3, bestfm, blend, lr3, poly3, rff3, routed, mix, protect          (Model_2)
    stage1                                                                                  (both)
calib: Model_1 [t,b] or {"by_cancer":{c:[t,b]},"default":[t,b]}; Model_2 the same per class. renorm (Model_2).
"""
INTERP_SRC = r'''
import numpy as np
from itertools import combinations_with_replacement
from typing import Dict, List
_EPS = 1e-6
def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))
def _idxmap(spec_order, fm_names):
    # position of each spec FM within the runtime fm_names (identity if same order)
    pos = {n: i for i, n in enumerate(list(fm_names))}
    return [pos.get(n, i) for i, n in enumerate(list(spec_order))]

def _sim_m1(P):
    with np.errstate(invalid="ignore"):
        return np.clip(np.nanmean(P, axis=0), 0.0, 1.0)
def _logitmean_m1(P):
    ok = np.isfinite(P); lg = np.where(ok, _logit(P), np.nan)
    with np.errstate(invalid="ignore"):
        return np.clip(_sig(np.nanmean(lg, axis=0)), 0.0, 1.0)
def _wmean_m1(P, w):
    w = np.asarray(w, float)[:, None]; ok = np.isfinite(P)
    num = np.nansum(np.where(ok, P, 0.0) * np.where(ok, w, 0.0), axis=0)
    den = np.nansum(np.where(ok, w, 0.0), axis=0)
    return np.clip(np.where(den > 0, num / den, np.nan), 0.0, 1.0)
def _bestfm_m1(P, i, fb):
    p = P[i]
    return np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), fb)
def _lr_m1(P, coef, b):
    K, N = P.shape
    X = _logit(np.where(np.isfinite(P), np.clip(P, _EPS, 1 - _EPS), 0.5))  # (K,N), NaN->logit .5
    z = np.asarray(coef, float) @ X + float(b)
    return np.clip(_sig(z), 0.0, 1.0)

def _logits_NK(P):                                  # (K,N) probs -> (N,K) logits, NaN->logit .5
    return _logit(np.where(np.isfinite(P), np.clip(P, _EPS, 1 - _EPS), 0.5)).T
def _poly_feats(L, degree):                          # (N,K) -> (N,F) no bias (canonical order)
    N, K = L.shape; cols = []
    for d in range(1, int(degree) + 1):
        for combo in combinations_with_replacement(range(K), d):
            v = np.ones(N)
            for k in combo:
                v = v * L[:, k]
            cols.append(v)
    return np.column_stack(cols)
def _poly_m1(P, node):
    X = _poly_feats(_logits_NK(P), node["degree"])
    return np.clip(_sig(X @ np.asarray(node["coef"], float) + float(node["b"])), 0.0, 1.0)
def _rff_m1(P, node):
    L = _logits_NK(P); Ls = (L - np.asarray(node["mean"], float)) / np.asarray(node["std"], float)
    X = np.cos(Ls @ np.asarray(node["W"], float).T + np.asarray(node["b"], float))
    return np.clip(_sig(X @ np.asarray(node["coef"], float) + float(node["int"])), 0.0, 1.0)

def _rankmean_m1(P):                                 # mean of per-FM ranks (NaN-aware) -> [0,1]
    K, N = P.shape; R = np.full((K, N), np.nan)
    for i in range(K):
        v = P[i]; ok = np.isfinite(v); m = int(ok.sum())
        if m == 0:
            continue
        ranks = np.argsort(np.argsort(v[ok]))
        R[i, ok] = ranks / (m - 1) if m > 1 else 0.5
    with np.errstate(invalid="ignore"):
        return np.clip(np.nanmean(R, axis=0), 0.0, 1.0)


def _base_m1(node, P, ct, rows=None):
    t = node["t"]
    if t == "stage1":                               # a Stage 1 candidate, evaluated once on the full input
        return np.clip(np.asarray(_S1[int(node["idx"])], float)[rows], 0.0, 1.0)
    if t == "sim": return _sim_m1(P)
    if t == "rankmean": return _rankmean_m1(P)
    if t == "poly": return _poly_m1(P, node)
    if t == "rff": return _rff_m1(P, node)
    if t == "protect":                               # listed cancers use their best single FM
        out = np.asarray(_base_m1(node["base"], P, ct, rows), float).copy(); sim = _sim_m1(P)
        for cancer, idx in node["overrides"].items():
            m = (ct == cancer)
            if m.any():
                out[m] = _bestfm_m1(P[:, m], int(idx), sim[m])
        return out
    if t == "logitmean": return _logitmean_m1(P)
    if t == "wmean": return _wmean_m1(P, node["w"])
    if t == "bestfm": return _bestfm_m1(P, int(node["idx"]), _sim_m1(P))
    if t == "blend":
        sim = _sim_m1(P); bf = _bestfm_m1(P, int(node["idx"]), sim)
        l = float(node["lam"]); return l * bf + (1 - l) * sim
    if t == "lr": return _lr_m1(P, node["coef"], node["b"])
    if t == "mix":
        g = float(node["g"]); return g * _base_m1(node["a"], P, ct, rows) + (1 - g) * _base_m1(node["b"], P, ct, rows)
    if t == "routed":
        out = _base_m1(node.get("default", {"t": "sim"}), P, ct, rows)
        ctv = np.asarray(ct, dtype=object)
        for cancer, sub in node["cells"].items():
            m = (ctv == cancer)
            if m.any():
                out = out.copy(); out[m] = _base_m1(sub, P[:, m], ctv[m], None if rows is None else rows[m])
        return out
    raise ValueError("bad node " + str(t))

def _renorm3(M):
    s = np.nansum(M, axis=1, keepdims=True)
    return np.where(s > 0, M / s, np.nan)
def _sim_m2(P):
    with np.errstate(invalid="ignore"):
        m = np.nanmean(P, axis=0)
    return _renorm3(m)
def _bestfm_m2(P, i):
    p = P[i].copy(); sim = _sim_m2(P)
    bad = ~np.isfinite(p).all(axis=1)
    p[bad] = sim[bad]; return _renorm3(p)
def _wmean3(P, w):
    w = np.asarray(w, float)[:, None, None]; ok = np.isfinite(P)
    num = np.nansum(np.where(ok, P, 0.0) * np.where(ok, w, 0.0), axis=0)
    den = np.nansum(np.where(ok, w, 0.0), axis=0)
    return _renorm3(np.where(den > 0, num / den, np.nan))
def _lr3_m2(P, coef, b):
    K, N, _ = P.shape; out = np.zeros((N, 3))
    for c in range(3):
        Xc = _logit(np.where(np.isfinite(P[:, :, c]), np.clip(P[:, :, c], _EPS, 1 - _EPS), 0.5))  # (K,N)
        out[:, c] = _sig(np.asarray(coef[c], float) @ Xc + float(b[c]))
    return _renorm3(out)

def _poly3_m2(P, node):
    K, N, _ = P.shape; out = np.zeros((N, 3)); deg = node["degree"]
    for c in range(3):
        L = _logit(np.where(np.isfinite(P[:, :, c]), np.clip(P[:, :, c], _EPS, 1 - _EPS), 0.5)).T
        X = _poly_feats(L, deg)
        out[:, c] = _sig(X @ np.asarray(node["coef"][c], float) + float(node["b"][c]))
    return _renorm3(out)
def _rff3_m2(P, node):
    K, N, _ = P.shape; out = np.zeros((N, 3))
    for c in range(3):
        L = _logit(np.where(np.isfinite(P[:, :, c]), np.clip(P[:, :, c], _EPS, 1 - _EPS), 0.5)).T
        Ls = (L - np.asarray(node["mean"][c], float)) / np.asarray(node["std"][c], float)
        X = np.cos(Ls @ np.asarray(node["W"][c], float).T + np.asarray(node["b"][c], float))
        out[:, c] = _sig(X @ np.asarray(node["coef"][c], float) + float(node["int"][c]))
    return _renorm3(out)

def _rankmean3_m2(P):
    K, N, _ = P.shape; out = np.zeros((N, 3))
    for c in range(3):
        R = np.full((K, N), np.nan)
        for i in range(K):
            v = P[i, :, c]; ok = np.isfinite(v); m = int(ok.sum())
            if m == 0:
                continue
            R[i, ok] = np.argsort(np.argsort(v[ok])) / (m - 1) if m > 1 else 0.5
        with np.errstate(invalid="ignore"):
            out[:, c] = np.nanmean(R, axis=0)
    return _renorm3(out)


def _base_m2(node, P, ct, rows=None):
    t = node["t"]
    if t == "stage1":
        return _renorm3(np.clip(np.asarray(_S1[int(node["idx"])], float)[rows], 0.0, 1.0))
    if t == "sim": return _sim_m2(P)
    if t == "rankmean3": return _rankmean3_m2(P)
    if t == "wmean3": return _wmean3(P, node["w"])
    if t == "poly3": return _poly3_m2(P, node)
    if t == "rff3": return _rff3_m2(P, node)
    if t == "protect":
        out = np.asarray(_base_m2(node["base"], P, ct, rows), float).copy()
        for cancer, idx in node["overrides"].items():
            m = (ct == cancer)
            if m.any():
                out[m] = _bestfm_m2(P[:, m], int(idx))
        return _renorm3(out)
    if t == "bestfm": return _bestfm_m2(P, int(node["idx"]))
    if t == "blend":
        sim = _sim_m2(P); bf = _bestfm_m2(P, int(node["idx"])); l = float(node["lam"])
        return _renorm3(l * bf + (1 - l) * sim)
    if t == "lr3": return _lr3_m2(P, node["coef"], node["b"])
    if t == "mix":
        g = float(node["g"]); return _renorm3(g * _base_m2(node["a"], P, ct, rows) + (1 - g) * _base_m2(node["b"], P, ct, rows))
    if t == "routed":
        out = _base_m2(node.get("default", {"t": "sim"}), P, ct, rows)
        ctv = np.asarray(ct, dtype=object)
        for cancer, sub in node["cells"].items():
            m = (ctv == cancer)
            if m.any():
                out = out.copy(); out[m] = _base_m2(sub, P[:, m], ctv[m], None if rows is None else rows[m])
        return out
    raise ValueError("bad node " + str(t))

def _cal1(p, tb):
    return _sig(float(tb[0]) * _logit(np.clip(p, _EPS, 1 - _EPS)) + float(tb[1]))
def _apply_cal_m1(p, cal, ct):
    if cal is None:
        return p
    if isinstance(cal, dict):                      # per-cancer calibration
        out = np.asarray(p, float).copy(); seen = np.zeros(len(p), bool)
        for c, tb in cal.get("by_cancer", {}).items():
            m = (ct == c); out[m] = _cal1(p[m], tb); seen |= m
        dft = cal.get("default")
        if dft is not None and (~seen).any():
            out[~seen] = _cal1(p[~seen], dft)
        return np.clip(out, 0.0, 1.0)
    return np.clip(_cal1(p, cal), 0.0, 1.0)        # global [t,b]
def _apply_cal_m2(M, cal, ct):
    if cal is None:
        return M
    out = np.asarray(M, float).copy()
    if isinstance(cal, dict):
        seen = np.zeros(len(M), bool)
        for c, tbs in cal.get("by_cancer", {}).items():
            m = (ct == c)
            for k in range(3):
                out[m, k] = _cal1(M[m, k], tbs[k])
            seen |= m
        dft = cal.get("default")
        if dft is not None and (~seen).any():
            for k in range(3):
                out[~seen, k] = _cal1(M[~seen, k], dft[k])
    else:
        for k in range(3):
            out[:, k] = _cal1(M[:, k], cal[k])
    return out

_S1 = []
def run_blend(SPEC, fm_probs, fm_names, cancer_types, context=None, stage1=()):
    # stage1: the Stage 1 candidates, each a precomputed output array or a callable with the
    # blend contract; a callable is evaluated once, on the full input in the runtime FM order.
    global _S1
    _S1 = [np.asarray(s(fm_probs, list(fm_names), cancer_types, context or {}), float) if callable(s)
           else np.asarray(s, float) for s in stage1]
    P = np.asarray(fm_probs, float)
    order = _idxmap(SPEC.get("fm_order", fm_names), fm_names)
    P = P[order]                                   # reorder rows to SPEC's fm_order
    ct = np.asarray(cancer_types, dtype=object)
    if SPEC["model"] == "Model_1":
        p = _base_m1(SPEC["base"], P, ct, np.arange(P.shape[1]))
        return _apply_cal_m1(p, SPEC.get("calib"), ct)
    else:
        M = _base_m2(SPEC["base"], P, ct, np.arange(P.shape[1]))
        M = _apply_cal_m2(M, SPEC.get("calib"), ct)
        if SPEC.get("renorm", True):
            M = _renorm3(M)
        return M
'''

_ns: dict = {}
exec(INTERP_SRC, _ns)
run_blend = _ns["run_blend"]
