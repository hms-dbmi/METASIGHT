"""The program space of the two searches, and the operators that draw, change and combine programs.

A candidate is a genome: {"base": node tree, "calib": "none" | "global" | "pc"}. Building it (`Space.build_spec`)
fits its learner nodes and its calibration on the search's training cohort and gives a SPEC that
search/interp.py computes. The node types are those of the interpreter:

    sim, logitmean, rankmean / rankmean3, wmean / wmean3, bestfm, blend    combinations without fitted numbers
    car                                  wmean with the Caruana weights of the training cohort
    fit: lr, poly, rff / lr3, poly3, rff3    learners fitted on the rows where the node sits
    routed                               one node per cancer type
    mix                                  weighted mean of two nodes
    protect                              listed cancer types use their best single FM

A learner is offered only where its rows allow it: logistic regression with both classes and at least 12 rows
(Model_2: 18), the polynomial and random-feature learners with at least 40 rows, the random-feature learners and the
Model_2 polynomial only over the whole cohort.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from metasightcmp.data import cohorts as C
from metasightcmp.search import fitters as F
from metasightcmp.search import interp
from metasightcmp.search import scoring

GRID_LAM = [round(float(x), 2) for x in np.linspace(0.0, 1.0, 11)]
GRID_G = [round(float(x), 2) for x in np.arange(0.10, 0.9001, 0.05)]
RFF_CFGS = [(128, 0.5), (128, 1.0), (256, 1.0)]


class Space:
    """What a candidate needs: the training cohort, its FM stack, which nodes are allowed where, and a cache of the
    fitted nodes (a fit depends only on the cohort, the rows, the learner and its configuration)."""

    def __init__(self, co):
        self.co = co
        self.model = co.model
        self.m1 = co.model == "Model_1"
        self.P = C.stack(co)
        self.K = self.P.shape[0]
        self.fo = list(co.fm_order)
        self.y = np.asarray(co.y).astype(float)
        self.ct = np.asarray(co.cancer_types, dtype=object)
        self.cancers = sorted(set(self.ct.tolist()))
        self.multi = len(self.cancers) >= 2
        self.mask = {c: self.ct == c for c in self.cancers}
        self.n = {c: int(self.mask[c].sum()) for c in self.cancers}
        self.ncls = {c: len(set(self.y[self.mask[c]].astype(int).tolist())) for c in self.cancers}
        self.n_all = len(self.y)
        self.ncls_all = len(set(self.y.astype(int).tolist()))
        fm_auroc = {k: (v.get("auroc_macro") or 0.5) for k, v in co.fm_metrics.items()}
        self.w_auc = [round(float(fm_auroc.get(f) or 0.5), 6) for f in self.fo]
        self.car = None
        if getattr(co, "tcga", None) is not None:
            wmap = {f: 0.0 for f in self.fo}
            for f, wv in zip(list(co.tcga.fm_order), np.asarray(co.tcga.weights, float)):
                if f in wmap:
                    wmap[f] = float(wv)
            wc = [wmap[f] for f in self.fo]
            if sum(wc) > 0:
                self.car = wc
        floors = dict(co.per_cancer_floor); best_fm = dict(co.per_cancer_best_fm)
        self.protect = {c: self.fo.index(best_fm[c]) for c in self.cancers
                        if floors.get(c) is not None and floors[c] >= 0.999 and self.n[c] >= 30
                        and best_fm.get(c) in self.fo}
        self.cache = {}

    # rows in scope (None: the whole cohort; otherwise one cancer type)
    def _rows(self, scope):
        return slice(None) if scope is None else self.mask[scope]

    def n_rows(self, scope):
        return self.n_all if scope is None else self.n[scope]

    def n_classes(self, scope):
        return self.ncls_all if scope is None else self.ncls[scope]

    def fit(self, k, cfg, scope):
        key = (scope, k, tuple(cfg) if cfg else ())
        if key in self.cache:
            return self.cache[key]
        m = self._rows(scope)
        P = self.P[:, m] if self.m1 else self.P[:, m, :]
        y = self.y[m]
        try:
            if k == "lr":
                node = F.fit_lr_m1(P, y)
            elif k == "poly":
                node = F.fit_poly_m1(P, y, 2)
            elif k == "rff":
                node = F.fit_rff_m1(P, y, cfgs=[tuple(cfg)])
            elif k == "lr3":
                node = F.fit_lr3_m2(P, y)
            elif k == "poly3":
                node = F.fit_poly3_m2(P, y, 2)
            elif k == "rff3":
                node = F.fit_rff3_m2(P, y, dim=int(cfg[0]), gamma=float(cfg[1]))
            else:
                node = None
        except Exception:
            node = None
        self.cache[key] = node
        return node

    # --------------------------------------------------------------- drawing programs
    def simple_types(self, scope):
        n, nc = self.n_rows(scope), self.n_classes(scope)
        if self.m1:
            t = ["sim", "logitmean", "rankmean", "wmean", "bestfm", "blend"]
            if self.car is not None:
                t.append("car")
            if nc >= 2 and n >= 12:
                t.append("lr")
            if nc >= 2 and n >= 40:
                t.append("poly")
            if scope is None and nc >= 2 and n >= 40:
                t.append("rff")
        else:
            t = ["sim", "rankmean3", "wmean", "bestfm", "blend"]
            if self.car is not None:
                t.append("car")
            if nc >= 2 and n >= 18:
                t.append("lr3")
            if scope is None and nc >= 2 and n >= 40:
                t += ["poly3", "rff3"]
        return t

    def rand_simple(self, rng, scope):
        t = self.simple_types(scope)[int(rng.integers(len(self.simple_types(scope))))]
        if t in ("sim", "logitmean", "rankmean", "rankmean3"):
            return {"t": t}
        if t == "wmean":
            w = self.w_auc if rng.random() < 0.5 else [round(float(v), 3) for v in rng.dirichlet(np.ones(self.K))]
            return {"t": "wmean" if self.m1 else "wmean3", "w": list(w)}
        if t == "car":
            return {"t": "wmean" if self.m1 else "wmean3", "w": list(self.car)}
        if t == "bestfm":
            return {"t": "bestfm", "idx": int(rng.integers(self.K))}
        if t == "blend":
            return {"t": "blend", "idx": int(rng.integers(self.K)), "lam": GRID_LAM[int(rng.integers(len(GRID_LAM)))]}
        if t in ("rff", "rff3"):
            return {"t": "fit", "k": t, "cfg": list(RFF_CFGS[int(rng.integers(len(RFF_CFGS)))])}
        return {"t": "fit", "k": t}

    def rand_routed(self, rng):
        node = {"t": "routed", "cells": {c: self.rand_simple(rng, c) for c in self.cancers},
                "default": {"t": "sim"}}
        if self.protect and rng.random() < 0.25:
            node = {"t": "protect", "base": node, "overrides": dict(self.protect)}
        return node

    def rand_node(self, rng):
        r = rng.random()
        if self.multi:
            if r < 0.40:
                return self.rand_simple(rng, None)
            if r < 0.75:
                return self.rand_mix(rng)
            return self.rand_routed(rng)
        return self.rand_simple(rng, None) if r < 0.55 else self.rand_mix(rng)

    def rand_mix(self, rng):
        a = self.rand_routed(rng) if (self.multi and rng.random() < 0.4) else self.rand_simple(rng, None)
        return {"t": "mix", "a": a, "b": self.rand_simple(rng, None), "g": GRID_G[int(rng.integers(len(GRID_G)))]}

    def calib_choices(self):
        return ["none", "global", "pc"] if self.multi else ["none", "global"]

    def rand_genome(self, rng):
        c = self.calib_choices()
        return {"base": self.rand_node(rng), "calib": c[int(rng.integers(len(c)))]}

    # --------------------------------------------------------------- building and scoring
    def materialize(self, node, scope=None):
        """The node with every learner fitted on the rows where it sits; None when a fit is not possible."""
        t = node["t"]
        if t == "fit":
            return self.fit(node["k"], node.get("cfg"), scope)
        if t == "mix":
            a = self.materialize(node["a"], scope); b = self.materialize(node["b"], scope)
            if a is None or b is None:
                return None
            return {"t": "mix", "a": a, "b": b, "g": float(node["g"])}
        if t == "routed":
            cells = {}
            for c, sub in node["cells"].items():
                m = self.materialize(sub, c)
                if m is None:
                    return None
                cells[c] = m
            return {"t": "routed", "cells": cells, "default": self.materialize(node["default"], scope)}
        if t == "protect":
            b = self.materialize(node["base"], scope)
            return None if b is None else {"t": "protect", "base": b, "overrides": dict(node["overrides"])}
        return copy.deepcopy(node)

    def build_spec(self, g):
        """The SPEC of genome `g`: fitted nodes, and the calibration fitted on the training cohort."""
        base = self.materialize(g["base"])
        if base is None:
            return None
        spec = {"model": self.model, "fm_order": self.fo, "base": base}
        if not self.m1:
            spec["renorm"] = True
        if g["calib"] == "none":
            return spec
        apply = interp._ns["_base_m1"] if self.m1 else interp._ns["_base_m2"]
        bb = apply(base, self.P, self.ct)
        spec = dict(spec)
        if g["calib"] == "global":
            spec["calib"], _renorm = F.calib_global(bb, self.y, self.model)
        else:
            spec["calib"], _renorm = F.calib_per_cancer(bb, self.y, self.ct, self.model)
        if not self.m1:
            spec["renorm"] = False                     # a calibrated Model_2 program is not renormalized
        return spec

    def score_spec(self, spec):
        bl = interp.run_blend(spec, self.P, self.fo, self.ct, None)
        with scoring.bootstrap_size(scoring.SEARCH_BOOTSTRAP):
            return scoring.evaluate(self.co, bl)

    @staticmethod
    def key(e):
        return scoring.obj_key(e)


# ------------------------------------------------------------------ operators on genomes
def _positions(g):
    """(path, kind, scope) of every position that can be replaced. kind: root | mix_a | mix_b | cell."""
    out = [(("base",), "root", None)]

    def walk(node, path):
        t = node["t"]
        if t == "mix":
            out.append((path + ("a",), "mix_a", None)); out.append((path + ("b",), "mix_b", None))
            walk(node["a"], path + ("a",))
        elif t == "protect":
            walk(node["base"], path + ("base",))
        elif t == "routed":
            for c in node["cells"]:
                out.append((path + ("cells", c), "cell", c))
    walk(g["base"], ("base",))
    return out


def _get(g, path):
    x = g
    for k in path:
        x = x[k]
    return x


def _set(g, path, val):
    x = g
    for k in path[:-1]:
        x = x[k]
    x[path[-1]] = val


def _routed_nodes(g):
    res = []

    def walk(node, path):
        if node["t"] == "routed":
            res.append(path)
        elif node["t"] == "mix":
            walk(node["a"], path + ("a",))
        elif node["t"] == "protect":
            walk(node["base"], path + ("base",))
    walk(g["base"], ("base",))
    return res


def _fresh(sp, rng, kind, scope):
    if kind == "root":
        return sp.rand_node(rng)
    if kind == "mix_a":
        return sp.rand_routed(rng) if (sp.multi and rng.random() < 0.4) else sp.rand_simple(rng, None)
    return sp.rand_simple(rng, scope)


def _eligible(sp, node, scope):
    if node["t"] != "fit":
        return True
    return node["k"] in sp.simple_types(scope)


def repair(sp, g):
    """Replace a node that is not allowed where it sits (for example a polynomial moved into a cancer type with too
    few rows) with the mean, so that every genome can be built."""
    for path, kind, scope in _positions(g):
        node = _get(g, path)
        if kind in ("cell", "mix_b") and node["t"] in ("mix", "routed", "protect"):
            _set(g, path, {"t": "sim"}); continue
        if node["t"] == "fit" and not _eligible(sp, node, scope):
            _set(g, path, {"t": "sim"})
    return g


def mutate(sp, rng, g):
    """Point mutation: change one parameter, replace one subtree, redraw one cancer type's node, or change the
    calibration."""
    g = copy.deepcopy(g)
    ops = ["param", "subtree"] + (["calib"] if len(sp.calib_choices()) > 1 else []) + \
          (["cell"] if _routed_nodes(g) else [])
    w = {"param": 0.35, "subtree": 0.30, "calib": 0.15, "cell": 0.20}
    p = np.array([w[o] for o in ops]); p = p / p.sum()
    op = ops[int(rng.choice(len(ops), p=p))]
    if op == "param":
        tun = []

        def walk(node, path):
            t = node["t"]
            if t in ("blend", "bestfm", "wmean", "wmean3", "mix") or (t == "fit" and node.get("cfg")):
                tun.append(path)
            if t == "mix":
                walk(node["a"], path + ("a",)); walk(node["b"], path + ("b",))
            elif t == "protect":
                walk(node["base"], path + ("base",))
            elif t == "routed":
                for c in node["cells"]:
                    walk(node["cells"][c], path + ("cells", c))
        walk(g["base"], ("base",))
        if not tun:
            op = "subtree"
        else:
            path = tun[int(rng.integers(len(tun)))]
            node = _get(g, path)
            t = node["t"]
            if t == "blend":
                if rng.random() < 0.3:
                    node["idx"] = int(rng.integers(sp.K))
                else:
                    node["lam"] = round(float(np.clip(node["lam"] + rng.choice([-0.1, 0.1]), 0.0, 1.0)), 2)
            elif t == "bestfm":
                node["idx"] = int(rng.integers(sp.K))
            elif t in ("wmean", "wmean3"):
                w_ = np.clip(np.asarray(node["w"], float) + rng.normal(0, 0.1, len(node["w"])), 0, None)
                if w_.sum() <= 0:
                    w_ = np.ones(len(node["w"]))
                node["w"] = [round(float(v), 3) for v in w_ / w_.sum()]
            elif t == "mix":
                node["g"] = round(float(np.clip(node["g"] + rng.choice([-0.1, -0.05, 0.05, 0.1]), 0.1, 0.9)), 2)
            elif t == "fit":
                node["cfg"] = list(RFF_CFGS[int(rng.integers(len(RFF_CFGS)))])
    if op == "subtree":
        pos = _positions(g)
        path, kind, scope = pos[int(rng.integers(len(pos)))]
        if kind == "root":
            g["base"] = sp.rand_node(rng)
        else:
            _set(g, path, _fresh(sp, rng, kind, scope))
    elif op == "cell":
        rp = _routed_nodes(g)
        r = _get(g, rp[int(rng.integers(len(rp)))])
        c = sp.cancers[int(rng.integers(len(sp.cancers)))]
        r["cells"][c] = sp.rand_simple(rng, c)
    elif op == "calib":
        ch = [c for c in sp.calib_choices() if c != g["calib"]]
        g["calib"] = ch[int(rng.integers(len(ch)))]
    return repair(sp, g)


def crossover(sp, rng, g1, g2):
    """A child of g1 and g2: uniform crossover of the per-cancer nodes of two routed nodes, or a subtree of g2 put
    into g1."""
    child = copy.deepcopy(g1)
    r1, r2 = _routed_nodes(child), _routed_nodes(g2)
    if r1 and r2 and rng.random() < 0.5:
        a = _get(child, r1[int(rng.integers(len(r1)))]); b = _get(g2, r2[int(rng.integers(len(r2)))])
        for c in sp.cancers:
            if rng.random() < 0.5:
                a["cells"][c] = copy.deepcopy(b["cells"][c])
    else:
        pc = _positions(child); pd_ = _positions(g2)
        path, kind, scope = pc[int(rng.integers(len(pc)))]
        donors = [q for q in pd_ if (kind in ("root", "mix_a")) or (_get(g2, q[0])["t"] not in ("mix", "routed", "protect"))]
        if kind == "mix_a":
            donors = [q for q in donors if _get(g2, q[0])["t"] != "mix"]
        if donors:
            dp = donors[int(rng.integers(len(donors)))][0]
            val = copy.deepcopy(_get(g2, dp))
            if kind == "root":
                child["base"] = val
            else:
                _set(child, path, val)
    if rng.random() < 0.5:
        child["calib"] = g2["calib"]
    return repair(sp, child)


def canon(g):
    return json.dumps(g, sort_keys=True)
