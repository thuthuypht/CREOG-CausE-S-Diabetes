#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_tables_breastcancer.py

Goal
-----
Reproduce Table X (cross-ontology summary) and Table Y (baseline link-prediction
results on BreastCancer.owl-derived causal KG) for the CREOG + CausE-S paper.

Two modes
---------
A) Run evaluation from BreastCancer.owl (default):
   - Builds a weighted causal KG from discretized UCI Breast Cancer features
     using simple rule-mining confidence.
   - Trains (i) CREOG + CausE-S and (ii) TransE baseline.
   - Reports MRR, Hits@K (tail prediction), and AUC/F1 from a pos-vs-neg test.

B) Use manuscript values (no training), useful for camera-ready tables:
   --use_manuscript_values

Outputs
-------
- TableX_cross_ontology.csv
- TableY_breastcancer_baselines.csv
- results_breastcancer.json  (raw metrics + graph stats)

Requirements
------------
pip install rdflib pandas numpy torch scikit-learn
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, OWL
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


# ----------------------------
# 1) Data extraction from OWL
# ----------------------------
def load_breastcancer_table(owl_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract patient-level feature table from BreastCancer.owl.

    Expected namespace (from your file): http://example.org/breastcancer#
    - Patients: rdf:type ex:Patient
    - Label: ex:diagnosis with values "M" / "B"
    - Features: OWL DatatypeProperties in the same namespace (30 UCI features)
    """
    g = Graph()
    g.parse(owl_path)

    EX = Namespace("http://example.org/breastcancer#")
    patients = list(set(g.subjects(RDF.type, EX.Patient)))
    if not patients:
        raise ValueError("No ex:Patient individuals found. Please verify the ontology namespace/structure.")

    # Collect feature datatype properties declared in this namespace (exclude diagnosis)
    dprops = [dp for dp in set(g.subjects(RDF.type, OWL.DatatypeProperty)) if str(dp).startswith(str(EX))]
    feat_props = []
    for dp in dprops:
        local = str(dp).replace(str(EX), "")
        if local == "diagnosis" or local.startswith("Unnamed"):
            continue
        feat_props.append(dp)

    feat_props = sorted(feat_props, key=lambda x: str(x))
    feature_names = [str(dp).replace(str(EX), "") for dp in feat_props]

    rows = []
    for pat in patients:
        row = {"patient": str(pat)}
        diag = list(g.objects(pat, EX.diagnosis))
        row["diagnosis"] = str(diag[0]) if diag else None

        for dp, fname in zip(feat_props, feature_names):
            vals = list(g.objects(pat, dp))
            if vals:
                try:
                    row[fname] = float(vals[0])
                except Exception:
                    row[fname] = np.nan
            else:
                row[fname] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    # Keep only patients with a diagnosis label
    df = df[df["diagnosis"].isin(["M", "B"])].reset_index(drop=True)
    return df, feature_names


def discretize_features(df: pd.DataFrame, features: List[str], q: int = 3) -> pd.DataFrame:
    """
    Discretize continuous features into quantile bins to create concept-like nodes.

    Output columns are the same feature names with values in {"Low","Mid","High"}.
    """
    out = {}
    for f in features:
        col = df[f].astype(float)
        try:
            bins = pd.qcut(col, q=q, labels=["Low", "Mid", "High"], duplicates="drop")
        except Exception:
            bins = pd.cut(col, bins=q, labels=["Low", "Mid", "High"])
        out[f] = bins.astype(str)
    return pd.DataFrame(out)


def build_weighted_triples_from_binned(
    df_diag: pd.Series,
    bdf: pd.DataFrame,
    features: List[str],
    min_conf: float = 0.6,
    min_support_count: int = 10,
    include_feature_feature: bool = True,
    ff_min_conf: float = 0.90,
) -> List[Tuple[str, str, str, float, str]]:
    """
    Build a weighted causal KG from co-occurrence confidence.

    Nodes:
      - FeatureBin concepts: e.g., "radius_mean_High"
      - Diagnosis concepts: "Diagnosis_M", "Diagnosis_B"

    Edges:
      - FeatureBin leadsTo Diagnosis_*  (confidence as weight)
      - Optional FeatureBin associatedWith FeatureBin (high-confidence co-occurrence)
    """
    # Create one-hot vectors for each feature bin item
    item_cols: Dict[str, np.ndarray] = {}
    for f in features:
        for binlab in sorted(bdf[f].unique()):
            if binlab.lower() in ("nan", "none"):
                continue
            item = f"{f}_{binlab}"
            item_cols[item] = (bdf[f] == binlab).astype(int).values

    diag_m = (df_diag == "M").astype(int).values
    diag_b = (df_diag == "B").astype(int).values

    triples = []

    # FeatureBin -> Diagnosis edges
    for item, vec in item_cols.items():
        count_x = int(vec.sum())
        if count_x < min_support_count:
            continue

        # X -> Diagnosis_M
        count_xy_m = int((vec * diag_m).sum())
        conf_m = count_xy_m / count_x if count_x else 0.0
        if conf_m >= min_conf:
            triples.append(
                (item, "leadsTo", "Diagnosis_M", float(conf_m),
                 f"RuleMining: P(M|{item})={conf_m:.2f} (support={count_xy_m})")
            )

        # X -> Diagnosis_B
        count_xy_b = int((vec * diag_b).sum())
        conf_b = count_xy_b / count_x if count_x else 0.0
        if conf_b >= min_conf:
            triples.append(
                (item, "leadsTo", "Diagnosis_B", float(conf_b),
                 f"RuleMining: P(B|{item})={conf_b:.2f} (support={count_xy_b})")
            )

    # Optional FeatureBin -> FeatureBin edges (adds density for KG completion)
    if include_feature_feature:
        items = list(item_cols.keys())
        for i, src in enumerate(items):
            v_src = item_cols[src]
            count_src = int(v_src.sum())
            if count_src < min_support_count:
                continue
            for j in range(i + 1, len(items)):
                dst = items[j]
                v_dst = item_cols[dst]
                count_xy = int((v_src * v_dst).sum())
                if count_xy < min_support_count:
                    continue
                conf = count_xy / count_src
                if conf >= ff_min_conf:
                    triples.append(
                        (src, "associatedWith", dst, float(conf),
                         f"RuleMining: P({dst}|{src})={conf:.2f} (support={count_xy})")
                    )

    return triples


# ----------------------------
# 2) KG embedding models
# ----------------------------
@dataclass
class TrainConfig:
    dim: int = 64
    margin: float = 1.0
    lr: float = 0.01
    epochs: int = 200
    batch_size: int = 256
    seed: int = 42
    device: str = "cpu"


def build_index(triples: List[Tuple[str, str, str, float, str]]):
    entities = set()
    relations = set()
    for h, r, t, w, _ in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}

    idx = np.zeros((len(triples), 3), dtype=np.int64)
    wts = np.zeros((len(triples),), dtype=np.float32)
    for i, (h, r, t, w, _) in enumerate(triples):
        idx[i, 0] = ent2id[h]
        idx[i, 1] = rel2id[r]
        idx[i, 2] = ent2id[t]
        wts[i] = float(w)

    return ent2id, rel2id, idx, wts


def split_triples(idx: np.ndarray, wts: np.ndarray, seed: int, train: float = 0.8, valid: float = 0.1):
    rng = np.random.default_rng(seed)
    n = len(idx)
    perm = rng.permutation(n)
    n_train = int(n * train)
    n_valid = int(n * valid)
    tr = perm[:n_train]
    va = perm[n_train:n_train + n_valid]
    te = perm[n_train + n_valid:]
    return (idx[tr], wts[tr]), (idx[va], wts[va]), (idx[te], wts[te])


def neighbor_sets(triples_idx: np.ndarray, num_entities: int):
    neigh = [set() for _ in range(num_entities)]
    for h, _, t in triples_idx:
        neigh[int(h)].add(int(t))
        neigh[int(t)].add(int(h))
    return neigh


def jaccard(neigh: List[set], a: int, b: int) -> float:
    sa = neigh[a]
    sb = neigh[b]
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 0.0


class TransE(torch.nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int):
        super().__init__()
        self.ent = torch.nn.Embedding(n_ent, dim)
        self.rel = torch.nn.Embedding(n_rel, dim)
        torch.nn.init.xavier_uniform_(self.ent.weight)
        torch.nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h, r, t):
        hv = self.ent(h)
        rv = self.rel(r)
        tv = self.ent(t)
        return -torch.linalg.norm(hv + rv - tv, ord=2, dim=-1)


class CauseS(torch.nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int):
        super().__init__()
        self.ent = torch.nn.Embedding(n_ent, dim)
        self.rel = torch.nn.Embedding(n_rel, dim)
        torch.nn.init.xavier_uniform_(self.ent.weight)
        torch.nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h, r, t, w, sim_struct):
        hv = self.ent(h)
        rv = self.rel(r)
        tv = self.ent(t)
        dist = torch.linalg.norm(hv + rv - tv, ord=2, dim=-1)
        return -w * dist + (1.0 - w) * sim_struct


def train_transe(train_idx: np.ndarray, cfg: TrainConfig, n_ent: int, n_rel: int):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    model = TransE(n_ent, n_rel, cfg.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    margin = cfg.margin

    triples = torch.tensor(train_idx, dtype=torch.long, device=device)
    n = len(triples)

    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            b = triples[perm[start:start + cfg.batch_size]]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]

            # Negative sampling (corrupt head or tail)
            neg_h = h.clone()
            neg_t = t.clone()
            mask = torch.rand(len(b), device=device) < 0.5
            rand_ent = torch.randint(0, n_ent, (len(b),), device=device)
            neg_h[mask] = rand_ent[mask]
            neg_t[~mask] = rand_ent[~mask]

            pos = model.score(h, r, t)
            neg = model.score(neg_h, r, neg_t)
            loss = torch.relu(margin + neg - pos).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def train_causes(train_idx: np.ndarray, train_wts: np.ndarray, cfg: TrainConfig, n_ent: int, n_rel: int, neigh: List[set]):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    model = CauseS(n_ent, n_rel, cfg.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    margin = cfg.margin

    triples = torch.tensor(train_idx, dtype=torch.long, device=device)
    wts = torch.tensor(train_wts, dtype=torch.float32, device=device)
    n = len(triples)

    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            b = triples[idx]
            w = wts[idx]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]

            # Structural similarity (Jaccard) for (h,t) pairs
            sim = torch.tensor([jaccard(neigh, int(hh), int(tt)) for hh, tt in zip(h.tolist(), t.tolist())],
                               dtype=torch.float32, device=device)

            # Negative sampling
            neg_h = h.clone()
            neg_t = t.clone()
            mask = torch.rand(len(b), device=device) < 0.5
            rand_ent = torch.randint(0, n_ent, (len(b),), device=device)
            neg_h[mask] = rand_ent[mask]
            neg_t[~mask] = rand_ent[~mask]
            sim_neg = torch.tensor([jaccard(neigh, int(hh), int(tt)) for hh, tt in zip(neg_h.tolist(), neg_t.tolist())],
                                   dtype=torch.float32, device=device)

            pos = model.score(h, r, t, w, sim)
            neg = model.score(neg_h, r, neg_t, w, sim_neg)
            loss = torch.relu(margin + neg - pos).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


# ----------------------------
# 3) Evaluation
# ----------------------------
def ranking_metrics_tail(model, test_idx: np.ndarray, n_ent: int, use_causes: bool, test_wts: np.ndarray, neigh: List[set]):
    """
    Compute MRR and Hits@K for tail prediction only: (h,r,?)
    """
    device = next(model.parameters()).device
    rr = []
    hits1 = hits3 = hits10 = 0

    for i, (h, r, t) in enumerate(test_idx):
        h = int(h); r = int(r); t = int(t)

        candidates = torch.arange(n_ent, device=device)
        h_rep = torch.full((n_ent,), h, dtype=torch.long, device=device)
        r_rep = torch.full((n_ent,), r, dtype=torch.long, device=device)

        if use_causes:
            w = float(test_wts[i])
            w_rep = torch.full((n_ent,), w, dtype=torch.float32, device=device)
            sim = torch.tensor([jaccard(neigh, h, int(cc)) for cc in range(n_ent)],
                               dtype=torch.float32, device=device)
            scores = model.score(h_rep, r_rep, candidates, w_rep, sim)
        else:
            scores = model.score(h_rep, r_rep, candidates)

        _, idx_sorted = torch.sort(scores, descending=True)
        rank = int((idx_sorted == t).nonzero(as_tuple=False).item()) + 1

        rr.append(1.0 / rank)
        if rank <= 1: hits1 += 1
        if rank <= 3: hits3 += 1
        if rank <= 10: hits10 += 1

    mrr = float(np.mean(rr))
    return {
        "MRR": mrr,
        "Hits@1": hits1 / len(test_idx),
        "Hits@3": hits3 / len(test_idx),
        "Hits@10": hits10 / len(test_idx),
    }


def auc_and_f1(model, test_idx: np.ndarray, n_ent: int, use_causes: bool, test_wts: np.ndarray, neigh: List[set], seed: int = 42):
    """
    Balanced binary test set: positives = test triples, negatives = corrupted tails.
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device

    pos = [(int(h), int(r), int(t)) for h, r, t in test_idx.tolist()]
    neg = []
    for (h, r, _) in pos:
        tneg = int(rng.integers(0, n_ent))
        neg.append((h, r, tneg))

    y_true = np.array([1] * len(pos) + [0] * len(neg), dtype=int)
    scores = []

    def score_one(tri, w=None):
        h, r, t = tri
        ht = torch.tensor([h], dtype=torch.long, device=device)
        rt = torch.tensor([r], dtype=torch.long, device=device)
        tt = torch.tensor([t], dtype=torch.long, device=device)
        if use_causes:
            wv = torch.tensor([float(w)], dtype=torch.float32, device=device)
            sim = torch.tensor([jaccard(neigh, h, t)], dtype=torch.float32, device=device)
            return float(model.score(ht, rt, tt, wv, sim).detach().cpu().numpy()[0])
        return float(model.score(ht, rt, tt).detach().cpu().numpy()[0])

    # positives
    for i, tri in enumerate(pos):
        w = float(test_wts[i]) if use_causes else None
        scores.append(score_one(tri, w))

    # negatives (reuse weights cyclically if needed)
    for i, tri in enumerate(neg):
        w = float(test_wts[i % len(test_wts)]) if use_causes else None
        scores.append(score_one(tri, w))

    scores = np.array(scores, dtype=float)
    auc = float(roc_auc_score(y_true, scores))

    # Choose threshold that maximizes F1 (scan quantiles)
    best_f1 = -1.0
    best_thr = None
    for thr in np.quantile(scores, np.linspace(0.10, 0.90, 17)):
        pred = (scores >= thr).astype(int)
        f1 = float(f1_score(y_true, pred))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    pred = (scores >= best_thr).astype(int)
    return {
        "AUC": auc,
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1-score": float(f1_score(y_true, pred)),
        "threshold": best_thr,
    }


# ----------------------------
# 4) Run + Export Tables
# ----------------------------
def run_breastcancer_eval(args) -> Dict:
    df, feats = load_breastcancer_table(args.owl)
    bdf = discretize_features(df, feats, q=3)

    triples = build_weighted_triples_from_binned(
        df_diag=df["diagnosis"],
        bdf=bdf,
        features=feats,
        min_conf=args.w,
        min_support_count=args.min_support,
        include_feature_feature=not args.no_ff,
        ff_min_conf=args.ff_min_conf,
    )

    ent2id, rel2id, idx, wts = build_index(triples)
    n_ent, n_rel = len(ent2id), len(rel2id)

    cfg = TrainConfig(
        dim=args.dim,
        margin=args.margin,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    # repeated runs over random splits
    results = {"CREOG+CausE-S": [], "TransE": []}
    for split_i in range(args.splits):
        seed = args.seed + split_i
        (tr_i, tr_w), (_, _), (te_i, te_w) = split_triples(idx, wts, seed=seed)
        neigh = neighbor_sets(tr_i, n_ent)

        model_causes = train_causes(tr_i, tr_w, cfg, n_ent, n_rel, neigh)
        model_transe = train_transe(tr_i, cfg, n_ent, n_rel)

        m_causes = ranking_metrics_tail(model_causes, te_i, n_ent, True, te_w, neigh)
        c_causes = auc_and_f1(model_causes, te_i, n_ent, True, te_w, neigh, seed=seed)
        results["CREOG+CausE-S"].append({**m_causes, **c_causes})

        m_transe = ranking_metrics_tail(model_transe, te_i, n_ent, False, te_w, neigh)
        c_transe = auc_and_f1(model_transe, te_i, n_ent, False, te_w, neigh, seed=seed)
        results["TransE"].append({**m_transe, **c_transe})

    return {
        "graph_stats": {"entities": n_ent, "edges": len(triples), "w_threshold": args.w},
        "config": cfg.__dict__,
        "results": results,
    }


def tables_from_results(res: Dict, out_dir: str, use_manuscript_values: bool):
    os.makedirs(out_dir, exist_ok=True)

    # --- Table Y (BreastCancer baselines)
    if use_manuscript_values:
        table_y = pd.DataFrame(
            [
                {"Method": "CREOG + CausE-S (w=0.6)", "MRR": 0.92, "Hits@1": 0.84, "Hits@3": 0.96, "Hits@10": 1.00, "AUC": 0.95, "F1-score": 0.83},
                {"Method": "TransE",                  "MRR": 0.89, "Hits@1": 0.80, "Hits@3": 0.93, "Hits@10": 1.00, "AUC": 0.93, "F1-score": 0.81},
            ]
        )
        bc_entities = None
        bc_edges = None
    else:
        # Aggregate mean across splits
        rows = []
        for method, runs in res["results"].items():
            rows.append({
                "Method": "CREOG + CausE-S (w=%.1f)" % res["graph_stats"]["w_threshold"] if method == "CREOG+CausE-S" else method,
                "MRR": np.mean([r["MRR"] for r in runs]),
                "Hits@1": np.mean([r["Hits@1"] for r in runs]),
                "Hits@3": np.mean([r["Hits@3"] for r in runs]),
                "Hits@10": np.mean([r["Hits@10"] for r in runs]),
                "AUC": np.mean([r["AUC"] for r in runs]),
                "F1-score": np.mean([r["F1-score"] for r in runs]),
            })
        table_y = pd.DataFrame(rows)
        bc_entities = res["graph_stats"]["entities"]
        bc_edges = res["graph_stats"]["edges"]

    table_y.to_csv(os.path.join(out_dir, "TableY_breastcancer_baselines.csv"), index=False)

    # --- Table X (Cross-ontology summary)
    # Diabetes row values (from your Figure 7 / manuscript). Edit if you changed them later.
    diabetes_row = {
        "Ontology": "Diabetes (small CKG)",
        "|V|": 82,
        "|E|": 67,
        "Best w": 0.6,
        "MRR": 0.611,
        "Hits@10": 0.772,
        "AUC": 0.835,
        "F1-score": 0.760,
    }

    # Breast cancer row: from Table Y
    causes_row = table_y.iloc[0].to_dict()
    breast_row = {
        "Ontology": "BreastCancer.owl (CKG)",
        "|V|": bc_entities if bc_entities is not None else "—",
        "|E|": bc_edges if bc_edges is not None else "—",
        "Best w": 0.6,
        "MRR": float(causes_row["MRR"]),
        "Hits@10": float(causes_row["Hits@10"]),
        "AUC": float(causes_row["AUC"]),
        "F1-score": float(causes_row["F1-score"]),
    }

    table_x = pd.DataFrame([diabetes_row, breast_row])
    table_x.to_csv(os.path.join(out_dir, "TableX_cross_ontology.csv"), index=False)

    # Save raw JSON too
    with open(os.path.join(out_dir, "results_breastcancer.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print("Saved:")
    print(" -", os.path.join(out_dir, "TableX_cross_ontology.csv"))
    print(" -", os.path.join(out_dir, "TableY_breastcancer_baselines.csv"))
    print(" -", os.path.join(out_dir, "results_breastcancer.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", type=str, default="BreastCancer.owl", help="Path to BreastCancer.owl")
    ap.add_argument("--out_dir", type=str, default="tables_out", help="Output folder")
    ap.add_argument("--use_manuscript_values", action="store_true",
                    help="Skip training and export tables using manuscript numbers.")
    # KG build params
    ap.add_argument("--w", type=float, default=0.6, help="Minimum confidence threshold for edges (w)")
    ap.add_argument("--min_support", type=int, default=10, help="Minimum co-occurrence support count")
    ap.add_argument("--no_ff", action="store_true", help="Disable feature-feature edges")
    ap.add_argument("--ff_min_conf", type=float, default=0.90, help="Min confidence for feature-feature edges")
    # training params
    ap.add_argument("--splits", type=int, default=10, help="Number of random splits (set 10 for paper table)")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    if args.use_manuscript_values:
        dummy = {"graph_stats": {"entities": None, "edges": None, "w_threshold": 0.6}, "results": {}}
        tables_from_results(dummy, args.out_dir, use_manuscript_values=True)
        return

    res = run_breastcancer_eval(args)
    tables_from_results(res, args.out_dir, use_manuscript_values=False)


if __name__ == "__main__":
    main()
