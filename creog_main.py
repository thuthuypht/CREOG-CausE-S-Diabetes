"""
CREOG – Causal Reasoning over Ontology-Enriched Graphs with CausE-S
===================================================================

Ontology-only variant:
- Input: an OWL file (e.g., Diabetes_large.owl) containing causal relations
  (causes / leadsTo / associatedWith) plus optional confidence scores and comments.
- Output:
    * Weighted causal knowledge graph (CKG) as CSV
    * Entity & relation embeddings (NumPy .npy)
    * Predicted causal links with explanations (JSON)
    * Metrics: MRR, Hits@K, AUC, Precision, Recall, F1 (JSON)

Run from Anaconda Prompt:

    cd /d D:\CREOG-CausE-S-Diabetes
    python creog_main.py --owl_path data\Diabetes_large.owl --output_dir outputs
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, OWL, RDFS
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm

# ==========================
# Hyperparameters (HYP_EMB, HYP_EVAL)
# ==========================

DEFAULT_HYP_EMB = {
    "embedding_dim": 64,
    "margin_gamma": 1.0,
    "learning_rate_emb": 0.001,
    "num_epochs": 30,
    "batch_size": 256,
    "num_negatives": 5,
    "structural_similarity_type": "Jaccard",
}

DEFAULT_HYP_EVAL = {
    "weight_thresholds": [0.5],
    "hits_ks": [1, 3, 10],
    "num_splits": 1,
    "metrics_list": ["MRR", "Hits@K", "AUC", "Precision", "Recall", "F1"],
}

# local names of relations we treat as causal edges
CAUSAL_RELATIONS_LOCAL = {"causes", "leadsTo", "associatedWith"}


# ==========================
# Utility: local name from URI
# ==========================

def local_name(uri: URIRef) -> str:
    """
    Extract local name from a URI: last fragment after '#' or '/'.
    """
    uri_str = str(uri)
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    return uri_str.rstrip("/").split("/")[-1]


# ==========================
# PHASE 4 – Load ontology & extract weighted triples
# ==========================

def load_weighted_triples_from_owl(
    owl_path: str,
    confidence_property_local: str = "hasConfidenceScore",
) -> List[Dict]:
    """
    Parse an OWL file and extract weighted causal triples.

    We let rdflib auto-detect the format. If that fails, we retry with
    RDF/XML (format="xml"). This avoids the 'application/owl+xml' plugin error.

    Assumes:
    - Object properties of interest have local names in CAUSAL_RELATIONS_LOCAL
      (e.g., causes, leadsTo, associatedWith).
    - Weights are specified via owl:Axiom with:
        owl:annotatedSource s
        owl:annotatedProperty p
        owl:annotatedTarget o
        :hasConfidenceScore "float"^^xsd:float
        rdfs:comment "text"
    """
    print(f"[INFO] Loading ontology from {owl_path}")
    g = Graph()

    # Try auto-detection first
    try:
        g.parse(owl_path)
        print("[INFO] Parsed OWL file with rdflib auto-detection.")
    except Exception as e_auto:
        print("[WARN] Auto-detection failed, trying RDF/XML (format='xml').")
        try:
            g.parse(owl_path, format="xml")
            print("[INFO] Parsed OWL file as RDF/XML (format='xml').")
        except Exception as e_xml:
            print("[ERROR] Could not parse ontology file with rdflib.")
            print("        Please ensure Diabetes_large.owl is saved as RDF/XML in Protégé.")
            print(f"        Auto-detect error: {e_auto}")
            print(f"        RDF/XML error    : {e_xml}")
            raise

    # Build a map from (s,p,o) -> (weight, comment) using annotation axioms
    print("[INFO] Extracting annotation axioms for edge metadata...")
    confidence_pred = None
    for s, p, o in g.triples((None, None, None)):
        # find URI of the confidence property by local name
        if isinstance(p, URIRef) and local_name(p) == confidence_property_local:
            confidence_pred = p
            break
    if confidence_pred is None:
        print(f"[WARN] Confidence property {confidence_property_local} not found; "
              f"default weight=1.0 will be used where missing.")

    axiom_metadata: Dict[Tuple[URIRef, URIRef, URIRef], Dict] = {}

    for ax in g.subjects(RDF.type, OWL.Axiom):
        ann_source = None
        ann_prop = None
        ann_target = None
        weight = None
        comment = ""
        for _, p, o in g.triples((ax, None, None)):
            lname = local_name(p)
            if lname == "annotatedSource":
                ann_source = o
            elif lname == "annotatedProperty":
                ann_prop = o
            elif lname == "annotatedTarget":
                ann_target = o
            elif confidence_pred is not None and p == confidence_pred:
                try:
                    weight = float(o)
                except Exception:
                    weight = None
            elif p == RDFS.comment:
                comment = str(o)

        if ann_source is not None and ann_prop is not None and ann_target is not None:
            key = (ann_source, ann_prop, ann_target)
            axiom_metadata[key] = {
                "weight": weight,
                "comment": comment,
            }

    print(f"[INFO] Found {len(axiom_metadata)} annotated axioms with metadata")

    # Extract causal triples and attach weights/comments if available
    print("[INFO] Extracting causal triples...")
    triples_weighted: List[Dict] = []
    for s, p, o in g.triples((None, None, None)):
        if not isinstance(p, URIRef):
            continue
        if local_name(p) not in CAUSAL_RELATIONS_LOCAL:
            continue
        key = (s, p, o)
        meta = axiom_metadata.get(key, {})
        w = meta.get("weight", 1.0)
        comment = meta.get("comment", "")

        triples_weighted.append(
            {
                "h": local_name(s),
                "r": local_name(p),
                "t": local_name(o),
                "w": float(w),
                "comment": comment,
            }
        )

    print(f"[INFO] Extracted {len(triples_weighted)} weighted causal triples")
    return triples_weighted


# ==========================
# Build CKG (graph) & structural similarity
# ==========================

def build_ckg_and_vocab(triples_weighted: List[Dict]):
    """
    Build NetworkX DiGraph and entity/relation vocabularies.
    """
    print("[INFO] Building CKG graph and vocabularies...")
    G = nx.DiGraph()
    entities = set()
    relations = set()

    for tr in triples_weighted:
        h, r, t, w, comment = tr["h"], tr["r"], tr["t"], tr["w"], tr["comment"]
        entities.add(h)
        entities.add(t)
        relations.add(r)
        G.add_edge(
            h,
            t,
            relation=r,
            weight=w,
            explanation=comment,
        )

    entity2id = {e: idx for idx, e in enumerate(sorted(entities))}
    id2entity = {idx: e for e, idx in entity2id.items()}
    relation2id = {r: idx for idx, r in enumerate(sorted(relations))}
    id2relation = {idx: r for r, idx in relation2id.items()}

    print(f"[INFO] CKG: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, "
          f"{len(relation2id)} relation types")

    return G, entity2id, id2entity, relation2id, id2relation


def precompute_structural_similarity(G: nx.DiGraph, entity2id: Dict[str, int]):
    """
    Build neighbor index N(v) and compute Jaccard similarity for each (h,t)
    that appears in a triple.

    Returns:
        sim_struct[(h_id, t_id)] = float in [0,1]
    """
    print("[INFO] Precomputing structural similarity (Jaccard)...")

    neighbors: Dict[str, set] = defaultdict(set)
    for u, v in G.edges():
        neighbors[u].add(v)
        neighbors[v].add(u)

    sim_struct = {}
    for u, v in G.edges():
        Nu, Nv = neighbors[u], neighbors[v]
        inter = len(Nu.intersection(Nv))
        union = len(Nu.union(Nv))
        jaccard = inter / union if union > 0 else 0.0
        uid, vid = entity2id[u], entity2id[v]
        sim_struct[(uid, vid)] = jaccard
        sim_struct[(vid, uid)] = jaccard  # symmetric

    print(f"[INFO] Structural similarity computed for ~{len(sim_struct)} pairs")
    return sim_struct


# ==========================
# Train / valid / test split
# ==========================

def split_triples(triples_weighted: List[Dict], train_ratio=0.8, valid_ratio=0.1):
    """
    Simple random split of triples into train/valid/test.
    """
    import random
    random.shuffle(triples_weighted)
    n = len(triples_weighted)
    n_train = int(train_ratio * n)
    n_valid = int(valid_ratio * n)
    train = triples_weighted[:n_train]
    valid = triples_weighted[n_train:n_train + n_valid]
    test = triples_weighted[n_train + n_valid:]
    print(f"[INFO] Split triples into {len(train)} train, "
          f"{len(valid)} valid, {len(test)} test")
    return train, valid, test


# ==========================
# CausE-S model
# ==========================

class CausESModel(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim, sim_struct):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.rel_emb = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        self.sim_struct = sim_struct  # dict[(h_id, t_id)] -> float

    def forward_score(self, h_idx, r_idx, t_idx, w_eff):
        """
        Compute CausE-S score:
        score = -w_eff * ||h + r - t||_2 + (1 - w_eff) * sim_struct(h,t)
        """
        h_vec = self.entity_emb(h_idx)
        r_vec = self.rel_emb(r_idx)
        t_vec = self.entity_emb(t_idx)

        dist_vec = h_vec + r_vec - t_vec
        dist = torch.norm(dist_vec, p=2, dim=-1)

        # structural similarity lookup (fallback to 0.0 if missing)
        hs = h_idx.detach().cpu().numpy()
        ts = t_idx.detach().cpu().numpy()
        sim_vals = []
        for hi, ti in zip(hs, ts):
            sim_vals.append(self.sim_struct.get((int(hi), int(ti)), 0.0))
        sim_struct = torch.tensor(sim_vals, dtype=torch.float32, device=h_idx.device)

        score = -w_eff * dist + (1.0 - w_eff) * sim_struct
        return score


def make_batches(triples, batch_size):
    """
    Yield mini-batches of triples.
    """
    import math
    n = len(triples)
    num_batches = math.ceil(n / batch_size)
    for i in range(num_batches):
        batch = triples[i * batch_size:(i + 1) * batch_size]
        yield batch


def sample_negatives(num_negatives, h, r, t, entities):
    """
    Simple negative sampling by corrupting tail.
    """
    import random
    neg_triples = []
    for _ in range(num_negatives):
        t_neg = random.choice(entities)
        while t_neg == t:
            t_neg = random.choice(entities)
        neg_triples.append((h, r, t_neg))
    return neg_triples


def train_cause_s(
    train_triples,
    valid_triples,
    entity2id,
    relation2id,
    sim_struct,
    hyp_emb,
    mode="full",  # "full", "w_equal_1", "w_equal_0"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_entities = len(entity2id)
    num_relations = len(relation2id)

    model = CausESModel(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=hyp_emb["embedding_dim"],
        sim_struct=sim_struct,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyp_emb["learning_rate_emb"])

    print(f"[INFO] Training CausE-S in mode={mode} on {len(train_triples)} triples")
    model.train()
    entities_list = list(entity2id.keys())

    for epoch in range(1, hyp_emb["num_epochs"] + 1):
        total_loss = 0.0
        for batch in make_batches(train_triples, hyp_emb["batch_size"]):
            optimizer.zero_grad()
            loss_batch = 0.0

            for tr in batch:
                h_name, r_name, t_name = tr["h"], tr["r"], tr["t"]
                w = tr["w"]

                if mode == "full":
                    w_eff = w
                elif mode == "w_equal_1":
                    w_eff = 1.0
                else:  # "w_equal_0"
                    w_eff = 0.0

                h_idx = torch.tensor([entity2id[h_name]], dtype=torch.long, device=device)
                t_idx = torch.tensor([entity2id[t_name]], dtype=torch.long, device=device)
                r_idx = torch.tensor([relation2id[r_name]], dtype=torch.long, device=device)
                w_tensor = torch.tensor([w_eff], dtype=torch.float32, device=device)

                # positive score
                f_pos = model.forward_score(h_idx, r_idx, t_idx, w_tensor)

                # negatives
                neg_triples = sample_negatives(
                    hyp_emb["num_negatives"],
                    h_name,
                    r_name,
                    t_name,
                    entities_list,
                )
                for (h_neg, r_neg, t_neg) in neg_triples:
                    h_neg_idx = torch.tensor(
                        [entity2id[h_neg]], dtype=torch.long, device=device
                    )
                    t_neg_idx = torch.tensor(
                        [entity2id[t_neg]], dtype=torch.long, device=device
                    )
                    r_neg_idx = torch.tensor(
                        [relation2id[r_neg]], dtype=torch.long, device=device
                    )
                    f_neg = model.forward_score(
                        h_neg_idx, r_neg_idx, t_neg_idx, w_tensor
                    )

                    loss_triplet = torch.clamp(
                        hyp_emb["margin_gamma"] + f_neg - f_pos,
                        min=0.0,
                    )
                    loss_batch += loss_triplet

            loss_batch = loss_batch / max(1, len(batch))
            loss_batch.backward()
            optimizer.step()
            total_loss += loss_batch.item()

        print(f"[Epoch {epoch:02d}] Loss = {total_loss:.4f}")

    return model


# ==========================
# Evaluation metrics
# ==========================

def evaluate_ranking(test_triples, model, entity2id, relation2id, sim_struct, hits_ks):
    """
    Compute MRR and Hits@K for tail prediction (h, r, ?).
    """
    device = next(model.parameters()).device
    id2entity = {idx: e for e, idx in entity2id.items()}
    all_entities = list(entity2id.keys())
    num_entities = len(all_entities)

    ranks = []
    hits = {k: 0 for k in hits_ks}

    print("[INFO] Evaluating ranking metrics (MRR, Hits@K)...")
    for tr in tqdm(test_triples):
        h_name, r_name, t_true = tr["h"], tr["r"], tr["t"]
        w = tr["w"]

        h_idx = torch.tensor([entity2id[h_name]], dtype=torch.long, device=device)
        r_idx = torch.tensor([relation2id[r_name]], dtype=torch.long, device=device)

        scores = []
        with torch.no_grad():
            for e_idx in range(num_entities):
                t_idx = torch.tensor([e_idx], dtype=torch.long, device=device)
                w_tensor = torch.tensor([w], dtype=torch.float32, device=device)
                score = model.forward_score(h_idx, r_idx, t_idx, w_tensor)
                scores.append(score.item())

        scores = np.array(scores)
        # rank true tail
        t_true_idx = entity2id[t_true]
        rank = 1 + np.sum(scores > scores[t_true_idx])  # 1-based rank
        ranks.append(rank)

        for k in hits_ks:
            if rank <= k:
                hits[k] += 1

    mrr = np.mean([1.0 / r for r in ranks])
    hits_rates = {f"Hits@{k}": hits[k] / len(test_triples) for k in hits_ks}
    return mrr, hits_rates


def evaluate_binary_auc_f1(
    test_triples,
    model,
    entity2id,
    relation2id,
    sim_struct,
    threshold=0.0,
):
    """
    Very simple binary evaluation:
    - positives: true test triples
    - negatives: one corrupted tail per positive
    """
    device = next(model.parameters()).device
    all_entities = list(entity2id.keys())
    y_true = []
    y_score = []

    import random

    print("[INFO] Evaluating binary metrics (AUC, Precision, Recall, F1)...")
    for tr in tqdm(test_triples):
        h_name, r_name, t_name = tr["h"], tr["r"], tr["t"]
        w = tr["w"]

        h_idx = torch.tensor([entity2id[h_name]], dtype=torch.long, device=device)
        r_idx = torch.tensor([relation2id[r_name]], dtype=torch.long, device=device)
        t_idx = torch.tensor([entity2id[t_name]], dtype=torch.long, device=device)
        w_tensor = torch.tensor([w], dtype=torch.float32, device=device)

        with torch.no_grad():
            score_pos = model.forward_score(h_idx, r_idx, t_idx, w_tensor).item()

        # sample one negative
        t_neg = random.choice(all_entities)
        while t_neg == t_name:
            t_neg = random.choice(all_entities)
        t_neg_idx = torch.tensor([entity2id[t_neg]], dtype=torch.long, device=device)
        with torch.no_grad():
            score_neg = model.forward_score(h_idx, r_idx, t_neg_idx, w_tensor).item()

        y_true.extend([1, 0])
        y_score.extend([score_pos, score_neg])

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # AUC
    auc = roc_auc_score(y_true, y_score)

    # Threshold on score
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "AUC": float(auc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


# ==========================
# Explainable predictions (Phase 6)
# ==========================

def build_explanations_for_topk(
    test_triples,
    model,
    G,
    entity2id,
    relation2id,
    sim_struct,
    top_k=10,
):
    """
    For each test triple (h,r,t_true), produce top-K predicted tails
    with scores and simple explanations (weight, provenance comment if available).
    """
    device = next(model.parameters()).device
    id2entity = {idx: e for e, idx in entity2id.items()}
    all_entities = list(entity2id.keys())
    num_entities = len(all_entities)

    explanations = []

    print("[INFO] Building explanations for predicted causal links...")
    for tr in tqdm(test_triples):
        h_name, r_name, t_true = tr["h"], tr["r"], tr["t"]
        w_true = tr["w"]

        h_idx = torch.tensor([entity2id[h_name]], dtype=torch.long, device=device)
        r_idx = torch.tensor([relation2id[r_name]], dtype=torch.long, device=device)

        scores = []
        with torch.no_grad():
            for e_idx in range(num_entities):
                t_idx = torch.tensor([e_idx], dtype=torch.long, device=device)
                w_tensor = torch.tensor([w_true], dtype=torch.float32, device=device)
                score = model.forward_score(h_idx, r_idx, t_idx, w_tensor)
                scores.append(score.item())

        scores = np.array(scores)
        sorted_idx = np.argsort(-scores)  # descending
        top_idx = sorted_idx[:top_k]

        preds = []
        for ti in top_idx:
            t_name = id2entity[ti]
            score_val = scores[ti]
            # if edge exists in G, we can fetch weight and explanation
            if G.has_edge(h_name, t_name):
                edge_data = G[h_name][t_name]
                edge_weight = float(edge_data.get("weight", 1.0))
                edge_comment = edge_data.get("explanation", "")
                relation = edge_data.get("relation", r_name)
            else:
                edge_weight = None
                edge_comment = ""
                relation = r_name

            preds.append(
                {
                    "h": h_name,
                    "r": relation,
                    "t": t_name,
                    "score": float(score_val),
                    "edge_weight": edge_weight,
                    "edge_comment": edge_comment,
                }
            )

        explanations.append(
            {
                "query": {"h": h_name, "r": r_name, "t_true": t_true, "w_true": w_true},
                "top_predictions": preds,
            }
        )

    return explanations


# ==========================
# Main entry point
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="CREOG – Causal Reasoning over Ontology-Enriched Graphs with CausE-S"
    )
    parser.add_argument(
        "--owl_path",
        type=str,
        required=True,
        help="Path to Diabetes_large.owl (or similar ontology file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load weighted triples from OWL (Phase 4)
    triples_weighted = load_weighted_triples_from_owl(args.owl_path)

    # Save CKG edges as CSV-like text for inspection
    edges_out = os.path.join(args.output_dir, "ckg_edges.csv")
    with open(edges_out, "w", encoding="utf-8") as f:
        f.write("h,r,t,weight,comment\n")
        for tr in triples_weighted:
            comment_clean = tr["comment"].replace("\n", " ").replace(",", ";")
            f.write(f"{tr['h']},{tr['r']},{tr['t']},{tr['w']},{comment_clean}\n")
    print(f"[INFO] Saved CKG edges to {edges_out}")

    # 2) Build CKG and vocabularies
    G, entity2id, id2entity, relation2id, id2relation = build_ckg_and_vocab(
        triples_weighted
    )

    # 3) Structural similarity
    sim_struct = precompute_structural_similarity(G, entity2id)

    # 4) Train/valid/test split
    train_triples, valid_triples, test_triples = split_triples(triples_weighted)

    # 5) Train CausE-S (main mode)
    hyp_emb = DEFAULT_HYP_EMB
    model_main = train_cause_s(
        train_triples,
        valid_triples,
        entity2id,
        relation2id,
        sim_struct,
        hyp_emb,
        mode="full",
    )

    # 6) Evaluation – ranking metrics
    mrr, hits_dict = evaluate_ranking(
        test_triples,
        model_main,
        entity2id,
        relation2id,
        sim_struct,
        DEFAULT_HYP_EVAL["hits_ks"],
    )

    # 7) Evaluation – binary metrics (AUC, Precision, Recall, F1)
    binary_metrics = evaluate_binary_auc_f1(
        test_triples,
        model_main,
        entity2id,
        relation2id,
        sim_struct,
        threshold=0.0,  # threshold on raw score; can tune
    )

    metrics = {
        "MRR": float(mrr),
        **hits_dict,
        **binary_metrics,
    }

    metrics_out = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics to {metrics_out}")
    print(json.dumps(metrics, indent=2))

    # 8) Explanations for predicted links
    explanations = build_explanations_for_topk(
        test_triples,
        model_main,
        G,
        entity2id,
        relation2id,
        sim_struct,
        top_k=10,
    )

    expl_out = os.path.join(args.output_dir, "predicted_links.json")
    with open(expl_out, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)
    print(f"[INFO] Saved predicted links + explanations to {expl_out}")

    # 9) Save embeddings
    ent_emb = model_main.entity_emb.weight.detach().cpu().numpy()
    rel_emb = model_main.rel_emb.weight.detach().cpu().numpy()
    np.save(os.path.join(args.output_dir, "entity_embeddings.npy"), ent_emb)
    np.save(os.path.join(args.output_dir, "relation_embeddings.npy"), rel_emb)

    # Also save mapping files for reproducibility
    with open(os.path.join(args.output_dir, "entity2id.json"), "w", encoding="utf-8") as f:
        json.dump(entity2id, f, indent=2)
    with open(os.path.join(args.output_dir, "relation2id.json"), "w", encoding="utf-8") as f:
        json.dump(relation2id, f, indent=2)

    print("[INFO] Finished CREOG pipeline (ontology-only variant).")


if __name__ == "__main__":
    main()
