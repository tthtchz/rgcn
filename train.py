import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from model import RGCNLinkPredictor


def split_edges(label_edge_index, val_frac=0.1, test_frac=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = label_edge_index.shape[1]
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    return label_edge_index[:, train_idx], label_edge_index[:, val_idx], label_edge_index[:, test_idx]


def pair_codes(edges, num_nodes):
    """Encode each (drug, disease) column of a [2, N] edge tensor as one int64,
    so membership in a set of known pairs can be checked with torch.isin."""
    return edges[0] * num_nodes + edges[1]


def sample_negatives(pos_edge_index, candidate_diseases, num_neg, num_nodes,
                     exclude_codes=None, generator=None, max_tries=100):
    """Corrupt the disease end of each positive (drug, disease) pair.

    candidate_diseases: pool the negative disease is drawn from. Drawing from
        diseases that do have indications (rather than all ~6k diseases, most of
        which never appear in any indication) removes the shortcut of scoring a
        pair low just because the disease is never a positive.
    exclude_codes: pair_codes of known positives; any negative that collides
        with one is redrawn, so a true indication is never labelled 0.
    generator: pass one generator for the whole run so each call (each epoch)
        draws fresh negatives. A new torch.Generator() always starts from the
        same default seed and would return identical negatives every time.
    """
    n_pos = pos_edge_index.shape[1]
    reps = -(-num_neg // n_pos)  # ceil division so we always have enough
    src = pos_edge_index[0].repeat(reps)[:num_neg]
    dst = candidate_diseases[torch.randint(0, len(candidate_diseases), (num_neg,), generator=generator)]
    if exclude_codes is not None:
        for _ in range(max_tries):
            bad = torch.isin(pair_codes(torch.stack([src, dst]), num_nodes), exclude_codes)
            if not bad.any():
                break
            dst[bad] = candidate_diseases[torch.randint(0, len(candidate_diseases), (int(bad.sum()),), generator=generator)]
    return torch.stack([src, dst])


@torch.no_grad()
def evaluate(model, edge_index, edge_type, pos_edges, candidate_diseases, exclude_codes, num_nodes,
             device, num_neg_per_pos=1, seed=42):
    model.eval()
    z = model.encode(edge_index.to(device), edge_type.to(device))

    g = torch.Generator().manual_seed(seed)
    neg_edges = sample_negatives(pos_edges, candidate_diseases, pos_edges.shape[1] * num_neg_per_pos,
                                 num_nodes, exclude_codes=exclude_codes, generator=g)

    pos_scores = model.decode(z, pos_edges[0].to(device), pos_edges[1].to(device)).sigmoid().cpu().numpy()
    neg_scores = model.decode(z, neg_edges[0].to(device), neg_edges[1].to(device)).sigmoid().cpu().numpy()

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return auc, ap


@torch.no_grad()
def ranking_eval(model, edge_index, edge_type, pos_edges, disease_ids, all_pos_pairs, device,
                  max_eval=2000, seed=0):
    model.eval()
    z = model.encode(edge_index.to(device), edge_type.to(device))
    disease_ids = disease_ids.to(device)
    disease_z = z[disease_ids]  # [D, dim]
    rel = model.decoder.rel_emb.weight[0]  # single target relation

    n = pos_edges.shape[1]
    if n > max_eval:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_eval]
        pos_edges = pos_edges[:, idx]
        n = max_eval

    drug_to_diseases = {}
    for s, d in all_pos_pairs.t().tolist():
        drug_to_diseases.setdefault(s, set()).add(d)

    disease_id_list = disease_ids.tolist()
    disease_pos_in_list = {d: i for i, d in enumerate(disease_id_list)}

    ranks = []
    for i in range(n):
        src = pos_edges[0, i].item()
        true_dst = pos_edges[1, i].item()
        src_z = z[src] * rel  # [dim]
        scores = (disease_z * src_z).sum(dim=-1)  # [D]

        known = drug_to_diseases.get(src, set())
        mask = torch.zeros(len(disease_id_list), dtype=torch.bool, device=device)
        for k in known:
            j = disease_pos_in_list.get(k)
            if j is not None:
                mask[j] = True
        true_pos = disease_pos_in_list[true_dst]
        true_score = scores[true_pos].clone()
        scores = scores.masked_fill(mask, float("-inf"))
        scores[true_pos] = true_score

        rank = (scores > true_score).sum().item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    hits1 = float(np.mean(ranks <= 1))
    hits3 = float(np.mean(ranks <= 3))
    hits10 = float(np.mean(ranks <= 10))
    return {"MRR": mrr, "Hits@1": hits1, "Hits@3": hits3, "Hits@10": hits10}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="data/processed/graph.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    # basis decomposition only saves parameters when num_bases < num_relations
    # (3 here); default None = one full weight matrix per relation.
    parser.add_argument("--num-bases", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--neg-ratio", type=int, default=1)
    # "indicated": training negatives drawn from diseases with >=1 training
    # indication; "all": from every disease (the original, easier setting).
    # Evaluation negatives always use diseases with >=1 indication.
    parser.add_argument("--neg-pool", choices=["indicated", "all"], default="indicated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", default="checkpoints/rgcn_indication.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data = torch.load(args.graph, weights_only=False)
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    label_edge_index = data["label_edge_index"]
    disease_ids = data["disease_ids"]
    num_nodes = data["num_nodes"]
    num_relations = data["num_relations"]

    train_edges, val_edges, test_edges = split_edges(label_edge_index, seed=args.seed)
    print(f"Indication edges -> train {train_edges.shape[1]}, "
          f"val {val_edges.shape[1]}, test {test_edges.shape[1]}")

    train_neg_pool = train_edges[1].unique() if args.neg_pool == "indicated" else disease_ids
    eval_neg_pool = label_edge_index[1].unique()
    train_exclude = pair_codes(train_edges, num_nodes)
    all_exclude = pair_codes(label_edge_index, num_nodes)
    print(f"Negative disease pool -> train {len(train_neg_pool)} ({args.neg_pool}), eval {len(eval_neg_pool)}")
    neg_gen = torch.Generator().manual_seed(args.seed)

    model = RGCNLinkPredictor(num_nodes, num_relations, hidden_dim=args.hidden_dim,
                               out_dim=args.out_dim, num_bases=args.num_bases,
                               num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    edge_index_dev = edge_index.to(device)
    edge_type_dev = edge_type.to(device)

    best_val_auc = 0.0
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(edge_index_dev, edge_type_dev)

        neg_edges = sample_negatives(train_edges, train_neg_pool, train_edges.shape[1] * args.neg_ratio,
                                     num_nodes, exclude_codes=train_exclude, generator=neg_gen)
        pos_score = model.decode(z, train_edges[0].to(device), train_edges[1].to(device))
        neg_score = model.decode(z, neg_edges[0].to(device), neg_edges[1].to(device))

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            val_auc, val_ap = evaluate(model, edge_index, edge_type, val_edges, eval_neg_pool,
                                     all_exclude, num_nodes, device)
            print(f"Epoch {epoch:03d} | loss {loss.item():.4f} | val AUC {val_auc:.4f} | val AP {val_ap:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), args.ckpt)

    print(f"Loading best checkpoint (val AUC {best_val_auc:.4f}) for final test evaluation.")
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    test_auc, test_ap = evaluate(model, edge_index, edge_type, test_edges, eval_neg_pool,
                                 all_exclude, num_nodes, device, seed=123)
    print(f"Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")

    all_pos = torch.cat([train_edges, val_edges, test_edges], dim=1)
    ranking = ranking_eval(model, edge_index, edge_type, test_edges, disease_ids, all_pos, device)
    print("Filtered ranking metrics on test indication edges:")
    for k, v in ranking.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
