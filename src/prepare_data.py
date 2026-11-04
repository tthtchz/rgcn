import argparse
import os

import numpy as np
import pandas as pd
import torch

MP_RELATIONS = ["drug_protein", "protein_protein", "disease_protein"]
TARGET_RELATION = "indication"
KEEP_RELATIONS = MP_RELATIONS + [TARGET_RELATION]

COLS = ["relation", "x_index", "x_type", "y_index", "y_type", "x_id", "y_id"]
DTYPES = {
    "relation": "category",
    "x_index": "int64",
    "x_type": "category",
    "y_index": "int64",
    "y_type": "category",
    "x_id": "string",
    "y_id": "string",
}


def load_filtered(kg_csv_path):
    print(f"Reading {kg_csv_path} ...")
    df = pd.read_csv(kg_csv_path, usecols=COLS, dtype=DTYPES)
    df = df[df["relation"].isin(KEEP_RELATIONS)].reset_index(drop=True)
    print(f"Kept {len(df)} rows across relations: "
          f"{df['relation'].value_counts().to_dict()}")
    return df


def build_graph(df, out_dir):
    mp_df = df[df["relation"].isin(MP_RELATIONS)]
    tgt_df = df[df["relation"] == TARGET_RELATION]

    node_keys = pd.concat([
        mp_df[["x_index", "x_type"]].rename(columns={"x_index": "index", "x_type": "type"}),
        mp_df[["y_index", "y_type"]].rename(columns={"y_index": "index", "y_type": "type"}),
        tgt_df[["x_index", "x_type"]].rename(columns={"x_index": "index", "x_type": "type"}),
        tgt_df[["y_index", "y_type"]].rename(columns={"y_index": "index", "y_type": "type"}),
    ]).drop_duplicates().reset_index(drop=True)

    node_keys["new_id"] = np.arange(len(node_keys))
    key_to_id = {(row.index, row.type): row.new_id for row in node_keys.itertuples()}
    num_nodes = len(node_keys)
    print(f"Total nodes in filtered graph: {num_nodes}")
    for t, c in node_keys["type"].value_counts().items():
        print(f"  {t}: {c}")

    def remap(sub_df):
        src = [key_to_id[(i, t)] for i, t in zip(sub_df["x_index"], sub_df["x_type"])]
        dst = [key_to_id[(i, t)] for i, t in zip(sub_df["y_index"], sub_df["y_type"])]
        return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)

    rel_to_id = {r: i for i, r in enumerate(MP_RELATIONS)}

    src_all, dst_all, rel_all = [], [], []
    for rel in MP_RELATIONS:
        sub = mp_df[mp_df["relation"] == rel]
        src, dst = remap(sub)
        src_all.append(src); dst_all.append(dst); rel_all.append(np.full(len(src), rel_to_id[rel]))

    edge_index = torch.tensor(np.stack([np.concatenate(src_all), np.concatenate(dst_all)]), dtype=torch.long)
    edge_type = torch.tensor(np.concatenate(rel_all), dtype=torch.long)
    num_relations = len(MP_RELATIONS)

    is_drug_x = (tgt_df["x_type"] == "drug").values
    drug_index = np.where(is_drug_x, tgt_df["x_index"].values, tgt_df["y_index"].values)
    disease_index = np.where(is_drug_x, tgt_df["y_index"].values, tgt_df["x_index"].values)
    tgt_src = np.array([key_to_id[(i, "drug")] for i in drug_index], dtype=np.int64)
    tgt_dst = np.array([key_to_id[(i, "disease")] for i in disease_index], dtype=np.int64)
    label_edges = np.stack([tgt_src, tgt_dst], axis=1)
    label_edges = np.unique(label_edges, axis=0)
    label_edge_index = torch.tensor(label_edges.T, dtype=torch.long)
    print(f"Message-passing edges (both directions, as stored in PrimeKG): {edge_index.shape[1]}")
    print(f"Unique indication (drug-disease) edges: {label_edge_index.shape[1]}")

    drug_ids = torch.tensor(node_keys.loc[node_keys["type"] == "drug", "new_id"].values, dtype=torch.long)
    disease_ids = torch.tensor(node_keys.loc[node_keys["type"] == "disease", "new_id"].values, dtype=torch.long)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "num_nodes": num_nodes,
        "num_relations": num_relations,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "label_edge_index": label_edge_index,
        "drug_ids": drug_ids,
        "disease_ids": disease_ids,
        "node_type": node_keys["type"].astype(str).values,
        "relations": MP_RELATIONS,
    }, os.path.join(out_dir, "graph.pt"))
    node_keys.to_csv(os.path.join(out_dir, "node_index.csv"), index=False)
    print(f"Saved graph to {os.path.join(out_dir, 'graph.pt')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-csv", default="data/raw/kg.csv")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    df = load_filtered(args.kg_csv)
    build_graph(df, args.out_dir)


if __name__ == "__main__":
    main()
