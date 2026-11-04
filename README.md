# RGCN Drug-Indication Prediction on PrimeKG

Predicts drug &rarr; disease `indication` relations using an [RGCN](https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.conv.RGCNConv.html)
trained on a subgraph of [PrimeKG](https://github.com/mims-harvard/PrimeKG) (Chandak, Huang & Zitnik,
*Nat. Med.* 2024, https://www.nature.com/articles/s41591-024-03233-x) restricted to:

- `drug_protein` (drug-gene)
- `protein_protein` (gene-gene)
- `disease_protein` (gene-disease)

The `indication` edges (drug-disease) are held out entirely from message passing and used purely
as the link-prediction target (train/val/test split).

## Setup

```bash
pip install -r requirements.txt
```

## 1. Download PrimeKG

```bash
mkdir -p data/raw
curl -L -o data/raw/kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620
```

## 2. Build the graph

```bash
cd src
python prepare_data.py --kg-csv ../data/raw/kg.csv --out-dir ../data/processed
```

This filters `kg.csv` down to the four relations above, remaps node ids to a contiguous space,
keeps one relation id per message-passing relation (3 in total; PrimeKG already stores
each of these edges in both directions, so no inverse edges are added), and saves everything to
`data/processed/graph.pt` (+ `node_index.csv` for interpretability).

## 3. Train

```bash
cd src
python train.py --epochs 100
```

Model: a 2-layer `RGCNConv` encoder (one weight matrix per relation; no basis decomposition, since there are only 3 relations) producing node embeddings,
decoded with a DistMult bilinear score for candidate (drug, indication, disease) triples.
Trained with binary cross-entropy against negatives made by replacing the disease of each
positive pair. Negatives are redrawn every epoch, drawn only from diseases that have at least one
training indication (most of PrimeKG's ~6k diseases have none, which would let the model score
pairs low just because the disease is never positive), and never collide with a known indication.
`--neg-pool all` restores the original all-disease pool for comparison.

Reports validation AUROC/AP during training, then on the best checkpoint: test AUROC/AP plus
filtered ranking metrics (MRR, Hits@1/3/10) over all diseases.

## Files

- `src/prepare_data.py` — PrimeKG CSV &rarr; PyG-ready tensors
- `src/model.py` — RGCN encoder + DistMult decoder
- `src/train.py` — training loop, negative sampling, evaluation
