# hist

Unified digital pathology pipeline with a single CLI entry point.

## Installation

```bash
pip install -e .
```

## Usage

### 1) Tiling (train split)

```bash
hist tile \
  --slides data/wss1_v2/out/train/slides \
  --annos  data/wss1_v2/anno \
  --out    data/wss1_v2/out/train \
  --patch-size 224 --stride 224 \
  --overwrite
```

### 2) Feature extraction (ViT-B/16 @224)

```bash
hist extract \
  --csv-dir data/wss1_v2/out/train/patch_csvs \
  --output  saves/features_vit224 \
  --encoder vit_b16 --device cuda \
  --batch-size 256 --num-workers 8
```

### 3) Train Hierarchical MIL with 5-fold CV

```bash
hist train \
  --features saves/features_vit224 \
  --labels-csv data/wss1_v2/out/train/labels.csv \
  --output   saves/experiments/hiermil_vit224 \
  --model hiermil --epochs 20 --k-folds 5 \
  --device cuda --oversample
```

### 4) Compare models (ABMIL, TransMIL, HierMIL, GCN/GAT)

```bash
hist compare \
  saves/experiments/abmil_vit224 \
  saves/experiments/transmil_vit224 \
  saves/experiments/hiermil_vit224 \
  --out saves/comparison_vit224 \
  --metrics balanced_accuracy f1_macro roc_auc mcc
```

### 5) Explainability (attention overlays and t-SNE)

```bash
hist explain \
  --runs   saves/experiments/hiermil_vit224 \
  --out    saves/experiments/hiermil_vit224/explain
```

## Testing

```bash
pytest
```
