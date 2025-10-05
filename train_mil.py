# train_mil.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from PIL import Image
from loguru import logger
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


DATA_DIR = Path("data/wss1_v2/out/train")
IMG_SIZE = 224
BATCH = 64
LR = 1e-4
EPOCHS = 5
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAME_TO_IDX: Dict[str, int] = {"AT": 0, "BG": 1, "LP": 2, "MM": 3, "DYS": 4}
IDX_TO_CLASS = {v: k for k, v in CLASS_NAME_TO_IDX.items()}


class PatchDataset(Dataset):
    """
    Flat patch dataset backed by exported CSVs.
    Returns (tensor, label_idx, slide_id).
    """

    def __init__(self, csv_paths: List[Path], root: Path, tfm):
        self.root = root
        self.transform = tfm
        df = [pd.read_csv(p) for p in csv_paths]
        self.df = pd.concat(df, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        path = self.root / r["rel_path"]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = int(r["label_idx"])
        slide_id = r["rel_path"].split("/")[0]
        return x, y, slide_id


class FeatExtractor(nn.Module):
    """Frozen ResNet-18 feature extractor (512-D)."""

    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.backbone = m
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, 512]


class ABMIL(nn.Module):
    """
    Attention MIL head: attention pooling + linear classifier.
    """

    def __init__(self, d: int = 512, n_cls: int = 5):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.cls = nn.Linear(d, n_cls)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # H: [N, d] — all instances of one slide (bag)
        a = self.att(H).squeeze(-1)  # [N]
        w = torch.softmax(a, dim=0).unsqueeze(-1)
        z = torch.sum(w * H, dim=0, keepdim=True)  # [1, d]
        logits = self.cls(z)  # [1, n_cls]
        return logits, w.squeeze(-1)  # (logits, weights)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@torch.no_grad()
def extract_embeddings(
    ds: PatchDataset, feat: FeatExtractor
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    dl = DataLoader(ds, batch_size=BATCH, num_workers=NUM_WORKERS)
    feat.eval().to(DEVICE)
    embs, labs, sids = [], [], []
    for x, y, sid in tqdm(dl, desc="Embed", unit="batch"):
        x = x.to(DEVICE, non_blocking=True)
        e = feat(x).cpu()
        embs.append(e)
        labs.append(y.clone())  # y is CPU LongTensor already
        sids.extend(list(sid))
    E = torch.cat(embs, dim=0)  # [N, 512]
    L = torch.cat(labs, dim=0).long()  # [N]
    return E, L, sids


def majority_slide_label(lbls: torch.Tensor) -> int:
    """
    Majority vote over non-BG; fallback to BG if none.
    """
    if lbls.numel() == 0:
        return CLASS_NAME_TO_IDX["BG"]
    non_bg = lbls[lbls != CLASS_NAME_TO_IDX["BG"]]
    target = non_bg if non_bg.numel() > 0 else lbls
    vals, cnts = target.unique(return_counts=True)
    return int(vals[cnts.argmax()].item())


def make_bags(
    embs: torch.Tensor, labels: torch.Tensor, slides: List[str]
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    Group instance embeddings by slide and assign slide-level labels.
    """
    bags, bag_y, bag_ids = [], [], []
    slide_ids = sorted(set(slides))
    for sid in slide_ids:
        idx = [i for i, s in enumerate(slides) if s == sid]
        H = embs[idx]  # [Ni, 512]
        y = majority_slide_label(labels[idx])
        bags.append(H)
        bag_y.append(y)
        bag_ids.append(sid)
    return bags, bag_y, bag_ids


def train_loop(
    bags: List[torch.Tensor], bag_y: List[int], d: int = 512, n_cls: int = 5
) -> ABMIL:
    model = ABMIL(d=d, n_cls=n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for H, y in zip(bags, bag_y):
            H = H.to(DEVICE, non_blocking=True)
            y_t = torch.tensor([y], device=DEVICE)
            opt.zero_grad()
            logits, _ = model(H)
            loss = ce(logits, y_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        logger.info(
            f"Epoch {ep+1}/{EPOCHS} | " f"loss={sum(losses)/max(1,len(losses)):.4f}"
        )
    return model


@torch.no_grad()
def evaluate(
    model: ABMIL,
    bags: List[torch.Tensor],
    bag_y: List[int],
) -> float:
    model.eval()
    correct, total = 0, 0
    for H, y in zip(bags, bag_y):
        H = H.to(DEVICE, non_blocking=True)
        logits, _ = model(H)
        pred = int(logits.argmax(dim=1).item())
        correct += int(pred == y)
        total += 1
    return correct / max(1, total)


def main() -> None:
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | {message}",
        diagnose=False,
    )

    csvs = sorted(list(DATA_DIR.glob("*.csv")))
    if not csvs:
        logger.error("No CSVs found. Run runner.py first.")
        return

    tfm = build_transform()
    ds = PatchDataset(csvs, DATA_DIR, tfm)

    feat = FeatExtractor()
    embs, patch_labels, slide_ids = extract_embeddings(ds, feat)

    bags, bag_y, bag_ids = make_bags(embs, patch_labels, slide_ids)
    n_cls = len(CLASS_NAME_TO_IDX)

    model = train_loop(bags, bag_y, d=feat.out_dim, n_cls=n_cls)
    acc = evaluate(model, bags, bag_y)
    logger.info(f"Train set slide accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
