from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

from features import build_train_transform, build_eval_transform, ResNet18Feats
from metrics_ext import iou_dice_from_patch_labels
from plotting import save_curve


DATA = Path("data/wss1_v2/out/train")
IMG_SIZE = 224
BATCH = 128
WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLS = 5


class PatchDS(Dataset):
    def __init__(self, csvs: List[Path], root: Path, tfm):
        self.root = root
        self.tfm = tfm
        df = [pd.read_csv(p) for p in csvs]
        self.df = pd.concat(df, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        x = self.tfm((self.root / r["rel_path"]).open("rb"))
        return x, int(r["label_idx"])


def _to_tensor(img_fp):
    from PIL import Image

    img = Image.open(img_fp).convert("RGB")
    return img


def main():
    logger.remove()
    logger.add(
        sink=lambda m: print(m, end=""),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        diagnose=False,
    )

    csvs = list(DATA.glob("*.csv"))
    if not csvs:
        logger.error("No train CSVs found.")
        return

    # Build dataset once, reuse indices per split
    tfm_tr = build_train_transform(IMG_SIZE)
    tfm_ev = build_eval_transform(IMG_SIZE)

    # Custom loader to keep transforms in Dataset
    class _DS(Dataset):
        def __init__(self, df: pd.DataFrame, root: Path, tfm):
            self.df = df.reset_index(drop=True)
            self.root = root
            self.tfm = tfm

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            r = self.df.iloc[i]
            from PIL import Image

            img = Image.open(self.root / r["rel_path"]).convert("RGB")
            x = self.tfm(img)
            return x, int(r["label_idx"])

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    y_all = df["label_idx"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

    f1m, f1M, bacc, iou_list, dice_list = [], [], [], [], []
    fold = 0
    for tr, te in skf.split(np.zeros_like(y_all), y_all):
        fold += 1
        ds_tr = _DS(df.iloc[tr], DATA, tfm_tr)
        ds_te = _DS(df.iloc[te], DATA, tfm_ev)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=WORKERS)
        dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=WORKERS)

        # Linear head on frozen ResNet18 features
        feat = ResNet18Feats().to(DEVICE).eval()
        head = nn.Linear(feat.out_dim, N_CLS).to(DEVICE)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()

        losses = []
        for ep in range(8):
            head.train()
            ep_loss = 0.0
            for x, y in dl_tr:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                with torch.no_grad():
                    z = feat(x)
                logits = head(z)
                loss = ce(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
            losses.append(ep_loss / max(1, len(dl_tr)))
        save_curve(
            losses,
            f"PatchHead Loss Fold {fold}",
            Path("data/images") / f"patch_loss_fold{fold}.png",
        )

        # Eval
        ys, ps = [], []
        head.eval()
        for x, y in dl_te:
            x = x.to(DEVICE)
            with torch.no_grad():
                z = feat(x)
                logits = head(z)
                pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)

        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        bacc_fold = balanced_accuracy_score(y_true, y_pred)
        iou, dice = iou_dice_from_patch_labels(y_true, y_pred, N_CLS)

        f1M.append(f1_macro)
        f1m.append(f1_micro)
        bacc.append(bacc_fold)
        iou_list.append(iou["iou_mean"])
        dice_list.append(dice["dice_mean"])

        logger.info(
            f"Fold {fold} PatchClf: "
            f"F1macro={f1_macro:.3f} F1micro={f1_micro:.3f} "
            f"BalAcc={bacc_fold:.3f} IoU={iou['iou_mean']:.3f} "
            f"Dice={dice['dice_mean']:.3f}"
        )

    logger.info(
        f"PatchClf 5-fold avg: "
        f"F1macro={np.mean(f1M):.3f} F1micro={np.mean(f1m):.3f} "
        f"BalAcc={np.mean(bacc):.3f} IoU={np.mean(iou_list):.3f} "
        f"Dice={np.mean(dice_list):.3f}"
    )


if __name__ == "__main__":
    main()
