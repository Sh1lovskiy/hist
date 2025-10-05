# train_mil_cv.py (drop-in replacement with CLI and extras)
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import numpy as np
from loguru import logger
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold

from data_wrappers import PatchDataset, group_bags_by_slide_level
from encoders import Encoder, EncCfg
from models_mil import ABMIL, HierMIL
from models_gnn import GCNHead, GATHead
from metrics_ext import compute_cls_metrics
from plotting import save_curve
from stain import macenko_normalize

DATA_TRAIN = Path("data/wss1_v2/out/train")
DATA_TEST = Path("data/wss1_v2/out/test")
IMG_OUT = Path("data/images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLS = 5  # AT,BG,LP,MM,DYS


def _build_transforms(img_size: int, do_stain: bool):
    from torchvision import transforms

    def _maybe_stain(img):
        if not do_stain:
            return img
        arr = np.array(img)
        arr = macenko_normalize(arr)
        return transforms.functional.to_pil_image(arr)

    tfm_tr = transforms.Compose(
        [
            transforms.Lambda(_maybe_stain),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tfm_ev = transforms.Compose(
        [
            transforms.Lambda(_maybe_stain),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm_tr, tfm_ev


def _extract(csvs: List[Path], enc: Encoder, tfm, desc: str):
    ds = PatchDataset(csvs, DATA_TRAIN, tfm)
    dl = DataLoader(ds, batch_size=64, num_workers=4)
    E, Y, S, L, XY = [], [], [], [], []
    with torch.no_grad():
        for x, y, sid, lvl, xy in tqdm(dl, desc=desc, unit="batch"):
            x = x.to(DEVICE, non_blocking=True)
            e = enc(x).cpu()
            E.append(e)
            Y.append(y)
            S.extend(list(sid))
            L.extend(list(lvl))
            XY.extend([tuple(map(int, a)) for a in xy])
    E = torch.cat(E, dim=0)
    Y = torch.cat(Y, dim=0).long().numpy()
    return E, Y, S, L, XY


def _bags_targets(E, Y, S, L, XY):
    bags, targets, locs = group_bags_by_slide_level(E, torch.tensor(Y), S, L, XY)
    slide_ids = sorted(bags.keys())
    X = [bags[sid] for sid in slide_ids]
    y = np.array([targets[sid] for sid in slide_ids], dtype=int)
    C = [locs[sid] for sid in slide_ids]
    return slide_ids, X, y, C


def choose_splitter(y: np.ndarray, desired: int = 5):
    cls, cnt = np.unique(y, return_counts=True)
    min_count = int(cnt.min())
    n_splits = int(min(desired, max(1, min_count)))
    if n_splits >= 2:
        logger.info(
            f"Using StratifiedKFold n_splits={n_splits} " f"(min class={min_count})."
        )
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17)
    logger.warning("Too few slides per class; fallback KFold(n_splits=2).")
    return KFold(n_splits=2, shuffle=True, random_state=17)


def fit_fold_label_map(y_tr: np.ndarray):
    present = np.unique(y_tr).tolist()
    to_local = {g: i for i, g in enumerate(present)}
    return to_local, {i: g for g, i in to_local.items()}, len(present)


def oversample_bags(X, y):
    uniq, cnt = np.unique(y, return_counts=True)
    m = cnt.max()
    Xb, yb = [], []
    for u, c in zip(uniq, cnt):
        idx = np.where(y == u)[0].tolist()
        rep = int(np.ceil(m / c))
        take = (idx * rep)[:m]
        Xb += [X[i] for i in take]
        yb += [y[i] for i in take]
    return Xb, np.array(yb, dtype=int)


def _train_abmil(X_tr, y_tr, d: int, n_cls: int, epochs=12):
    model = ABMIL(d=d, n_cls=n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for bag, y in zip(X_tr, y_tr):
            H_all = torch.cat(list(bag.values()), dim=0).to(DEVICE)
            t = torch.tensor([y], device=DEVICE)
            opt.zero_grad()
            logits, _ = model(H_all)
            loss = F.cross_entropy(logits, t)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
        losses.append(ep_loss / max(1, len(X_tr)))
    return model, losses


def _train_hiermil(X_tr, y_tr, d: int, n_cls: int, epochs=12):
    model = HierMIL(d=d, n_cls=n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for bag, y in zip(X_tr, y_tr):
            bag_dev = {k: v.to(DEVICE) for k, v in bag.items()}
            t = torch.tensor([y], device=DEVICE)
            opt.zero_grad()
            logits = model(bag_dev)
            loss = F.cross_entropy(logits, t)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
        losses.append(ep_loss / max(1, len(X_tr)))
    return model, losses


@torch.no_grad()
def _predict_abmil(model, X_te):
    y_pred, y_prob = [], []
    model.eval()
    for bag in X_te:
        H_all = torch.cat(list(bag.values()), dim=0).to(DEVICE)
        logits, att = model(H_all)
        p = torch.softmax(logits, dim=1).cpu().numpy()[0]
        y_prob.append(p)
        y_pred.append(int(p.argmax()))
    return np.array(y_pred), np.array(y_prob)


@torch.no_grad()
def _predict_hier(model, X_te):
    y_pred, y_prob = [], []
    model.eval()
    for bag in X_te:
        bag_dev = {k: v.to(DEVICE) for k, v in bag.items()}
        logits = model(bag_dev)
        p = torch.softmax(logits, dim=1).cpu().numpy()[0]
        y_prob.append(p)
        y_pred.append(int(p.argmax()))
    return np.array(y_pred), np.array(y_prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoder",
        default="resnet18",
        choices=["resnet18", "vit_b16", "convnext_t", "clip_vit_b32", "hipt"],
    )
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--stain", action="store_true", help="Apply Macenko normalization.")
    ap.add_argument(
        "--model", default="hiermil", choices=["abmil", "hiermil", "gcn", "gat"]
    )
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    logger.remove()
    logger.add(
        sink=lambda m: print(m, end=""),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        diagnose=False,
    )

    tr_csvs = list(DATA_TRAIN.glob("*.csv"))
    if not tr_csvs:
        logger.error("No train CSVs found.")
        return

    enc = Encoder(EncCfg(name=args.encoder, img_size=args.img_size)).to(DEVICE)
    tfm_tr, tfm_ev = _build_transforms(args.img_size, do_stain=args.stain)

    E_tr, Y_tr, S_tr, L_tr, XY_tr = _extract(tr_csvs, enc, tfm_tr, "Embed-train")
    slide_ids, X_all, y_all, C_all = _bags_targets(E_tr, Y_tr, S_tr, L_tr, XY_tr)

    splitter = choose_splitter(y_all, desired=args.cv)
    scores = []

    for fold, (idx_tr, idx_te) in enumerate(splitter.split(X_all, y_all), 1):
        X_tr = [X_all[i] for i in idx_tr]
        y_tr_g = y_all[idx_tr]
        X_te = [X_all[i] for i in idx_te]
        y_te_g = y_all[idx_te]

        to_local, to_global, n_local = fit_fold_label_map(y_tr_g)
        y_tr = np.array([to_local[g] for g in y_tr_g], dtype=int)
        X_tr, y_tr = oversample_bags(X_tr, y_tr)

        d = next(iter(X_tr[0].values())).shape[1]

        if args.model == "abmil":
            model, losses = _train_abmil(
                X_tr, y_tr, d=d, n_cls=n_local, epochs=args.epochs
            )
            save_curve(
                losses,
                f"ABMIL Loss Fold {fold}",
                IMG_OUT / f"abmil_loss_fold{fold}.png",
            )
            y_hat, y_prob = _predict_abmil(model, X_te)

        elif args.model == "hiermil":
            model, losses = _train_hiermil(
                X_tr, y_tr, d=d, n_cls=n_local, epochs=args.epochs
            )
            save_curve(
                losses,
                f"HierMIL Loss Fold {fold}",
                IMG_OUT / f"hiermil_loss_fold{fold}.png",
            )
            y_hat, y_prob = _predict_hier(model, X_te)

        else:
            # GNN: flatten all levels per slide into a single graph
            from graphs import build_graph, GraphCfg
            import numpy as np

            def _to_graph(bag, coords):
                H, XY = [], []
                for L, T in bag.items():
                    H.append(T.numpy())
                for L, Lxy in coords.items():
                    XY += Lxy
                H = np.concatenate(H, axis=0)
                XY = np.array(XY, dtype=float)
                return build_graph(H, XY, GraphCfg(kind="knn", k=8))

            # Train simple head with cross-entropy
            if args.model == "gcn":
                head = GCNHead(d=d, n_cls=n_local).to(DEVICE)
            else:
                head = GATHead(d=d, n_cls=n_local).to(DEVICE)
            opt = torch.optim.Adam(head.parameters(), lr=1e-4)
            losses = []
            coords_tr = [C_all[i] for i in idx_tr]
            for ep in range(args.epochs):
                ep_loss = 0.0
                head.train()
                for bag, y, c in zip(X_tr, y_tr, coords_tr):
                    g = _to_graph(bag, c)
                    if isinstance(g, dict):
                        raise RuntimeError("Install torch_geometric for graph heads.")
                    g = g.to(DEVICE)
                    t = torch.tensor([y], device=DEVICE)
                    opt.zero_grad()
                    logits = head(g)
                    loss = F.cross_entropy(logits, t)
                    loss.backward()
                    opt.step()
                    ep_loss += float(loss.item())
                losses.append(ep_loss / max(1, len(X_tr)))
            save_curve(
                losses,
                f"{args.model.upper()} Loss Fold {fold}",
                IMG_OUT / f"{args.model}_loss_fold{fold}.png",
            )

            # Predict
            y_hat, y_prob = [], []
            head.eval()
            coords_te = [C_all[i] for i in idx_te]
            for bag, c in zip(X_te, coords_te):
                g = _to_graph(bag, c)
                g = g.to(DEVICE)
                with torch.no_grad():
                    p = torch.softmax(head(g), dim=1).cpu().numpy()[0]
                y_prob.append(p)
                y_hat.append(int(p.argmax()))
            y_hat = np.array(y_hat)
            y_prob = np.array(y_prob)

        # Metrics in local space to avoid absent-class issues
        y_te_loc = np.array([to_local.get(g, None) for g in y_te_g], dtype=object)
        mask = np.array([v is not None for v in y_te_loc], dtype=bool)
        y_te_loc = y_te_loc[mask].astype(int)
        y_hat = y_hat[mask]
        y_prob = y_prob[mask]

        m = compute_cls_metrics(y_te_loc, y_hat, y_prob, labels_all=None)
        scores.append(m)
        logger.info(f"Fold {fold} {args.model}: {m}")

    def _avg(scores: List[Dict[str, float]]) -> Dict[str, float]:
        keys = sorted({k for s in scores for k in s})
        return {k: float(np.mean([s.get(k, np.nan) for s in scores])) for k in keys}

    logger.info(f"{args.model} {args.cv}-fold avg: {_avg(scores)}")


if __name__ == "__main__":
    main()
