# alex_part_pipeline.py
from __future__ import annotations
import os, json, collections
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# torchvision (AlexNet)
try:
    from torchvision.models import alexnet, AlexNet_Weights
except Exception:
    alexnet, AlexNet_Weights = None, None


# -------------------- utils --------------------
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    if logits is None or logits.numel() == 0: return 0.0
    return (logits.argmax(1) == y).float().mean().item()

def _class_weights(counts: List[int]) -> torch.Tensor:
    tot = float(sum(counts)) + 1e-8
    inv = [tot / (c if c > 0 else 1.0) for c in counts]
    w = torch.tensor(inv, dtype=torch.float32)
    return w / w.mean()

def _weighted_sampler(labels: List[int], K: int) -> WeightedRandomSampler:
    counts = [0]*K
    for y in labels: counts[int(y)] += 1
    inv = [1.0 / (c if c > 0 else 1.0) for c in counts]
    weights = [inv[int(y)] for y in labels]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(labels), replacement=True)


# -------------------- models --------------------
class AlexBackbone(nn.Module):
    """TorchVision AlexNet.features + AdaptiveAvgPool2d((6,6)) -> 9216-d vec."""
    def __init__(self, trainable: bool = False, weights: Optional[str] = None):
        super().__init__()
        if alexnet is None:
            raise RuntimeError("torchvision.models.alexnet() unavailable.")
        tv_w = None
        if weights == "imagenet" and AlexNet_Weights is not None:
            try: tv_w = AlexNet_Weights.IMAGENET1K_V1
            except Exception: tv_w = None
        m = alexnet(weights=tv_w)
        self.features = m.features
        self.avgpool  = m.avgpool     # -> (6,6)
        for p in self.features.parameters(): p.requires_grad = bool(trainable)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)   # (B,256,6,6)
        return x.flatten(1)   # (B,9216)

class PartDetectorAlexNet(nn.Module):
    """Backbone (frozen by default) + small MLP emb + linear classifier."""
    def __init__(self, emb_dim: int, num_classes: int, backbone_trainable: bool = False, weights: Optional[str] = None):
        super().__init__()
        self.backbone = AlexBackbone(trainable=backbone_trainable, weights=weights)
        self.emb = nn.Sequential(
            nn.Linear(256*6*6, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        z = self.emb(f)
        logits = self.cls(z)
        return z, logits


# -------------------- data helper --------------------
class PartOnlyDataset(Dataset):
    """Wrap a base PartCropDataset2 but keep only items with part == part_name."""
    def __init__(self, base_ds: Dataset, part_name: str):
        self.base = base_ds
        self.indices = [i for i, it in enumerate(base_ds.items) if it.part_canonical == part_name]
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.base[self.indices[i]]  # (x, det_idx, cluster_idx, image_id)


# -------------------- training (per detector) --------------------
def _remap_val_by_name(part_name: str, y_val: torch.Tensor, ds_val: Dataset, train_name2idx: Dict[str,int]) -> torch.Tensor:
    """
    Map val cluster indices -> cluster NAMES -> train indices, so validation is comparable even if ds_val has its own map.
    If names already match, this is a no-op.
    Unknowns -> -1 (masked out).
    """
    inv_val = {v:k for k,v in ds_val.detector_clusters.get(part_name, {}).items()}  # idx_val -> name
    out = []
    for y in y_val.tolist():
        if int(y) < 0: out.append(-1); continue
        nm = inv_val.get(int(y))
        out.append(train_name2idx.get(nm, -1))
    return torch.tensor(out, device=y_val.device, dtype=torch.long)

def train_single_detector(
    part_name: str,
    ds_train: Dataset, ds_val: Optional[Dataset],
    clusters_map: Dict[str,int],          # TRAIN: {original cat name -> idx}
    emb_dim: int = 128,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    backbone_trainable: bool = False,
    weights: Optional[str] = None,        # 'imagenet' if you have weights cached
    use_amp: bool = True,
    num_workers: int = 0,
) -> Tuple[PartDetectorAlexNet, Dict]:
    device = _device()
    K = max(clusters_map.values()) + 1 if clusters_map else 0
    if K < 1: raise ValueError(f"[{part_name}] no classes.")

    tr = PartOnlyDataset(ds_train, part_name)
    vl = PartOnlyDataset(ds_val, part_name) if ds_val is not None else None

    # labels for sampler/priors
    y_tr = [int(tr[i][2]) for i in range(len(tr)) if tr[i][2] >= 0]
    if not y_tr: raise ValueError(f"[{part_name}] no valid training labels.")
    sampler = _weighted_sampler(y_tr, K)

    tr_loader = DataLoader(tr, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    vl_loader = DataLoader(vl, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if (vl and len(vl)>0) else None

    model = PartDetectorAlexNet(emb_dim=emb_dim, num_classes=K, backbone_trainable=backbone_trainable, weights=weights).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    # weighted CE (+ smoothing if available)
    counts = [0]*K
    for y in y_tr: counts[y] += 1
    cw = _class_weights(counts).to(device)
    try:
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=cw)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type=="cuda"))
    logs = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(1, epochs+1):
        # train
        model.train(); tl=ta=0.0; nbt=0
        for xb, _, yb, _ in tr_loader:
            m = (yb >= 0)
            if m.sum().item()==0: continue
            xb = xb[m].to(device, non_blocking=True); yb = yb[m].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type=="cuda")):
                _, logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss.item()); ta += _acc(logits, yb); nbt += 1
        logs["train_loss"].append(tl/max(1,nbt)); logs["train_acc"].append(ta/max(1,nbt))

        # val
        if vl_loader is not None:
            model.eval(); vl=va=0.0; nvb=0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(use_amp and device.type=="cuda")):
                for xb, _, yb_val, _ in vl_loader:
                    # remap val labels to TRAIN indices by ORIGINAL NAME (safe even if maps already match)
                    yb = _remap_val_by_name(part_name, yb_val, ds_val, clusters_map)
                    m = (yb >= 0)
                    if m.sum().item()==0: continue
                    xb = xb[m].to(device); yb = yb[m].to(device)
                    _, logits = model(xb)
                    vloss = criterion(logits, yb)
                    vl += float(vloss.item()); va += _acc(logits, yb); nvb += 1
            logs["val_loss"].append(vl/max(1,nvb)); logs["val_acc"].append(va/max(1,nvb))
            print(f"[{part_name}] ep {ep:02d}/{epochs}  tr_loss={logs['train_loss'][-1]:.4f} tr_acc={logs['train_acc'][-1]:.3f}  "
                  f"val_loss={logs['val_loss'][-1]:.4f} val_acc={logs['val_acc'][-1]:.3f}")
        else:
            logs["val_loss"].append(None); logs["val_acc"].append(None)
            print(f"[{part_name}] ep {ep:02d}/{epochs}  tr_loss={logs['train_loss'][-1]:.4f} tr_acc={logs['train_acc'][-1]:.3f}")

    # priors for gating
    priors = torch.tensor(counts, dtype=torch.float32)
    priors = priors / priors.sum().clamp_min(1e-6)

    state = {
        "part": part_name,
        "state_dict": model.state_dict(),
        "emb_dim": emb_dim,
        "num_classes": K,
        "clusters": clusters_map,                    # name -> idx (TRAIN)
        "inv_clusters": {v:k for k,v in clusters_map.items()},
        "priors": priors.tolist(),
        "logs": logs,
    }
    return model, state


# -------------------- train all detectors --------------------
def train_alex_detectors(
    ds_train: Dataset,
    ds_val: Optional[Dataset],
    outdir: str = "alex_ckpts",
    emb_dim: int = 128,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    backbone_trainable: bool = False,
    weights: Optional[str] = None,     # set 'imagenet' if weights cached
    use_amp: bool = True,
    num_workers: int = 0,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    # discover parts from TRAIN split
    parts = sorted({it.part_canonical for it in ds_train.items})
    manifest = {"parts": parts, "emb_dim": emb_dim, "detectors": []}

    for part in parts:
        clusters_map = ds_train.detector_clusters.get(part, {})
        if len(clusters_map) < 1:
            print(f"[{part}] skipped (0 classes)."); continue
        print(f"==> Training '{part}' with {len(clusters_map)} classes")
        model, state = train_single_detector(
            part, ds_train, ds_val, clusters_map,
            emb_dim=emb_dim, epochs=epochs, batch_size=batch_size, lr=lr,
            backbone_trainable=backbone_trainable, weights=weights, use_amp=use_amp, num_workers=num_workers
        )
        torch.save(state, os.path.join(outdir, f"det_{part}.pt"))
        manifest["detectors"].append({"part": part, "file": f"det_{part}.pt"})

    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest['detectors'])} detectors to: {outdir}")


# -------------------- ensemble & fusion --------------------
class PartEnsembleAlex:
    def __init__(self, ckpt_dir: str, device: Optional[torch.device] = None):
        self.device = device or _device()
        man = json.load(open(os.path.join(ckpt_dir, "manifest.json")))
        self.parts: List[str] = man["parts"]
        self.emb_dim: int = man["emb_dim"]
        self.detectors: List[PartDetectorAlexNet] = []
        self.inv_clusters: List[Dict[int,str]] = []
        self.priors: List[torch.Tensor] = []
        self.part_names_in_order: List[str] = []

        for d in man["detectors"]:
            state = torch.load(os.path.join(ckpt_dir, d["file"]), map_location=self.device)
            m = PartDetectorAlexNet(emb_dim=state["emb_dim"], num_classes=state["num_classes"])
            m.load_state_dict(state["state_dict"], strict=True)
            m.to(self.device).eval()
            self.detectors.append(m)
            self.inv_clusters.append(state["inv_clusters"])
            self.priors.append(torch.tensor(state["priors"], dtype=torch.float32, device=self.device))
            self.part_names_in_order.append(state["part"])

    def score_all(self, x: torch.Tensor):
        zs, logits = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device.type=="cuda")):
            for d in self.detectors:
                z, lg = d(x)
                zs.append(z); logits.append(lg)
        return zs, logits


def fuse_image_embeddings(ds: Dataset, ens: PartEnsembleAlex, batch_size: int = 128, num_workers: int = 0):
    device = ens.device
    # group ds indices by image_id
    by_img = collections.defaultdict(list)
    for i, it in enumerate(ds.items):
        by_img[it.image_id].append(i)
    image_ids = sorted(by_img.keys())

    rows = []
    for iid in image_ids:
        idxs = by_img[iid]
        per_part = {p: [] for p in ens.part_names_in_order}

        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s+batch_size]
            xb = torch.cat([ds[j][0].unsqueeze(0) for j in chunk], 0).to(device, non_blocking=True)
            zs_list, lg_list = ens.score_all(xb)

            # gating: sum(softmax * prior) per detector
            softs = [F.softmax(lg, 1) if lg is not None else None for lg in lg_list]
            scores = []
            for d, sf in enumerate(softs):
                if sf is None: scores.append(torch.zeros(xb.size(0), device=device)); continue
                sc = (sf * ens.priors[d].view(1, -1)).sum(1)
                scores.append(sc)
            S = torch.stack(scores, 0).transpose(0,1)  # (B, D)
            winners = S.argmax(1).tolist()

            for bi, d_idx in enumerate(winners):
                pn = ens.part_names_in_order[d_idx]
                per_part[pn].append(zs_list[d_idx][bi].detach().cpu())

        # mean-pool per part, zero if none
        feats = []
        for pn in ens.part_names_in_order:
            if len(per_part[pn]) == 0:
                feats.append(torch.zeros(ens.emb_dim))
            else:
                feats.append(torch.stack(per_part[pn], 0).mean(0))
        rows.append(torch.cat(feats, 0))

    X = torch.stack(rows, 0)
    return X, image_ids


# -------------------- image labels & head --------------------
def image_labels_from_supercategory(ds: Dataset, image_ids: List[int], label_map: Optional[Dict[str,int]] = None):
    anns_by_img = collections.defaultdict(list)
    for it in ds.items:
        anns_by_img[it.image_id].append(it.bbox_or_seg_ann)

    names = []
    for iid in image_ids:
        cats = []
        for a in anns_by_img[iid]:
            cid = a.get("category_id")
            if cid is None: continue
            cat = ds.cat_map.get(int(cid), {})
            sc = cat.get("supercategory")
            if sc: cats.append(sc)
        names.append(collections.Counter(cats).most_common(1)[0][0] if cats else "Unknown")

    if label_map is None:
        uniq = sorted(set(names))
        label_map = {n:i for i,n in enumerate(uniq)}
    y = torch.tensor([label_map.get(n, -1) for n in names], dtype=torch.long)
    return y, label_map


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.fc(x)

def train_linear_head(Xtr: torch.Tensor, ytr: torch.Tensor, Xte: torch.Tensor, yte: torch.Tensor,
                      epochs: int = 10, lr: float = 5e-3, wd: float = 1e-4):
    device = _device()
    C = int(ytr.max().item()) + 1
    model = LinearHead(Xtr.shape[1], C).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    try:
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    except TypeError:
        crit = nn.CrossEntropyLoss()

    Xtr, ytr, Xte, yte = Xtr.to(device), ytr.to(device), Xte.to(device), yte.to(device)
    logs = {"loss": [], "val_acc": []}
    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = crit(model(Xtr), ytr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = _acc(model(Xte), yte)
        logs["loss"].append(float(loss.item())); logs["val_acc"].append(va)
        print(f"[Head] Epoch {ep}/{epochs} loss={loss.item():.4f} val_acc={va:.3f}")
    return model, logs
