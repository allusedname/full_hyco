
# -*- coding: utf-8 -*-
"""
part_detector_refactor.py  (CUDA-optimized, with ResNet backbones)

Adds:
  1) TinyCNN backbone (small, fast)
  2) SqueezeNet backbone (optional)
  3) ResNet backbones: resnet18/34/50/101/152
  4) A single-backbone, many-heads model (SharedBackboneMultiHead)
  5) CUDA-focused toggles: device override, AMP, TF32, channels_last, torch.compile, dataloader prefetch, DataParallel

Author: ChatGPT
"""

from typing import Dict, List, Optional, Tuple, Union
import os, json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

# --------- tiny backbone(s) ---------
class TinyCNNBackbone(nn.Module):
    """
    A very small CNN (~0.8M params) that takes 224x224 and returns a 256-D feature vector.
    3x3 convs with stride-2 downsampling; global average pool to (B,256).
    """
    def __init__(self, trainable: bool = True):
        super().__init__()
        C = [24, 48, 96, 192, 256]
        layers = []
        in_c = 3
        for i, out_c in enumerate(C):
            stride = 2  # downsample every stage
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B,256,1,1)
        for p in self.parameters():
            p.requires_grad = bool(trainable)

    @property
    def feat_dim(self) -> int:
        return 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)  # (B,256)
        return x

# Optional: SqueezeNet1_1
try:
    from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
except Exception:
    squeezenet1_1, SqueezeNet1_1_Weights = None, None

class SqueezeNetBackbone(nn.Module):
    def __init__(self, trainable: bool = False, weights: Optional[str] = None):
        super().__init__()
        if squeezenet1_1 is None:
            raise RuntimeError("torchvision.models.squeezenet1_1 unavailable.")
        tv_w = None
        if weights in ("imagenet","IMAGENET1K_V1"):
            try: tv_w = SqueezeNet1_1_Weights.IMAGENET1K_V1
            except Exception: tv_w = None
        m = squeezenet1_1(weights=tv_w) if tv_w is not None else squeezenet1_1()
        self.features = m.features
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B,512,1,1)
        for p in self.features.parameters():
            p.requires_grad = bool(trainable)

    @property
    def feat_dim(self) -> int:
        return 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)  # (B,512)
        return x

# --------- ResNet backbone(s) ---------
try:
    from torchvision.models import (
        resnet18, resnet34, resnet50, resnet101, resnet152,
        ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
    )
    _has_resnet_enums = True
except Exception:
    try:
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
        _has_resnet_enums = False
        ResNet18_Weights = ResNet34_Weights = ResNet50_Weights = ResNet101_Weights = ResNet152_Weights = None
    except Exception:
        resnet18 = resnet34 = resnet50 = resnet101 = resnet152 = None
        _has_resnet_enums = False
        ResNet18_Weights = ResNet34_Weights = ResNet50_Weights = ResNet101_Weights = ResNet152_Weights = None

_RESNET_FACTORY = {
    "resnet18":  (resnet18,  ResNet18_Weights),
    "resnet34":  (resnet34,  ResNet34_Weights),
    "resnet50":  (resnet50,  ResNet50_Weights),
    "resnet101": (resnet101, ResNet101_Weights),
    "resnet152": (resnet152, ResNet152_Weights),
}


# Optional: AlexNet backbone
try:
    from torchvision.models import alexnet as _alexnet_ctor, AlexNet_Weights as _AlexNet_Weights
    _has_alexnet_enum = True
except Exception:
    try:
        from torchvision.models import alexnet as _alexnet_ctor
        _has_alexnet_enum = False
        _AlexNet_Weights = None
    except Exception:
        _alexnet_ctor = None
        _has_alexnet_enum = False
        _AlexNet_Weights = None

class AlexNetBackbone(nn.Module):
    """
    Wraps torchvision AlexNet features -> (B,9216) flattened vector by default.
    """
    def __init__(self, trainable: bool = True, weights: Optional[str] = None):
        super().__init__()
        if _alexnet_ctor is None:
            raise RuntimeError("torchvision.models.alexnet unavailable.")
        tv_w = None
        if weights is not None and _has_alexnet_enum:
            try:
                tv_w = _AlexNet_Weights.IMAGENET1K_V1
            except Exception:
                tv_w = None
        try:
            m = _alexnet_ctor(weights=tv_w) if tv_w is not None else _alexnet_ctor()
        except TypeError:
            m = _alexnet_ctor(pretrained=bool(weights))
        # AlexNet has: features -> avgpool -> classifier
        self.features = m.features
        self.avgpool = m.avgpool
        self._feat_dim = 256 * 6 * 6  # default flatten size after avgpool(6x6)
        for p in self.features.parameters():
            p.requires_grad = bool(trainable)

    @property
    def feat_dim(self) -> int:
        return int(self._feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNetBackbone(nn.Module):
    def __init__(self, variant: str = "resnet50", trainable: bool = True, weights: Optional[str] = None):
        super().__init__()
        if variant not in _RESNET_FACTORY or _RESNET_FACTORY[variant][0] is None:
            raise RuntimeError(f"ResNet variant '{variant}' unavailable.")
        ctor, enum = _RESNET_FACTORY[variant]
        tv_w = None
        if weights is not None and enum is not None:
            try:
                if variant == "resnet50":
                    tv_w = ResNet50_Weights.IMAGENET1K_V2 if hasattr(ResNet50_Weights, "IMAGENET1K_V2") else ResNet50_Weights.IMAGENET1K_V1
                elif variant == "resnet18":
                    tv_w = ResNet18_Weights.IMAGENET1K_V1
                elif variant == "resnet34":
                    tv_w = ResNet34_Weights.IMAGENET1K_V1
                elif variant == "resnet101":
                    tv_w = ResNet101_Weights.IMAGENET1K_V1
                elif variant == "resnet152":
                    tv_w = ResNet152_Weights.IMAGENET1K_V1
            except Exception:
                tv_w = None
        try:
            m = ctor(weights=tv_w) if tv_w is not None else ctor()
        except TypeError:
            m = ctor(pretrained=bool(weights))
        self.trunk = nn.Sequential(*(list(m.children())[:-1]))
        self._feat_dim = m.fc.in_features if hasattr(m, "fc") else 2048
        for p in self.trunk.parameters():
            p.requires_grad = bool(trainable)
    @property
    def feat_dim(self) -> int:
        return int(self._feat_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        return x.flatten(1)

# --------- utilities ---------
def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:

    if logits is None or logits.numel() == 0:
        return 0.0
    return (logits.argmax(1) == y).float().mean().item()


def _acc_masked(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """Accuracy on subset mask; returns float('nan') if mask has no true entries."""
    if mask is None or mask.sum().item() == 0:
        return float('nan')
    y_sub = y[mask]
    lg_sub = logits[mask]
    if lg_sub.numel() == 0:
        return float('nan')
    return (lg_sub.argmax(1) == y_sub).float().mean().item()

def _class_weights(counts: List[int]) -> torch.Tensor:
    n = float(sum(counts)) if sum(counts) > 0 else 1.0
    inv = [n/max(1,c) for c in counts]
    s = sum(inv)
    return torch.tensor([v/s for v in inv], dtype=torch.float32)

class EarlyStopper:
    def __init__(self, monitor: str = "val_acc", patience: int = 8, min_delta: float = 1e-3):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0
        self.mode = "max" if monitor == "val_acc" else "min"
    def step(self, current: Optional[float]) -> bool:
        if current is None:
            return False
        if self.best is None:
            self.best = current; self.bad_epochs = 0; return False
        improved = (current - self.best) > self.min_delta if self.mode == "max" else (self.best - current) > self.min_delta
        if improved:
            self.best = current; self.bad_epochs = 0; return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience

def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def _enable_cuda_fastpaths(device: torch.device, use_tf32: bool):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = bool(use_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

def _maybe_channels_last(m: nn.Module, enable: bool):
    if enable:
        try:
            m.to(memory_format=torch.channels_last)
        except Exception:
            pass

def _to_device_batch(x: torch.Tensor, device: torch.device, channels_last: bool):
    x = x.to(device, non_blocking=True)
    if channels_last and x.ndim == 4:
        try:
            x = x.to(memory_format=torch.channels_last)
        except Exception:
            pass
    return x

# --------- generic per-part detector (supports resnet) ---------
class PartDetectorTiny(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, backbone: str = "tinycnn", backbone_trainable: bool = True, weights: Optional[str] = None):
        super().__init__()
        if backbone == "tinycnn":
            self.backbone = TinyCNNBackbone(trainable=backbone_trainable)
        elif backbone == "squeezenet":
            self.backbone = SqueezeNetBackbone(trainable=backbone_trainable, weights=weights)
        elif backbone == "alexnet":
            self.backbone = AlexNetBackbone(trainable=backbone_trainable, weights=weights)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(variant=backbone, trainable=backbone_trainable, weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.emb = nn.Sequential(
            nn.Linear(self.backbone.feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.cls = nn.Linear(emb_dim, num_classes)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        z = self.emb(f)
        logits = self.cls(z)
        return z, logits

# --------- optimizer param groups & scheduler helpers ---------
def _param_groups_adamw(model: nn.Module, wd: float, norm_wd: float, bias_wd: float):
    decay, norms, biases = [], [], []
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    for module_name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if not param.requires_grad: 
                continue
            if pname.endswith("bias"):
                biases.append(param)
            elif isinstance(module, norm_types):
                norms.append(param)
            else:
                decay.append(param)
    seen = set(map(id, decay + norms + biases))
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in seen:
            if name.endswith("bias"):
                biases.append(param); seen.add(id(param))
            else:
                decay.append(param); seen.add(id(param))
    groups = [
        {"params": decay,  "weight_decay": float(wd)},
        {"params": norms,  "weight_decay": float(norm_wd)},
        {"params": biases, "weight_decay": float(bias_wd)},
    ]
    return groups

def _build_scheduler(optimizer, scheduler_type: str, epochs: int, warmup_epochs: int = 0):
    scheduler_type = (scheduler_type or "plateau").lower()
    if scheduler_type == "cosine":
        try:
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
            main = CosineAnnealingLR(optimizer, T_max=max(1, epochs - max(0, warmup_epochs)))
            if warmup_epochs and warmup_epochs > 0:
                warm = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
                sch = SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup_epochs])
            else:
                sch = main
            return sch, "cosine"
        except Exception:
            pass
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    return sch, "plateau"

# --------- per-part trainer ---------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1); p = torch.exp(logp)
        focal = (1 - p).pow(self.gamma)
        return F.nll_loss(focal * logp, target, weight=self.weight, reduction=self.reduction)


def _compute_informative_categories_for_part(ds_part: Dataset, label_fn, batch_size: int, num_workers: int, device: torch.device, min_clusters: int = 2) -> set:
    """
    Iterate the per-part Dataset view and return the set of CATEGORY ids that have >= min_clusters
    distinct clusters for this part.
    """
    if label_fn is None:
        return set()
    from torch.utils.data import DataLoader
    dl = DataLoader(ds_part, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers//2), pin_memory=(device.type=="cuda"))
    cats_to_clusters = {}
    for batch in dl:
        if not isinstance(batch, (list, tuple)) or len(batch) < 4:
            continue
        _, _, y_part, meta = batch
        m = (y_part >= 0)
        if m.sum().item() == 0:
            continue
        y_part = y_part[m].cpu()
        # label_fn expects a list of metas
        meta_m = [meta[i] for i,v in enumerate(m.tolist()) if v]
        y_cat = label_fn(meta_m)
        if not isinstance(y_cat, torch.Tensor):
            y_cat = torch.tensor(y_cat, dtype=torch.long)
        y_cat = y_cat.cpu()
        for yc, yp in zip(y_cat.tolist(), y_part.tolist()):
            cats_to_clusters.setdefault(int(yc), set()).add(int(yp))
    informative = {c for c, s in cats_to_clusters.items() if len(s) >= min_clusters}
    return informative


def _build_informative_categories_for_part(ds_part, label_fn, min_clusters: int = 2, device: Optional[Union[str, torch.device]] = None) -> set:
    dev = _resolve_device(device)
    dl = DataLoader(ds_part, batch_size=256, shuffle=False, num_workers=0, pin_memory=(dev.type=="cuda"))
    cat2clusters = {}
    for batch in dl:
        if not isinstance(batch, (list, tuple)) or len(batch) < 4:
            continue
        _, _, y_part, meta = batch
        y_cat = label_fn(meta)
        if not isinstance(y_cat, torch.Tensor):
            y_cat = torch.tensor(y_cat, dtype=torch.long)
        y_part = y_part.cpu(); y_cat = y_cat.cpu()
        for c, k in zip(y_cat.tolist(), y_part.tolist()):
            if k < 0:  # unlabeled
                continue
            s = cat2clusters.get(int(c), set())
            s.add(int(k))
            cat2clusters[int(c)] = s
    informative = {c for c, s in cat2clusters.items() if len(s) >= int(min_clusters)}
    return informative

def train_small_detectors(
    ds_train: Dataset,
    ds_val: Optional[Dataset],
    outdir: str = "tiny_ckpts",
    emb_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    backbone: str = "tinycnn",
    backbone_trainable: bool = True,
    weights: Optional[str] = None,
    use_amp: bool = True,
    # CUDA / perf toggles
    device: Optional[Union[str, torch.device]] = None,
    channels_last: bool = True,
    use_tf32: bool = True,
    use_compile: bool = False,
    data_parallel: bool = False,
    # Dataloader perf
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    # Early stop & scheduler
    early_stop: bool = True,
    patience: int = 8,
    min_delta: float = 1e-3,
    weight_decay: float = 1e-4, norm_weight_decay: float = 0.0, bias_weight_decay: float = 0.0,
    scheduler_type: str = 'plateau', warmup_epochs: int = 0,
    # Validation metric
    val_metric: str = 'cluster',
    min_clusters_per_category: int = 2,
    label_fn = None,
    # Loss
    loss_type: str = 'ce',
    focal_gamma: float = 2.0,
    log_every: int = 1):
    device = _resolve_device(device)
    _enable_cuda_fastpaths(device, use_tf32=use_tf32)
    os.makedirs(outdir, exist_ok=True)

    parts: List[str] = list(ds_train.detector_clusters.keys())
    manifest = {"parts": parts, "emb_dim": emb_dim, "arch": f"per_part::{backbone}", "val_metric": val_metric, "detectors": []}

    for part_name in parts:
        # ======== revised criterion: multi-cluster-aware metrics ========
        has_label_fn = (label_fn is not None)

        class PartOnly(Dataset):
            def __init__(self, base_ds, part):
                self.base = base_ds
                self.part = part
                self.ids = [i for i,it in enumerate(base_ds.items) if it.part_canonical == part]
            def __len__(self): return len(self.ids)
            def __getitem__(self, i): return self.base[self.ids[i]]

        tr = PartOnly(ds_train, part_name)
        vl = PartOnly(ds_val,   part_name) if ds_val is not None else None

        # precompute informative categories from TRAIN for this part
        informative_cats = _compute_informative_categories_for_part(tr, label_fn, batch_size=max(64, batch_size//2), num_workers=num_workers, device=device, min_clusters=min_clusters_per_category) if has_label_fn else set()


        K = len(ds_train.detector_clusters[part_name])
        y_tr = [int(tr[i][2]) for i in range(len(tr)) if tr[i][2] >= 0]
        assert len(y_tr) > 0, f"[{part_name}] no valid training labels."

        counts = [0]*K
        for y in y_tr: counts[y] += 1
        inv = [1.0/max(1,c) for c in counts]
        weights_ = [inv[int(tr[i][2])] for i in range(len(tr)) if tr[i][2] >= 0]
        sampler = WeightedRandomSampler(torch.tensor(weights_, dtype=torch.double),
                                        num_samples=len(weights_), replacement=True)

        dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=(device.type=="cuda"))
        if num_workers > 0:
            dl_kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

        tr_loader = DataLoader(tr, sampler=sampler, **dl_kwargs)
        vl_loader = DataLoader(vl, shuffle=False, **dl_kwargs) if (vl and len(vl)>0) else None

        model = PartDetectorTiny(emb_dim=emb_dim, num_classes=K, backbone=backbone, backbone_trainable=backbone_trainable, weights=weights)
        model = model.to(device)
        _maybe_channels_last(model, channels_last)
        if use_compile and hasattr(torch, "compile") and not data_parallel:
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception:
                pass
        if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        opt = torch.optim.AdamW(_param_groups_adamw(model, wd=weight_decay, norm_wd=norm_weight_decay, bias_wd=bias_weight_decay), lr=lr)

        if loss_type == 'focal':
            criterion = FocalLoss(gamma=focal_gamma, weight=_class_weights(counts).to(device))
        else:
            try: criterion = nn.CrossEntropyLoss(weight=_class_weights(counts).to(device), label_smoothing=0.1)
            except TypeError: criterion = nn.CrossEntropyLoss(weight=_class_weights(counts).to(device))

        scheduler, _sch_kind = _build_scheduler(opt, scheduler_type, epochs, warmup_epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type=="cuda"))
        stopper = EarlyStopper(monitor="val_acc", patience=patience, min_delta=min_delta) if early_stop else None

        # Informative categories for validation metric
        informative_cats = None
        if val_metric in ('cluster_informative',) and (vl_loader is not None) and (label_fn is not None):
            try:
                informative_cats = _build_informative_categories_for_part(vl, label_fn, min_clusters=min_clusters_per_category, device=device)
                if len(informative_cats) == 0:
                    print(f"[{part_name}] Warning: no informative categories (>= {min_clusters_per_category} clusters) found in val set; falling back to raw cluster acc.")
                    informative_cats = None
            except Exception as e:
                print(f"[{part_name}] Unable to compute informative categories ({e}); falling back to raw cluster acc.")
                informative_cats = None

        best_state = None; best_ep = -1; best_va = -1.0
        for ep in range(1, epochs+1):
            tr_acc_mc = 0.0; tr_mc_batches = 0
            model.train(); tr_sum = tr_acc = nb = 0
            for xb, _, yb, meta in tr_loader:
                m = (yb >= 0)
                if m.sum().item()==0: continue
                xb = _to_device_batch(xb[m], device, channels_last)
                yb = yb[m].to(device, non_blocking=True)
                # category labels for multi-cluster metric
                if has_label_fn and meta is not None:
                    meta_m = [meta[i] for i,v in enumerate(m.tolist()) if v]
                    y_cat = label_fn(meta_m)
                    if not isinstance(y_cat, torch.Tensor):
                        y_cat = torch.tensor(y_cat, dtype=torch.long)
                    y_cat = y_cat.to(device, non_blocking=True)
                    keep_mask = torch.tensor([int(c.item()) in informative_cats for c in y_cat], device=device, dtype=torch.bool)
                else:
                    y_cat = None
                    keep_mask = None
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                    _, logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                tr_sum += float(loss.item()); tr_acc += _acc(logits, yb); 
                # multi-cluster-only acc
                if keep_mask is not None and keep_mask.any().item():
                    tr_acc_mc = tr_acc_mc + (_acc_masked(logits, yb, keep_mask) if keep_mask is not None else float('nan'))
                    tr_mc_batches += 1
                nb += 1
            tr_loss = tr_sum/max(1,nb); tr_acc = tr_acc/max(1,nb)

            if vl_loader is not None:
                model.eval(); vl_sum = va_sum = nvb = 0; va_sum_mc = 0.0; nvb_mc = 0; va_inf_sum = nvb_inf = 0
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                    for xb, _, yb, meta in vl_loader:
                        m = (yb >= 0)
                        if m.sum().item()==0: continue
                        xb = _to_device_batch(xb[m], device, channels_last)
                        yb = yb[m].to(device, non_blocking=True)
                        _, logits = model(xb)
                        vloss = criterion(logits, yb)
                        vl_sum += float(vloss.item()); va_sum += _acc(logits, yb); nvb += 1

                        if (informative_cats is not None) and (label_fn is not None):
                            try:
                                # build y_cat aligned to the masked samples
                                keep_meta = [meta[i] for i in torch.nonzero(m, as_tuple=False).squeeze(1).tolist()]
                                y_cat = label_fn(keep_meta)
                                if not isinstance(y_cat, torch.Tensor):
                                    y_cat = torch.tensor(y_cat, dtype=torch.long)
                                y_cat = y_cat.to(yb.device, non_blocking=True)
                                keep_mask = torch.tensor([int(c.item()) in informative_cats for c in y_cat], device=yb.device, dtype=torch.bool)
                                if keep_mask.sum().item() > 0:
                                    va_inf_sum += _acc(logits[keep_mask], yb[keep_mask]); nvb_inf += 1
                            except Exception:
                                pass

                val_loss = vl_sum/max(1,nvb)
                val_acc_raw = va_sum/max(1,nvb)
                val_acc_inf = (va_inf_sum/max(1,nvb_inf)) if nvb_inf > 0 else float('nan')

                val_acc = val_acc_inf if (val_metric == 'cluster_informative' and not math.isnan(val_acc_inf)) else val_acc_raw

                scheduler.step(val_loss) if _sch_kind=='plateau' else scheduler.step()

                if val_acc > best_va + min_delta:
                    best_va, best_ep = val_acc, ep
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                print(f"[{part_name}] ep {ep:02d}/{epochs}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} (raw={val_acc_raw:.3f}, inf={val_acc_inf:.3f})  lr={opt.param_groups[0]['lr']:.2e}")
                if stopper and stopper.step(val_acc):
                    print(f"[{part_name}] early stop at epoch {ep} (best ep {best_ep} metric={best_va:.3f})")
                    break
            else:
                print(f"[{part_name}] ep {ep:02d}/{epochs}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}")

        if best_state is not None:
            target = model.module if isinstance(model, nn.DataParallel) else model
            target.load_state_dict(best_state, strict=True)

        tot = float(sum(counts)) if sum(counts)>0 else 1.0
        priors = [c/tot for c in counts]
        inv_clusters = {int(v): str(k) for k,v in ds_train.detector_clusters[part_name].items()}

        state = dict(
            part=part_name,
            emb_dim=emb_dim,
            num_classes=K,
            inv_clusters=inv_clusters,
            priors=priors,
            state_dict=(model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
            backbone=backbone,
        )
        torch.save(state, os.path.join(outdir, f"det_{part_name}.pt"))
        manifest["detectors"].append({"part": part_name, "file": f"det_{part_name}.pt"})
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest['detectors'])} tiny detectors to: {outdir}")

# --------- per-part ensemble ---------
class PartEnsembleTiny:
    def __init__(self, ckpt_dir: str, device: Optional[Union[str, torch.device]] = None, channels_last: bool = True, use_tf32: bool = True):
        self.device = _resolve_device(device)
        _enable_cuda_fastpaths(self.device, use_tf32=use_tf32)
        man = json.load(open(os.path.join(ckpt_dir, "manifest.json"), "r"))
        self.parts: List[str] = man["parts"]
        self.emb_dim: int = man["emb_dim"]
        self.detectors: List[PartDetectorTiny] = []
        self.inv_clusters: List[Dict[int,str]] = []
        self.priors: List[torch.Tensor] = []
        self.part_names_in_order: List[str] = []
        self.backbones: List[str] = []
        self.channels_last = channels_last
        for d in man["detectors"]:
            state = torch.load(os.path.join(ckpt_dir, d["file"]), map_location=self.device)
            m = PartDetectorTiny(emb_dim=state["emb_dim"], num_classes=state["num_classes"], backbone=state.get("backbone","tinycnn"))
            m.load_state_dict(state["state_dict"], strict=True)
            m.to(self.device)
            _maybe_channels_last(m, channels_last)
            m.eval()
            self.detectors.append(m)
            self.inv_clusters.append(state["inv_clusters"])
            self.priors.append(torch.tensor(state["priors"], dtype=torch.float32, device=self.device))
            self.part_names_in_order.append(state["part"])
            self.backbones.append(state.get("backbone","tinycnn"))

    def score_all(self, x: torch.Tensor):
        zs, logits = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device.type=="cuda")):
            for d in self.detectors:
                xb = _to_device_batch(x, self.device, self.channels_last)
                z, lg = d(xb)
                zs.append(z); logits.append(lg)
        return zs, logits

# --------- shared backbone (unchanged compile-safe score_all routing) ---------
class SharedBackboneMultiHead(nn.Module):
    def __init__(self, parts_to_K: Dict[str, int], emb_dim: int = 128, backbone: str = "tinycnn", backbone_trainable: bool = True, weights: Optional[str] = None):
        super().__init__()
        if backbone == "tinycnn":
            self.backbone = TinyCNNBackbone(trainable=backbone_trainable)
        elif backbone == "squeezenet":
            self.backbone = SqueezeNetBackbone(trainable=backbone_trainable, weights=weights)
        elif backbone == "alexnet":
            self.backbone = AlexNetBackbone(trainable=backbone_trainable, weights=weights)
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(variant=backbone, trainable=backbone_trainable, weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.parts_order = list(parts_to_K.keys())
        self.emb_dim = emb_dim
        self.emb_heads = nn.ModuleDict()
        self.cls_heads = nn.ModuleDict()
        for p, K in parts_to_K.items():
            self.emb_heads[p] = nn.Sequential(
                nn.Linear(self.backbone.feat_dim, emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
            self.cls_heads[p] = nn.Linear(emb_dim, K)
    def forward(self, x: torch.Tensor, part_names: List[str]):
        f = self.backbone(x)
        out = {}
        for p in set(part_names):
            z = self.emb_heads[p](f)
            lg = self.cls_heads[p](z)
            out[p] = (z, lg)
        return out
    def score_all(self, x: torch.Tensor):
        f = self.backbone(x)
        zs, logits = [], []
        for p in self.parts_order:
            z = self.emb_heads[p](f)
            lg = self.cls_heads[p](z)
            zs.append(z); logits.append(lg)
        return zs, logits

def build_sample_weights_for_shared(ds_train: Dataset) -> Tuple[List[float], Dict[str, List[int]]]:
    parts = list(ds_train.detector_clusters.keys())
    counts: Dict[str, List[int]] = {p: [0]*len(ds_train.detector_clusters[p]) for p in parts}
    for it in ds_train.items:
        p = it.part_canonical
        c = ds_train.detector_clusters[p].get(it.cluster_name, -1)
        if c >= 0:
            counts[p][c] += 1
    weights: List[float] = []
    inv: Dict[str, List[float]] = {p: [1.0/max(1,c) for c in counts[p]] for p in parts}
    for it in ds_train.items:
        p = it.part_canonical
        c = ds_train.detector_clusters[p].get(it.cluster_name, -1)
        w = inv[p][c] if c >= 0 else 0.0
        weights.append(w)
    return weights, counts

def train_shared_multihead(
    ds_train: Dataset,
    ds_val: Optional[Dataset],
    outdir: str = "shared_ckpt",
    emb_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    backbone: str = "tinycnn",
    backbone_trainable: bool = True,
    weights: Optional[str] = None,
    use_amp: bool = True,
    # CUDA / perf toggles
    device: Optional[Union[str, torch.device]] = None,
    channels_last: bool = True,
    use_tf32: bool = True,
    use_compile: bool = False,
    # Dataloader perf
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    # Early stop
    early_stop: bool = True,
    patience: int = 8,
    min_delta: float = 1e-3,
    weight_decay: float = 1e-4, norm_weight_decay: float = 0.0, bias_weight_decay: float = 0.0,
    scheduler_type: str = 'plateau', warmup_epochs: int = 0,
):
    device = _resolve_device(device)
    _enable_cuda_fastpaths(device, use_tf32=use_tf32)
    os.makedirs(outdir, exist_ok=True)

    parts = list(ds_train.detector_clusters.keys())
    parts_to_K = {p: len(ds_train.detector_clusters[p]) for p in parts}
    model = SharedBackboneMultiHead(parts_to_K, emb_dim=emb_dim, backbone=backbone, backbone_trainable=backbone_trainable, weights=weights).to(device)
    _maybe_channels_last(model, channels_last)
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            try:
                model.backbone = torch.compile(model.backbone, mode="max-autotune")
            except Exception:
                pass

    sample_weights, counts = build_sample_weights_for_shared(ds_train)
    sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double),
                                    num_samples=int(sum(w>0 for w in sample_weights)),
                                    replacement=True)

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=(device.type=="cuda"))
    if num_workers > 0:
        dl_kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    tr_loader = DataLoader(ds_train, sampler=sampler, **dl_kwargs)
    vl_loader = DataLoader(ds_val,   shuffle=False, **dl_kwargs) if (ds_val and len(ds_val)>0) else None

    crit: Dict[str, nn.Module] = {}
    for p in parts:
        cw = _class_weights(counts[p]).to(device)
        try: crit[p] = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        except TypeError: crit[p] = nn.CrossEntropyLoss(weight=cw)

    opt = torch.optim.AdamW(_param_groups_adamw(model, wd=weight_decay, norm_wd=norm_weight_decay, bias_wd=bias_weight_decay), lr=lr)
    scheduler, _sch_kind = _build_scheduler(opt, scheduler_type, epochs, warmup_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type=="cuda"))
    stopper = EarlyStopper(monitor="val_acc", patience=patience, min_delta=min_delta) if early_stop else None

    best_state = None; best_ep = -1; best_va = -1.0
    for ep in range(1, epochs+1):
            tr_acc_mc = 0.0; tr_mc_batches = 0
            model.train(); tr_loss_sum = 0.0; tr_acc_sum = 0.0; tr_batches = 0
            for xb, det_idx, yb, _ in tr_loader:
                m = (yb >= 0)
                if m.sum().item()==0: continue
                xb = _to_device_batch(xb[m], device, channels_last)
                yb = yb[m].to(device, non_blocking=True)
                det_idx_t = det_idx[m].to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                    zs_all, logits_all = model.score_all(xb)
                    loss = 0.0; acc_sum = 0.0; groups = 0
                    for p_idx, p_name in enumerate(parts):
                        mask = (det_idx_t == p_idx)
                        if mask.sum().item()==0: continue
                        logits = logits_all[p_idx][mask]
                        y = yb[mask]
                        loss = loss + crit[p_name](logits, y)
                        acc_sum += _acc(logits, y); groups += 1
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()

                tr_loss_sum += float(loss.item()); tr_acc_sum += (acc_sum/max(1,groups)); tr_batches += 1
            tr_loss = tr_loss_sum/max(1,tr_batches); tr_acc = tr_acc_sum/max(1,tr_batches)

            if vl_loader is not None:
                model.eval(); vl_loss_sum = 0.0; va_sum = 0.0; vl_batches = 0
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                    for xb, det_idx, yb, _ in vl_loader:
                        m = (yb >= 0)
                        if m.sum().item()==0: continue
                        xb = _to_device_batch(xb[m], device, channels_last)
                        yb = yb[m].to(device, non_blocking=True)
                        det_idx_t = det_idx[m].to(device, non_blocking=True)

                        zs_all, logits_all = model.score_all(xb)
                        loss = 0.0; acc_sum = 0.0; groups = 0
                        for p_idx, p_name in enumerate(parts):
                            mask = (det_idx_t == p_idx)
                            if mask.sum().item()==0: continue
                            logits = logits_all[p_idx][mask]
                            y = yb[mask]
                            loss = loss + crit[p_name](logits, y)
                            acc_sum += _acc(logits, y); groups += 1
                        vl_loss_sum += float(loss.item()); va_sum += (acc_sum/max(1,groups)); vl_batches += 1
                val_loss, val_acc = vl_loss_sum/max(1,vl_batches), va_sum/max(1,vl_batches)
                scheduler.step(val_loss) if _sch_kind=='plateau' else scheduler.step()
                if val_acc > best_va + min_delta:
                    best_va, best_ep = val_acc, ep
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                print(f"[shared:{backbone}] ep {ep:02d}/{epochs}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  lr={opt.param_groups[0]['lr']:.2e}")
                if stopper and stopper.step(val_acc):
                    print(f"[shared:{backbone}] early stop at epoch {ep} (best ep {best_ep} val_acc={best_va:.3f})")
                    break
            else:
                print(f"[shared:{backbone}] ep {ep:02d}/{epochs}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    priors = {p: [c/max(1,sum(counts[p])) for c in counts[p]] for p in parts}
    inv_clusters = {p: {int(v): str(k) for k,v in ds_train.detector_clusters[p].items()} for p in parts}

    ckpt = dict(
        arch=f"shared::{backbone}",
        emb_dim=emb_dim,
        parts=parts,
        parts_to_K=parts_to_K,
        priors=priors,
        inv_clusters=inv_clusters,
        state_dict=model.state_dict(),
    )
    torch.save(ckpt, os.path.join(outdir, "shared.pt"))
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump({"arch": f"shared::{backbone}", "emb_dim": emb_dim, "parts": parts, "file": "shared.pt"}, f, indent=2)
    print(f"Saved shared multi-head model to {outdir}/shared.pt")

class PartEnsembleShared:
    def __init__(self, ckpt_dir: str, device: Optional[Union[str, torch.device]] = None, channels_last: bool = True, use_tf32: bool = True):
        self.device = _resolve_device(device)
        _enable_cuda_fastpaths(self.device, use_tf32=use_tf32)
        man = json.load(open(os.path.join(ckpt_dir, "manifest.json"), "r"))
        ck = torch.load(os.path.join(ckpt_dir, man["file"]), map_location=self.device)
        self.emb_dim: int = ck["emb_dim"]
        self.parts: List[str] = ck["parts"]
        self.part_names_in_order: List[str] = list(self.parts)
        self.priors: List[torch.Tensor] = [torch.tensor(ck["priors"][p], dtype=torch.float32, device=self.device) for p in self.parts]
        self.inv_clusters: List[Dict[int,str]] = [ck["inv_clusters"][p] for p in self.parts]

        backbone = "tinycnn"
        if isinstance(ck.get("arch",""), str) and "::" in ck["arch"]:
            backbone = ck["arch"].split("::",1)[1]
        self.model = SharedBackboneMultiHead(ck["parts_to_K"], emb_dim=self.emb_dim, backbone=backbone).to(self.device)
        _maybe_channels_last(self.model, channels_last)
        self.model.load_state_dict(ck["state_dict"], strict=True)
        self.model.eval()
        self.channels_last = channels_last

    def score_all(self, x: torch.Tensor):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device.type=="cuda")):
            xb = _to_device_batch(x, self.device, self.channels_last)
            zs, logits = self.model.score_all(xb)
        return zs, logits


def train_alex_detectors_revised(
    ds_train: Dataset,
    ds_val: Optional[Dataset],
    outdir: str = "alex_ckpts_revised",
    emb_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    backbone_trainable: bool = True,
    weights: Optional[str] = "imagenet",
    device: Optional[Union[str, torch.device]] = None,
    use_amp: bool = True,
    channels_last: bool = True,
    use_tf32: bool = True,
    # perf
    num_workers: int = 8,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    # regularization & scheduler (same defaults as the main trainer)
    weight_decay: float = 1e-4, norm_weight_decay: float = 0.0, bias_weight_decay: float = 0.0,
    scheduler_type: str = 'plateau', warmup_epochs: int = 0,
    # revised metrics
    label_fn=None, min_clusters_per_category: int = 2, log_every: int = 1,
):
    """
    Convenience wrapper to train per-part detectors with **AlexNet** backbone and
    multi-cluster-aware metrics/logging.
    """
    return train_small_detectors(
        ds_train=ds_train, ds_val=ds_val, outdir=outdir, emb_dim=emb_dim, epochs=epochs, batch_size=batch_size,
        lr=lr, backbone="alexnet", backbone_trainable=backbone_trainable, weights=weights,
        use_amp=use_amp, device=device, channels_last=channels_last, use_tf32=use_tf32,
        num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
        weight_decay=weight_decay, norm_weight_decay=norm_weight_decay, bias_weight_decay=bias_weight_decay,
        scheduler_type=scheduler_type, warmup_epochs=warmup_epochs,
        label_fn=label_fn, min_clusters_per_category=min_clusters_per_category, log_every=log_every,
    )
