
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
from collections import deque
import numpy as np

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

def _remap_val_indices_by_name(part_name: str, y_val: torch.Tensor, ds_val: Dataset,
                               train_name2idx: Dict[str, int]) -> torch.Tensor:
    """
    Remap y_val (val-split cluster indices) to the TRAIN index space using original cluster names.
    Unknowns -> -1 (masked out).
    """
    inv_val = {v: k for k, v in ds_val.detector_clusters.get(part_name, {}).items()}  # idx_val -> name
    y_list = y_val.detach().cpu().tolist()
    out = []
    for y in y_list:
        yi = int(y)
        if yi < 0:
            out.append(-1); continue
        nm = inv_val.get(yi, None)
        out.append(int(train_name2idx.get(nm, -1)) if nm is not None else -1)
    return torch.tensor(out, dtype=torch.long, device=y_val.device)

def _balanced_accuracy(logits: torch.Tensor, y: torch.Tensor, K: int) -> float:
    """
    Macro (per-class) accuracy to reduce class-imbalance bias.
    y is already masked to valid indices in [0..K-1].
    """
    if logits.numel() == 0 or y.numel() == 0:
        return float("nan")
    pred = logits.argmax(1)
    tot = torch.bincount(y, minlength=K).float()
    cor = torch.bincount(y[pred == y], minlength=K).float()
    per_cls = torch.where(tot > 0, cor / tot.clamp_min(1), torch.full_like(tot, float("nan")))
    # mean over classes that appear in this split
    m = torch.isfinite(per_cls)
    return (per_cls[m].mean().item() if m.any() else float("nan"))


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

from typing import Optional, Union, List, Dict, Tuple
import os, json, math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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
    # CUDA / perf
    device: Optional[Union[str, torch.device]] = None,
    channels_last: bool = True,
    use_tf32: bool = True,
    use_compile: bool = False,
    data_parallel: bool = False,
    # DataLoader perf
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    # Early stop & scheduler
    early_stop: bool = True,
    patience: int = 10,
    min_delta: float = 1e-3,
    weight_decay: float = 1e-4, norm_weight_decay: float = 0.0, bias_weight_decay: float = 0.0,
    scheduler_type: str = 'plateau', warmup_epochs: int = 0,
    # Validation / metrics
    val_metric: str = 'cluster',            # 'cluster', 'cluster_balanced', or 'cluster_informative'
    min_clusters_per_category: int = 2,
    label_fn = None,
    log_every: int = 1
,
    use_background: bool = False
):
    """
    Trains one tiny detector per canonical part in ds_train.detector_clusters.

    K >= 2 (multiclass):
      * CE computed in float32 (outside autocast) to avoid AMP dtype mismatch with class weights
      * Reports val_acc_raw and val_acc_bal (balanced/macro), optional early stop on balanced

    K == 1 (one-class):
      * Center loss vs delayed center (EMA/queue) + variance floor + small covariance penalty (all float32)
      * Logs dispersion s=mean(std) and binary validation metrics vs negatives: val_oc_auc / val_oc_bacc

    Always:
      * Balanced sampler with one weight per dataset element (invalid labels → 0)
      * Validation labels remapped by ORIGINAL cluster names → TRAIN index space before scoring
      * Saves priors (K>=2) and center/var (always) for fusion
    """

    # ---------------- local helpers ----------------
    def _maybe_channels_last(m: nn.Module, flag: bool):
        if flag:
            try: m.to(memory_format=torch.channels_last)
            except Exception: pass

    def _remap_val_indices_by_name(part_name: str, y_val: torch.Tensor, ds_val_: Dataset,
                                   train_name2idx_: Dict[str, int]) -> torch.Tensor:
        """
        Map val indices -> val names -> train indices; missing classes -> -1.
        """
        inv_val = {v: k for k, v in (ds_val_.detector_clusters.get(part_name, {}) or {}).items()}  # idx_val -> name
        mapped = []
        for yi in y_val.detach().cpu().tolist():
            yi = int(yi)
            if yi < 0:
                mapped.append(-1); continue
            nm = inv_val.get(yi, None)
            mapped.append(int(train_name2idx_.get(nm, -1)) if nm is not None else -1)
        return torch.tensor(mapped, dtype=torch.long, device=y_val.device)

    def _balanced_accuracy(logits: torch.Tensor, y: torch.Tensor, K_: int) -> float:
        """
        Macro (per-class) recall averaged over classes present in y.
        """
        if logits.numel() == 0 or y.numel() == 0:
            return float("nan")
        pred = logits.argmax(1)
        tot = torch.bincount(y, minlength=K_).float()
        cor = torch.bincount(y[pred == y], minlength=K_).float()
        per_cls = torch.where(tot > 0, cor / tot.clamp_min(1), torch.full_like(tot, float("nan")))
        m = torch.isfinite(per_cls)
        return (per_cls[m].mean().item() if m.any() else float("nan"))

    def _oc_scores(z32: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """RBF/Mahalanobis score in [0,1] for one-class parts (higher=more in-class)."""
        d2 = ((z32 - mu.unsqueeze(0))**2 / (var.unsqueeze(0) + 1e-6)).sum(1)
        return torch.exp(-0.5 * d2)

    def _roc_auc_and_best_balanced(pos: torch.Tensor, neg: torch.Tensor) -> Tuple[float, float]:
        """Returns (ROC AUC, best balanced accuracy) for positive/negative scores."""
        if pos.numel() == 0 or neg.numel() == 0:
            return float('nan'), float('nan')
        s = torch.cat([pos, neg], 0)
        y = torch.cat([torch.ones_like(pos, dtype=torch.long), torch.zeros_like(neg, dtype=torch.long)], 0)
        order = torch.argsort(s, descending=True)
        y = y[order]
        P = int((y == 1).sum().item()); N = int(y.numel() - P)
        if P == 0 or N == 0:
            return float('nan'), float('nan')
        tp = fp = 0
        prev_tpr = prev_fpr = 0.0
        auc = 0.0
        best_J = -1.0
        for lab in y.tolist():
            if lab == 1: tp += 1
            else:        fp += 1
            tpr = tp / P
            fpr = fp / N
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5  # trapezoid area
            J = tpr - fpr
            if J > best_J: best_J = J
            prev_tpr, prev_fpr = tpr, fpr
        bacc = 0.5 * (1.0 + best_J)
        return float(auc), float(bacc)

    class NotPartVal(Dataset):
        """All items from ds_val whose canonical part != current part_name."""
        def __init__(self, base_ds, part):
            self.base = base_ds
            if base_ds is None:
                self.ids = []
            else:
                self.ids = [i for i, it in enumerate(base_ds.items) if getattr(it, "part_canonical", None) != part]
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return self.base[self.ids[i]]

    # ---------------- setup ----------------
    device = _resolve_device(device)
    _enable_cuda_fastpaths(device, use_tf32=use_tf32)
    os.makedirs(outdir, exist_ok=True)

    parts: List[str] = list(ds_train.detector_clusters.keys())
    manifest = {
        "parts": parts,
        "emb_dim": emb_dim,
        "arch": f"per_part::{backbone}",
        "val_metric": val_metric,
        "detectors": []
    }

    step_print_every = None  # e.g., set to 50 for per-step OC diagnostics

    # ---------------- per-part ----------------
    for part_name in parts:

        # Per-part dataset view
        class PartOnly(Dataset):
            def __init__(self, base_ds, part):
                self.base = base_ds
                self.part = part
                self.ids = [i for i, it in enumerate(base_ds.items) if getattr(it, "part_canonical", None) == part]
            def __len__(self): return len(self.ids)
            def __getitem__(self, i): return self.base[self.ids[i]]

        
        class PartVsRest(Dataset):
            """All items; positives keep their cluster index (0..K-1), negatives become background K."""
            def __init__(self, base_ds, part, K, name2idx):
                self.base = base_ds
                self.part = part
                self.K = int(K)  # number of positive clusters
                self.name2idx = name2idx or {}
                if base_ds is None:
                    self.ids = []
                else:
                    # include everything for negatives
                    try:
                        self.ids = list(range(len(base_ds.items)))
                    except Exception:
                        self.ids = list(range(len(base_ds)))
            def __len__(self): return len(self.ids)
            def __getitem__(self, i):
                x, det_idx, y_cluster, meta = self.base[self.ids[i]]
                pcanon = getattr(meta, "part_canonical", None)
                if pcanon == self.part:
                    # keep cluster index if valid; otherwise mark as unlabeled (-1)
                    yc = int(y_cluster)
                    if yc < 0 or yc >= self.K:
                        yc = -1
                    return x, torch.tensor(0), torch.tensor(yc), meta
                else:
                    # background label = K
                    return x, torch.tensor(0), torch.tensor(self.K), meta
        tr = PartVsRest(ds_train, part_name, K_pos, ds_train.detector_clusters.get(part_name, {})) if use_background else PartOnly(ds_train, part_name)
        vl = (PartVsRest(ds_val,   part_name, K_pos, ds_train.detector_clusters.get(part_name, {})) if (ds_val is not None and use_background) else (PartOnly(ds_val,   part_name) if ds_val is not None else None))

        K_pos = len(ds_train.detector_clusters.get(part_name, {}))
        K = K_pos + 1 if use_background else K_pos
        if len(tr) == 0:
            print(f"[{part_name}] skipped (no training items).")
            continue

        # Maps and flags
        train_name2idx: Dict[str, int] = ds_train.detector_clusters.get(part_name, {}) or {}
        val_has_map = (ds_val is not None) and hasattr(ds_val, "detector_clusters") and (part_name in (ds_val.detector_clusters or {}))

        # Balanced sampler (one weight per element; invalid -> 0)
        labels_full = [int(tr[i][2]) for i in range(len(tr))]    # dataset tuple: (x, det_idx, cluster_idx, meta)
        valid_labels = [y for y in labels_full if y >= 0]
        assert len(valid_labels) > 0, f"[{part_name}] no valid training labels."

        counts = [0] * max(1, K)
        for y in valid_labels: counts[y] += 1
        inv = [1.0 / max(1, c) for c in counts]
        weights_full = [(inv[y] if y >= 0 else 0.0) for y in labels_full]
        num_pos = int(sum(w > 0 for w in weights_full))
        if num_pos == 0:
            raise ValueError(f"[{part_name}] no positives after filtering; check cluster mapping.")

        dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        if num_workers > 0:
            dl_kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

        sampler = WeightedRandomSampler(torch.tensor(weights_full, dtype=torch.double),
                                        num_samples=num_pos, replacement=True)
        tr_loader = DataLoader(tr, sampler=sampler, **dl_kwargs)
        vl_loader = DataLoader(vl, shuffle=False, **dl_kwargs) if (vl is not None and len(vl) > 0) else None

        # Model / opt
        model = PartDetectorTiny(
            emb_dim=emb_dim, num_classes=max(1, K),
            backbone=backbone, backbone_trainable=backbone_trainable, weights=weights
        ).to(device)
        _maybe_channels_last(model, channels_last)
        if use_compile and hasattr(torch, "compile") and not data_parallel:
            try: model = torch.compile(model, mode="max-autotune")
            except Exception: pass
        if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        opt = torch.optim.AdamW(
            _param_groups_adamw(model, wd=weight_decay, norm_wd=norm_weight_decay, bias_wd=bias_weight_decay),
            lr=lr
        )
        scheduler, _sch_kind = _build_scheduler(opt, scheduler_type, epochs, warmup_epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
        stopper = EarlyStopper(monitor="val_acc", patience=patience, min_delta=min_delta) if (early_stop and K >= 2) else None

        # CE weights (we’ll call F.cross_entropy in float32)
        cls_w = _class_weights(counts).to(device)
        label_smoothing = 0.1

        # Informative categories (optional, only if you provide the helper)
        has_label_fn = (label_fn is not None)
        informative_cats = None
        if (K >= 2) and has_label_fn and (vl is not None) and (val_metric == 'cluster_informative') \
           and ('_build_informative_categories_for_part' in globals()):
            try:
                informative_cats = globals()['_build_informative_categories_for_part'](
                    vl, label_fn, min_clusters=min_clusters_per_category, device=device
                )
                if len(informative_cats) == 0:
                    print(f"[{part_name}] no informative categories; fallback to raw accuracy.")
                    informative_cats = None
            except Exception as e:
                print(f"[{part_name}] informative-category build failed: {e}; fallback to raw accuracy.")
                informative_cats = None

        # One-class state (K==1)
        oc_mu = None; oc_var = None
        oc_mom = 0.05
        oc_queue = deque(maxlen=16)
        tau_std = 0.05
        w_center, w_var, w_cov = 1.0, 1.0, 0.005

        # Best tracking (K>=2)
        best_state, best_ep, best_va = None, -1, -1e9

        # --------------- epochs ---------------
        for ep in range(1, epochs + 1):
            model.train()
            tr_sum = 0.0; tr_acc_sum = 0.0; nb = 0
            tr_std_sum = 0.0; tr_std_batches = 0

            for i, (xb, _, yb, meta) in enumerate(tr_loader):
                m = (yb >= 0)
                if m.sum().item() == 0:
                    continue
                xb = _to_device_batch(xb[m], device, channels_last)
                yb = yb[m].to(device, non_blocking=True).long()

                opt.zero_grad(set_to_none=True)

                # Forward (AMP for forward only; losses in float32 where needed)
                use_amp_now = (scaler.is_enabled() and (K >= 2))
                with torch.cuda.amp.autocast(enabled=use_amp_now):
                    z, logits = model(xb)

                if K >= 2:
                    # CE in float32 (outside autocast) to avoid type mismatch with class weights
                    with torch.cuda.amp.autocast(enabled=False):
                        try:
                            loss = F.cross_entropy(
                                logits.float(), yb,
                                weight=cls_w.float(),
                                label_smoothing=label_smoothing
                            )
                        except TypeError:
                            loss = F.cross_entropy(
                                logits.float(), yb,
                                weight=cls_w.float()
                            )
                else:
                    # One-class objective in float32
                    z32 = F.normalize(z.float(), dim=1)
                    mu_b = z32.mean(0)
                    std_b = z32.std(0, unbiased=False)

                    if oc_mu is None:
                        mu_ref = mu_b.detach()
                        oc_mu  = mu_b.detach()
                        oc_var = (z32.var(0, unbiased=True) + 1e-6).detach()
                    else:
                        mu_ref = (torch.stack(list(oc_queue), 0).mean(0) if len(oc_queue) > 0 else oc_mu)

                    loss_center = (z32 - mu_ref.detach()).pow(2).mean()
                    tau_std = 0.8 * (1.0 / (z32.size(1) ** 0.5))
                    loss_var    = F.relu(tau_std - std_b).mean()
                    zc = z32 - z32.mean(0)
                    cov = (zc.T @ zc) / max(1, z32.size(0) - 1)
                    off = cov - torch.diag(torch.diag(cov))
                    loss_cov = off.pow(2).mean()
                    loss = w_center*loss_center + w_var*loss_var + w_cov*loss_cov

                # Step
                if scaler.is_enabled():
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()

                tr_sum += float(loss.item()); nb += 1

                if K >= 2:
                    tr_acc_sum += _acc(logits, yb)
                else:
                    with torch.no_grad():
                        oc_mu  = (1 - oc_mom)*oc_mu + oc_mom*mu_b.detach()
                        oc_var = (1 - oc_mom)*oc_var + oc_mom*(z32.var(0, unbiased=True) + 1e-6).detach()
                        oc_queue.append(mu_b.detach())
                        tr_std_sum += std_b.mean().item(); tr_std_batches += 1

                if (K == 1) and step_print_every and ((i + 1) % step_print_every == 0):
                    print(f"[{part_name}] step {i+1}: "
                          f"center={loss_center.item():.4e} var={loss_var.item():.4e} "
                          f"cov={loss_cov.item():.4e} std={std_b.mean().item():.3f}")

            tr_loss = tr_sum / max(1, nb)
            tr_acc_out = (tr_acc_sum / max(1, nb)) if K >= 2 else float("nan")
            s_train = (tr_std_sum / max(1, tr_std_batches)) if K == 1 else float("nan")

            # ---------- validation ----------
            if vl_loader is not None:
                model.eval()
                vl_sum = 0.0; nvb = 0
                val_std_sum = 0.0; val_std_batches = 0
                va_sum_raw = 0.0; nvb_acc = 0
                va_sum_bal = 0.0; nvb_bal = 0

                # For one-class metrics vs negatives
                pos_scores_list = []

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(scaler.is_enabled() and (K >= 2))):
                    for xb, _, yb_val, meta in vl_loader:
                        m0 = (yb_val >= 0)
                        if m0.sum().item() == 0:
                            continue
                        xb = _to_device_batch(xb[m0], device, channels_last)

                        if K >= 2:
                            # Remap validation labels: positives need mapping to train clusters; negatives -> background K
                            yb_map = yb_val[m0].clone()
                            if use_background:
                                # Identify which samples in this batch actually belong to the current part
                                idxs = torch.nonzero(m0, as_tuple=False).view(-1).tolist()
                                pos_mask_list = []
                                for j, ii in enumerate(idxs):
                                    m_j = meta[ii]
                                    pcanon = getattr(m_j, "part_canonical", None)
                                    pos_mask_list.append(pcanon == part_name)
                                import numpy as _np
                                pos_mask = torch.tensor(_np.array(pos_mask_list), dtype=torch.bool, device=yb_map.device)
                                # Map positives to train cluster index if needed
                                if val_has_map and pos_mask.any():
                                    yb_map[pos_mask] = _remap_val_indices_by_name(part_name, yb_map[pos_mask], ds_val, train_name2idx)
                                # Set negatives to background class index = K-1 if their canonical part != this part
                                if (~pos_mask).any():
                                    yb_map[~pos_mask] = K - 1
                            else:
                                if val_has_map:
                                    yb_map = _remap_val_indices_by_name(part_name, yb_map, ds_val, train_name2idx)
                                else:
                                    yb_map = yb_map.to(device, non_blocking=True)
                            m = (yb_map >= 0)
                            if m.sum().item() == 0:
                                continue
                            xb = xb[m]; yb = yb_map[m].to(device, non_blocking=True).long()

                            _, logits = model(xb)
                            with torch.cuda.amp.autocast(enabled=False):
                                try:
                                    vloss = F.cross_entropy(logits.float(), yb, weight=cls_w.float(), label_smoothing=label_smoothing)
                                except TypeError:
                                    vloss = F.cross_entropy(logits.float(), yb, weight=cls_w.float())
                            vl_sum += float(vloss.item()); nvb += 1

                            va_sum_raw += _acc(logits, yb); nvb_acc += 1
                            ba = _balanced_accuracy(logits, yb, K)
                            if math.isfinite(ba):
                                va_sum_bal += ba; nvb_bal += 1
                        else:
                            z, _ = model(xb)
                            z32 = F.normalize(z.float(), dim=1)
                            mu_ref = oc_mu if oc_mu is not None else z32.mean(0)
                            std_b  = z32.std(0, unbiased=False)
                            v_center = (z32 - mu_ref.detach()).pow(2).mean()
                            v_var    = F.relu(tau_std - std_b).mean()
                            zc = z32 - z32.mean(0)
                            cov = (zc.T @ zc) / max(1, z32.size(0) - 1)
                            off = cov - torch.diag(torch.diag(cov))
                            v_cov = off.pow(2).mean()
                            vloss = w_center*v_center + w_var*v_var + w_cov*v_cov
                            vl_sum += float(vloss.item()); nvb += 1
                            val_std_sum += std_b.mean().item(); val_std_batches += 1

                            if (oc_mu is not None) and (oc_var is not None):
                                pos_scores_list.append(_oc_scores(z32, oc_mu, oc_var).detach().cpu())

                val_loss = vl_sum / max(1, nvb)

                if K >= 2:
                    val_acc_raw = va_sum_raw / max(1, nvb_acc) if nvb_acc > 0 else float('nan')
                    val_acc_bal = va_sum_bal / max(1, nvb_bal) if nvb_bal > 0 else float('nan')
                    use_balanced = (val_metric in ('cluster_balanced', 'cluster_bal', 'balanced'))
                    val_acc_for_es = (val_acc_bal if use_balanced and math.isfinite(val_acc_bal) else val_acc_raw)

                    scheduler.step(val_loss) if _sch_kind == 'plateau' else scheduler.step()
                    if val_acc_for_es > best_va + min_delta:
                        best_va, best_ep = val_acc_for_es, ep
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                    if (ep % max(1, log_every)) == 0:
                        print(f"[{part_name}] ep {ep:02d}/{epochs}  "
                              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc_out:.3f}  "
                              f"val_loss={val_loss:.4f} val_acc_raw={val_acc_raw:.3f} val_acc_bal={val_acc_bal:.3f}  "
                              f"lr={opt.param_groups[0]['lr']:.2e}")

                    if stopper and stopper.step(val_acc_for_es):
                        print(f"[{part_name}] early stop at epoch {ep} (best ep {best_ep} metric={best_va:.3f})")
                        break

                else:
                    # Build a small negative set from other parts for binary OC metrics
                    val_oc_auc = float('nan'); val_oc_bacc = float('nan')
                    if (ds_val is not None) and (len(pos_scores_list) > 0) and (oc_mu is not None) and (oc_var is not None):
                        neg_vl = NotPartVal(ds_val, part_name)
                        if len(neg_vl) > 0:
                            neg_loader = DataLoader(
                                neg_vl, batch_size=batch_size, shuffle=True,
                                num_workers=max(0, num_workers // 2), pin_memory=(device.type == "cuda")
                            )
                            pos_scores = torch.cat(pos_scores_list, 0).float()
                            target_neg = min(int(pos_scores.numel() * 2), 10000)
                            neg_scores_list = []
                            seen = 0
                            with torch.no_grad():
                                for xb_neg, _, yb_neg, _ in neg_loader:
                                    mneg = (yb_neg >= 0)
                                    if mneg.sum().item() == 0: continue
                                    xb_neg = _to_device_batch(xb_neg[mneg], device, channels_last)
                                    z_neg, _ = model(xb_neg)
                                    z_neg = F.normalize(z_neg.float(), dim=1)
                                    neg_scores_list.append(_oc_scores(z_neg, oc_mu, oc_var).detach().cpu())
                                    seen += neg_scores_list[-1].numel()
                                    if seen >= target_neg:
                                        break
                            if len(neg_scores_list) > 0:
                                neg_scores = torch.cat(neg_scores_list, 0).float()
                                val_oc_auc, val_oc_bacc = _roc_auc_and_best_balanced(pos_scores, neg_scores)

                    s_val = (val_std_sum / max(1, val_std_batches)) if val_std_batches > 0 else float('nan')
                    scheduler.step(val_loss) if _sch_kind == 'plateau' else scheduler.step()
                    if (ep % max(1, log_every)) == 0:
                        oc_auc_str  = f"{val_oc_auc:.3f}"  if math.isfinite(val_oc_auc)  else "nan"
                        oc_bacc_str = f"{val_oc_bacc:.3f}" if math.isfinite(val_oc_bacc) else "nan"
                        if math.isfinite(s_val):
                            print(f"[{part_name}] ep {ep:02d}/{epochs}  s={s_train:.4f}  "
                                  f"tr_loss={tr_loss:.4f} tr_acc=N/A  val_loss={val_loss:.4f} (s_val={s_val:.4f})  "
                                  f"val_oc_auc={oc_auc_str} val_oc_bacc={oc_bacc_str}  "
                                  f"lr={opt.param_groups[0]['lr']:.2e}")
                        else:
                            print(f"[{part_name}] ep {ep:02d}/{epochs}  s={s_train:.4f}  "
                                  f"tr_loss={tr_loss:.4f} tr_acc=N/A  val_loss={val_loss:.4f}  "
                                  f"val_oc_auc={oc_auc_str} val_oc_bacc={oc_bacc_str}  "
                                  f"lr={opt.param_groups[0]['lr']:.2e}")
            # end epochs

        # Restore best (K>=2)
        if (K >= 2) and (best_state is not None):
            target = model.module if isinstance(model, nn.DataParallel) else model
            target.load_state_dict(best_state, strict=True)

        # Final center/var on TRAIN (normalized embedding)
        with torch.no_grad():
            dl_stats = DataLoader(tr, batch_size=max(256, batch_size//2), shuffle=False,
                                  num_workers=max(0, num_workers//2), pin_memory=(device.type == "cuda"))
            Zs = []
            for xb, _, yb, _ in dl_stats:
                m = (yb >= 0)
                if m.sum().item() == 0: continue
                xb = _to_device_batch(xb[m], device, channels_last)
                z, _ = model(xb)
                Zs.append(F.normalize(z.float(), dim=1).detach().cpu())
            if len(Zs) > 0:
                Z = torch.cat(Zs, 0)
                center = Z.mean(0); var = Z.var(0, unbiased=True).clamp_min(1e-6)
            else:
                center = torch.zeros(emb_dim); var = torch.ones(emb_dim)

        priors = None
        if K >= 2:
            tot = float(sum(counts)) if sum(counts) > 0 else 1.0
            priors = [c / tot for c in counts]

        inv_clusters = {int(v): str(k) for k, v in (ds_train.detector_clusters.get(part_name, {}) or {}).items()}
        state = dict(
            part=part_name,
            emb_dim=emb_dim,
            num_classes=max(1, K),
            background_index=(K-1 if use_background else None),
            inv_clusters=inv_clusters,
            priors=(priors if priors is not None else None),
            state_dict=(model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
            backbone=backbone,
            center=center.tolist(),
            var=var.tolist(),
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
        self.priors: List[Optional[torch.Tensor]] = []
        self.part_names_in_order: List[str] = []
        self.backbones: List[str] = []
        self.centers: List[Optional[torch.Tensor]] = []
        self.vars: List[Optional[torch.Tensor]] = []
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
            pri = state.get("priors", None)
            self.priors.append(torch.tensor(pri, dtype=torch.float32, device=self.device) if pri is not None else None)
            self.part_names_in_order.append(state["part"])
            self.backbones.append(state.get("backbone","tinycnn"))

            # load one-class stats (always present in new checkpoints)
            ctr = state.get("center", None)
            var = state.get("var", None)
            self.centers.append(torch.tensor(ctr, dtype=torch.float32, device=self.device) if ctr is not None else None)
            self.vars.append(torch.tensor(var, dtype=torch.float32, device=self.device) if var is not None else None)

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


# ---- AlexNet convenience wrapper (now forwards val_metric) ----
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
    # regularization & scheduler
    weight_decay: float = 1e-4, norm_weight_decay: float = 0.0, bias_weight_decay: float = 0.0,
    scheduler_type: str = 'plateau', warmup_epochs: int = 0,
    # metrics & logging
    val_metric: str = "cluster",                 # <-- NEW
    label_fn=None, min_clusters_per_category: int = 2, log_every: int = 1,
):
    """
    Convenience wrapper to train per-part detectors with **AlexNet** backbone.
    Accepts `val_metric`: 'cluster' (raw acc), 'cluster_balanced' (macro acc),
    or 'cluster_informative' (subset acc).
    """
    return train_small_detectors(
        ds_train=ds_train, ds_val=ds_val, outdir=outdir, emb_dim=emb_dim,
        epochs=epochs, batch_size=batch_size, lr=lr,
        backbone="alexnet", backbone_trainable=backbone_trainable, weights=weights,
        use_amp=use_amp, device=device, channels_last=channels_last, use_tf32=use_tf32,
        num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
        weight_decay=weight_decay, norm_weight_decay=norm_weight_decay, bias_weight_decay=bias_weight_decay,
        scheduler_type=scheduler_type, warmup_epochs=warmup_epochs,
        # forward the metric controls
        val_metric=val_metric,                     # <-- NEW
        label_fn=label_fn, min_clusters_per_category=min_clusters_per_category,
        log_every=log_every,
    )

