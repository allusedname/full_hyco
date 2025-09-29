
# -*- coding: utf-8 -*-
"""
metrics_and_viz.py : evaluation + visualization helpers.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import math, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def _device(dev=None): return torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
def _to_channels_last(x, enable): 
    return x.to(memory_format=torch.channels_last) if (enable and isinstance(x, torch.Tensor) and x.ndim==4) else x

@torch.no_grad()
def routing_winner_histogram(ds, ensemble, batch_size: int = 256, num_workers: int = 4, device=None, channels_last: bool=True, mode: str="attention", conf_method: str="max", temperature: float=1.0):
    dev = _device(device)
    parts = getattr(ensemble, "part_names_in_order", None); P = len(parts)
    priors = getattr(ensemble, "priors", None)
    counts = np.zeros(P, dtype=np.int64); total=0
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(dev.type=="cuda"))
    def _softmax(lg): return F.softmax(lg/temperature, dim=-1)
    for batch in dl:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = _to_channels_last(xb.to(dev, non_blocking=True), channels_last)
        zs_list, logits_list = ensemble.score_all(xb)
        scores = []
        for p in range(P):
            probs = _softmax(logits_list[p])
            if mode=="attention":
                if conf_method=="max":
                    sc = probs.max(-1).values
                else:
                    ent = -(probs*torch.clamp(probs,min=1e-9).log()).sum(-1)
                    sc = 1.0 - ent/ math.log(max(2, probs.shape[-1]))
            else:
                sc = probs.max(-1).values
            if priors is not None and priors[p] is not None:
                pr = priors[p].to(probs.device, non_blocking=True)
                top = logits_list[p].argmax(-1)
                sc = sc * pr[top]
            scores.append(sc.unsqueeze(-1))
        S = torch.cat(scores, dim=-1)
        win = S.argmax(-1).detach().cpu().numpy()
        for idx in win.tolist(): counts[int(idx)] += 1
        total += len(win)
    return {"counts": counts, "parts_order": parts, "total": int(total)}

def plot_routing_winner_histogram(result: Dict[str, Any], title: str="Routing winner histogram"):
    import matplotlib.pyplot as plt
    counts = result["counts"]; parts = result["parts_order"]; xs = list(range(len(parts)))
    plt.figure(); plt.bar(xs, counts); plt.xticks(xs, parts, rotation=45, ha='right')
    plt.title(title); plt.xlabel("Part"); plt.ylabel("Count"); plt.tight_layout(); plt.show()
