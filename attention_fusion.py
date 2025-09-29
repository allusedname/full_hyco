
# -*- coding: utf-8 -*-
"""
attention_fusion.py

Adds attention-based fusion for per-part embeddings produced by an ensemble
(PartEnsembleTiny / PartEnsembleShared).

Two options:
  1) ConfidenceAttentionPool (parameter-free): uses attention weights computed
     from per-part confidence (e.g., max softmax prob, entropy) and optional priors.
     -> Drop-in replacement for your current "last fusion" weighted-sum.

  2) TransformerAttentionFusion (trainable): a light Transformer pooling head
     with a CLS token across parts, producing a fused image embedding.
     -> Use with the provided training helper to learn the fusion end-to-end.
"""
from typing import List, Optional, Dict, Tuple, Callable, Any, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def _softmax_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0: temperature = 1.0
    return F.softmax(logits / temperature, dim=-1)

def _conf_from_logits(logits: torch.Tensor, method: str = "max", temperature: float = 1.0) -> torch.Tensor:
    probs = _softmax_logits(logits, temperature=temperature)  # (B,K)
    if method == "max":
        conf = probs.max(dim=-1).values
    elif method in ("neg_entropy", "1-entropy"):
        ent = -(probs * torch.clamp(probs, min=1e-9).log()).sum(dim=-1)
        K = probs.shape[-1]; ent = ent / math.log(max(2, K))
        conf = 1.0 - ent
    else:
        raise ValueError(f"Unknown confidence method: {method}")
    return conf

def _apply_priors_to_conf(conf: torch.Tensor, logits: torch.Tensor, priors_for_part: Optional[torch.Tensor] = None) -> torch.Tensor:
    if priors_for_part is None: return conf
    with torch.no_grad():
        top = logits.argmax(dim=-1)
        prior_mult = priors_for_part[top]
    return conf * prior_mult

class ConfidenceAttentionPool(nn.Module):
    def __init__(self, method: str = "max", temperature: float = 1.0, eps: float = 1e-8):
        super().__init__(); self.method=method; self.temperature=temperature; self.eps=eps

    @torch.no_grad()
    def forward(self, zs_list: List[torch.Tensor], logits_list: List[torch.Tensor], priors: Optional[List[torch.Tensor]] = None):
        B, E = zs_list[0].shape; P = len(zs_list)
        confs = []
        for p in range(P):
            logits = logits_list[p]
            conf = _conf_from_logits(logits, method=self.method, temperature=self.temperature)
            if priors is not None and priors[p] is not None:
                pr = priors[p].to(logits.device, non_blocking=True)
                conf = _apply_priors_to_conf(conf, logits, pr)
            confs.append(conf.unsqueeze(-1))
        conf_mat = torch.cat(confs, dim=-1)         # (B,P)
        attn = torch.softmax(conf_mat, dim=-1)      # (B,P)
        Z = torch.stack(zs_list, dim=1)             # (B,P,E)
        fused = torch.bmm(attn.unsqueeze(1), Z).squeeze(1)  # (B,E)
        return fused, attn

class TransformerAttentionFusion(nn.Module):
    def __init__(self, num_parts: int, d_in: int, d_model: int = 256, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1, use_confidence: bool = True):
        super().__init__()
        self.num_parts=num_parts; self.d_in=d_in; self.d_model=d_model; self.use_confidence=use_confidence
        add = 1 if use_confidence else 0
        self.proj = nn.Sequential(nn.Linear(d_in+add, d_model), nn.LayerNorm(d_model))
        self.part_embed = nn.Embedding(num_parts, d_model)
        self.cls = nn.Parameter(torch.zeros(1,1,d_model)); nn.init.trunc_normal_(self.cls, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, zs_list: List[torch.Tensor], logits_list: Optional[List[torch.Tensor]] = None, priors: Optional[List[torch.Tensor]] = None, conf_method: str = "max", conf_temperature: float = 1.0):
        B, E = zs_list[0].shape; P = len(zs_list)
        if self.use_confidence and logits_list is not None:
            confs = []
            for p in range(P):
                conf = _conf_from_logits(logits_list[p], method=conf_method, temperature=conf_temperature)
                if priors is not None and priors[p] is not None:
                    pr = priors[p].to(logits_list[p].device, non_blocking=True)
                    conf = _apply_priors_to_conf(conf, logits_list[p], pr)
                confs.append(conf.unsqueeze(-1))
            conf_mat = torch.cat(confs, dim=-1)
        else:
            conf_mat = zs_list[0].new_ones(B, P)
        attn_weights = torch.softmax(conf_mat, dim=-1)
        Z = torch.stack(zs_list, dim=1)
        if self.use_confidence and logits_list is not None:
            Z = torch.cat([Z, attn_weights.unsqueeze(-1)], dim=-1)
        tokens = self.proj(Z) + self.part_embed.weight.unsqueeze(0)
        cls = self.cls.expand(B,1,self.d_model)
        X = torch.cat([cls, tokens], dim=1)
        X = self.dropout(X)
        H = self.enc(X)
        fused = H[:,0,:]
        return fused, attn_weights

@torch.no_grad()
def fuse_image_embeddings_with_attention(ds, ensemble, batch_size: int = 256, num_workers: int = 4, device: Optional[Union[str, torch.device]] = None, method: str = "max", temperature: float = 1.0, channels_last: bool = True):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(dev.type=="cuda"))
    X_all, A_all, ids = [], [], []
    ensemble_device = getattr(ensemble, "device", dev)
    for batch in dl:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        meta = batch[3] if (isinstance(batch, (list, tuple)) and len(batch)>3) else None
        xb = xb.to(ensemble_device, non_blocking=True)
        if channels_last and xb.ndim==4: xb = xb.to(memory_format=torch.channels_last)
        zs_list, logits_list = ensemble.score_all(xb)
        pool = ConfidenceAttentionPool(method=method, temperature=temperature)
        fused, attn = pool(zs_list, logits_list, priors=getattr(ensemble, "priors", None))
        X_all.append(fused.detach().cpu()); A_all.append(attn.detach().cpu())
        if meta is not None and isinstance(meta, (list, tuple)):
            ids.extend([ (m.get("id") if isinstance(m, dict) and "id" in m else None) for m in meta ])
        else:
            start = len(ids); ids.extend(list(range(start, start+fused.shape[0])))
    X = torch.cat(X_all, dim=0).contiguous(); A = torch.cat(A_all, dim=0).contiguous()
    return X, ids, A
