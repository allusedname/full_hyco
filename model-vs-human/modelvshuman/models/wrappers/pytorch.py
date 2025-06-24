import math
import PIL
import clip
import numpy as np
import torch.nn as nn
import torch
from PIL.Image import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from tqdm import tqdm

import hycoclip.utils.distributed as dist
import hycoclip.lorentz as L

from .base import AbstractModel
from ..pytorch.clip.imagenet_classes import imagenet_classes
from ..pytorch.clip.imagenet_templates import imagenet_templates


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def undo_default_preprocessing(images):
    """Convenience function: undo standard preprocessing."""
    assert isinstance(images, torch.Tensor)
    
    # ====================================================================
    # THE FIX for the device mismatch RuntimeError
    # Get the device from the input tensor itself for robust operation
    tensor_device = images.device
    default_mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_device)
    default_std = torch.tensor([0.229, 0.224, 0.225], device=tensor_device)
    # ====================================================================

    # Reshape mean and std to be broadcastable
    images = images * default_std.view(1, -1, 1, 1)
    images = images + default_mean.view(1, -1, 1, 1)
    return images

class PytorchModel(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args
        self.model.to(device())

    def to_numpy(self, x):
        if x.is_cuda:
            return x.detach().cpu().numpy()
        else:
            return x.numpy()

    def softmax(self, logits):
        assert type(logits) is np.ndarray

        softmax_op = torch.nn.Softmax(dim=1)
        softmax_output = softmax_op(torch.Tensor(logits))
        return self.to_numpy(softmax_output)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        self.model.eval()
        logits = self.model(images)
        return self.to_numpy(logits)


class PyContrastPytorchModel(PytorchModel):
    """
    This class inherits PytorchModel class to adapt model validation for Pycontrast pre-trained models from
    https://github.com/HobbitLong/PyContrast
    """

    def __init__(self, model, classifier, model_name, *args):
        super(PyContrastPytorchModel, self).__init__(model, model_name, args)
        self.classifier = classifier
        self.classifier.to(device())

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()
        self.classifier.eval()
        feat = self.model(images, mode=2)
        output = self.classifier(feat)
        return self.to_numpy(output)


class ViTPytorchModel(PytorchModel):

    def __init__(self, model, model_name, img_size=(384, 384), *args):
        self.img_size = img_size
        super(ViTPytorchModel, self).__init__(model, model_name, args)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        logits = self.model(images)
        return self.to_numpy(logits)

    def preprocess(self):
        # custom preprocessing from:
        # https://github.com/lukemelas/PyTorch-Pretrained-ViT

        return Compose([
            Resize(self.img_size),
            ToTensor(),
            Normalize(0.5, 0.5),
        ])


class ClipPytorchModel(PytorchModel):
    def __init__(self, model, model_name, *args):
        super(ClipPytorchModel, self).__init__(model, model_name, *args)
        # Precompute zero-shot weights
        self.zeroshot_weights = self._get_zeroshot_weights(imagenet_classes, imagenet_templates)

    def _get_zeroshot_weights(self, class_names, templates):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(class_names):
                texts = [template.format(class_name) for template in templates]
                tokenized = clip.tokenize(texts).to(device)
                embeddings = self.model.encode_text(tokenized)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                class_emb = embeddings.mean(dim=0)
                class_emb /= class_emb.norm()
                zeroshot_weights.append(class_emb)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights

    def preprocess(self):
        n_px = self.model.visual.input_resolution
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward_batch(self, images):
        # images: Tensor[C,H,W] normalized by default loader
        assert isinstance(images, torch.Tensor)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Undo default preprocessing
        images = undo_default_preprocessing(images)
        # Apply CLIP preprocess
        proc = self.preprocess()
        imgs = [proc(ToPILImage()(img)) for img in images]
        batch = torch.stack(imgs, axis=0).to(device)

        # Debug pixel patch
        patch = batch[0, :, 100:110, 100:110]
        print(f"[CLIP DEBUG] patch mean={patch.mean().item():.4f}, std={patch.std().item():.4f}")

        # Encode images
        self.model.eval()
        image_features = self.model.encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Debug feature samples
        for i in range(min(3, image_features.size(0))):
            print(f"[CLIP DEBUG] clip_feats[{i}][:10]={image_features[i, :10].cpu().numpy()}")
        print(f"[CLIP DEBUG] clip_feats mean={image_features.mean().item():.4f}, std={image_features.std().item():.4f}")

        # Sanity test: white vs black
        white = torch.ones_like(batch[0:1]).to(device)
        black = torch.zeros_like(batch[0:1]).to(device)
        w_feat = self.model.encode_image(white)
        w_feat /= w_feat.norm(dim=-1, keepdim=True)
        b_feat = self.model.encode_image(black)
        b_feat /= b_feat.norm(dim=-1, keepdim=True)
        wb_dist = (w_feat - b_feat).norm(dim=-1)
        print(f"[CLIP DEBUG] white-black dist={wb_dist.item():.4f}")

        # Compute logits and debug
        logits = 100. * image_features @ self.zeroshot_weights
        print(f"[CLIP DEBUG] logits mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

        return self.to_numpy(logits)


class EfficientNetPytorchModel(PytorchModel):

    def __init__(self, model, model_name, *args):
        super(EfficientNetPytorchModel, self).__init__(model, model_name, *args)

    def preprocess(self):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        img_size = 475
        crop_pct = 0.936
        scale_size = int(math.floor(img_size / crop_pct)) 
        return Compose([
            Resize(scale_size, interpolation=PIL.Image.BICUBIC),
            CenterCrop(img_size),
            ToTensor(),
            normalize,
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        logits = self.model(images)
        return self.to_numpy(logits)


class SwagPytorchModel(PytorchModel):

    def __init__(self, model, model_name, input_size, *args):
        super(SwagPytorchModel, self).__init__(model, model_name, *args)
        self.input_size = input_size

    def preprocess(self):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

        return Compose([
            Resize(self.input_size, interpolation=PIL.Image.BICUBIC),
            CenterCrop(self.input_size),
            ToTensor(),
            normalize,
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()
        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())
        logits = self.model(images)
        return self.to_numpy(logits)    

import os
import timm
from tqdm import tqdm
from typing import Sequence
from PIL import Image
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
)
from pathlib import Path

from hycoclip.models                  import HyCoCLIP
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders  import TransformerTextEncoder
from hycoclip.tokenizer import Tokenizer

CHECKPOINT_PATH = "/home/xjzb2/compo_learning/model-vs-human/modelvshuman/models/hycoclip_vit_s.pth"


from hycoclip.tokenizer import Tokenizer

def final_robust_encode_text(self, tokens: list[torch.Tensor], project: bool):
    """
    A robust version of encode_text that explicitly finds the last non-padding
    token and fixes the textual_proj layer's dimension bug.
    This function will be monkey-patched onto the model instance.
    """
    context_length = self.textual.context_length
    for i, inst_tokens in enumerate(tokens):
        if len(inst_tokens) > context_length:
            eot_token = inst_tokens[-1:]
            inst_tokens = torch.cat([
                inst_tokens[:context_length - 1],
                eot_token
            ])
            tokens[i] = inst_tokens

    tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    tokens_padded = tokens_padded.to(self.device)

    text_feats = self.textual(tokens_padded)

    eos_indices = torch.count_nonzero(tokens_padded, dim=1) - 1
    
    batch_idxs = torch.arange(text_feats.shape[0], device=self.device)
    
    text_feats = text_feats[batch_idxs, eos_indices]
    
    text_feats = self.textual_proj(text_feats)

    if text_feats.shape[-1] != self.embed_dim:
        text_feats = text_feats[:, :self.embed_dim]

    if hasattr(self, 'textual_alpha') and project:
        text_feats = text_feats * self.textual_alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
             text_feats = L.exp_map0(text_feats, self.curv.exp())
    elif project:
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)

    return text_feats

class HyCoCLIPModel(PytorchModel):
    def __init__(
        self,
        model_name: str,
        *args,
        arch: str = "vit_small_patch16_224"
    ):
        # Build vision and text backbones
        visual = build_timm_vit(arch=arch, global_pool="token", grad_checkpointing=False)
        textual = TransformerTextEncoder(arch="L12_W512_A8", vocab_size=49408, context_length=77)
        hyco = HyCoCLIP(
            visual=visual,
            textual=textual,
            embed_dim=512,
            curv_init=1.0,
            learn_curv=True,
            entail_weight=0.0,
            use_boxes=True,
            pixel_mean=(0.485, 0.456, 0.406),
            pixel_std=(0.229, 0.224, 0.225),
        )
        # Load weights
        base_models_dir = Path(__file__).resolve().parent.parent
        ckpt_path = base_models_dir / "hycoclip_vit_s.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        hyco.load_state_dict(state, strict=True)

        # Swap in the robust text encoder method
        import types
        hyco.encode_text = types.MethodType(final_robust_encode_text, hyco)

        # Initialize the PytorchModel wrapper
        super().__init__(hyco, model_name, hyco.embed_dim, hyco.pixel_mean, hyco.pixel_std)
        self.tokenizer = Tokenizer()

        # Prepare or load cached zero-shot weights
        project_root = Path(__file__).resolve().parents[4]
        cache_file = project_root / "hycoclip_zeroshot_weights.pt"
        if cache_file.exists():
            zs = torch.load(cache_file)
        else:
            zs = self._get_zeroshot_weights(imagenet_classes, imagenet_templates)
            torch.save(zs.cpu(), cache_file)
        self.zeroshot_weights = zs.to(device())
        hyco.to(device())

    def _get_zeroshot_weights(self, class_names: Sequence[str], templates: Sequence[str]) -> torch.Tensor:
        hyco = self.model
        dev = device()
        with torch.no_grad():
            ws = []
            for cls in tqdm(class_names, desc="HyCoCLIP zeroshot"):
                prompts = [t.format(cls) for t in templates]
                toks = self.tokenizer(prompts)
                # Euclidean pooling
                eu_feats = hyco.encode_text(toks, project=False)
                mean_eu = eu_feats.mean(dim=0, keepdim=True)
                # Hyperbolic map
                hyper_feat = mean_eu * hyco.textual_alpha.exp()
                with torch.autocast(dev.type, dtype=torch.float32):
                    hyp_feat = L.exp_map0(hyper_feat.float(), hyco.curv.exp()).squeeze(0)
                ws.append(hyp_feat)
            zeroshot_weights = torch.stack(ws, dim=1).to(dev)
        return zeroshot_weights

    def forward_batch(self, images: torch.Tensor) -> np.ndarray:
        hyco = self.model
        hyco.eval()
        with torch.no_grad():
            # Undo default loader preprocessing
            images_unnorm = undo_default_preprocessing(images)
            # Encode into hyperbolic space
            img_feats = hyco.encode_image(images_unnorm, project=True)
            # Compute logits via hyperbolic distance
            curv = hyco.curv.exp()
            logits = -L.pairwise_dist(img_feats, self.zeroshot_weights.T, curv)
            logits = hyco.logit_scale.exp() * logits
            return logits.detach().cpu().numpy()
        
    def preprocess(self):
        pass




        
        #python -m modelvshuman --models hycoclip --datasets cue-conflict --batch-size 64 --num-workers 16