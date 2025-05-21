import math
import PIL
import clip
import os
import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from tqdm import tqdm

import torch.nn.functional as F
from hycoclip.models import HyCoCLIP
from hycoclip.encoders.image_encoders import vit_small_mocov3_patch16_224
from hycoclip.encoders.text_encoders import TransformerTextEncoder

from .base import AbstractModel
from ..pytorch.clip.imagenet_classes import imagenet_classes
from ..pytorch.clip.imagenet_templates import imagenet_templates


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def undo_default_preprocessing(images):
    """Convenience function: undo standard preprocessing."""

    assert type(images) is torch.Tensor
    default_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device())
    default_std = torch.Tensor([0.229, 0.224, 0.225]).to(device())

    images *= default_std[None, :, None, None]
    images += default_mean[None, :, None, None]

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
        self.zeroshot_weights=self._get_zeroshot_weights(imagenet_classes, imagenet_templates)
        
    def _get_zeroshot_weights(self, class_names, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(class_names):
                texts = [template.format(class_name) for template in templates]  # format with class
                texts = clip.tokenize(texts).to(device())  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device())

        return zeroshot_weights

    def preprocess(self):
        n_px = self.model.visual.input_resolution
        return Compose([
            Resize(n_px, interpolation=PIL.Image.BICUBIC),
            CenterCrop(n_px),
            # lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        self.model.eval()
        
        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ self.zeroshot_weights
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

from hycoclip.models import HyCoCLIP, CLIPBaseline
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders  import TransformerTextEncoder
from modelvshuman.models.registry import register_model
from modelvshuman.models.wrappers.base import AbstractModel


@register_model("pytorch")
class HyCoCLIPPytorchModel(CLIPBaseline):
    def __init__(self, model_name: str, *args, **kwargs):
        # ────────────────────────────────────────────────────────────
        # load CLIP’s full 1000-class ImageNet zero-shot weights
        from modelvshuman.utils import load_model
        clip_wrapper, _ = load_model("clip", "imagenet")
        # ClipPytorchModel stashes them as `zeroshot_weights` (1000×D)
        self.class_embeddings = clip_wrapper.zeroshot_weights

        visual  = build_timm_vit(
            arch="vit_base_patch16_224",
            global_pool="token",
            grad_checkpointing=False
        )

        # Build the text encoder exactly as in HyCoCLIP
        # correct text call: (arch, vocab_size, context_length)
        textual = TransformerTextEncoder(
            arch = "L12_W512_A8",
            vocab_size = 49408,
            context_length = 77,
        )

        # Instantiate the hyperbolic CLIP model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            visual, 
            textual, 
            embed_dim=512, 
            pixel_mean=(0.485,0.456,0.406),
            pixel_std=(0.229,0.224,0.225),
        )

        # ────────────────────────────────────────────────────────────
        self.to(device)
        self.model_name = model_name

    def forward_batch(self, images: torch.Tensor, *args):
        device = next(self.parameters()).device
        images = images.to(device)
        emb = self.encode_image(images, project=True)  # e.g. float32 or float16

        # ensure weights have same dtype as emb
        weights = self.class_embeddings.to(device=device, dtype=emb.dtype)  # now same type
        weights_t = weights                                             # (D, 1000)

        logits = emb @ weights_t                                           # (B, 1000)
        return logits.detach().cpu().numpy()

    def softmax(self, logits: np.ndarray):
        """
        Must turn raw logits into probabilities (same shape).
        """

        device = next(self.parameters()).device
        if isinstance(logits, torch.Tensor):
            t = logits.to(device)
        else:
            t = torch.from_numpy(logits).to(device)

        # Apply PyTorch softmax and return Tensor
        return F.softmax(t, dim=-1)
    
    #accuracy (top-1): 7.66

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

    assert type(images) is torch.Tensor
    default_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device())
    default_std = torch.Tensor([0.229, 0.224, 0.225]).to(device())

    images *= default_std[None, :, None, None]
    images += default_mean[None, :, None, None]

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
        self.zeroshot_weights=self._get_zeroshot_weights(imagenet_classes, imagenet_templates)
        
    def _get_zeroshot_weights(self, class_names, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(class_names):
                texts = [template.format(class_name) for template in templates]  # format with class
                texts = clip.tokenize(texts).to(device())  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device())

        return zeroshot_weights

    def preprocess(self):
        n_px = self.model.visual.input_resolution
        return Compose([
            Resize(n_px, interpolation=PIL.Image.BICUBIC),
            CenterCrop(n_px),
            # lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        self.model.eval()
        
        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ self.zeroshot_weights
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
import torch
import clip
import numpy as np
from tqdm import tqdm
from typing import Sequence
from PIL import Image
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
)

from hycoclip.models                  import HyCoCLIP
import torch.distributed as tdist
import hycoclip.utils.distributed as dist 
from hycoclip.lorentz                 import pairwise_dist
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders  import TransformerTextEncoder

class HyCoCLIPModel(PytorchModel):
    def __init__(
        self,
        model_name: str,
        *args,
        arch: str = "vit_small_patch16_224"
    ):
        """
        Wraps the official HyCoCLIP model for zero‐shot evaluation.
        """
        # 1) build the exact HyCoCLIP architecture (small vs base by arch)
        visual = build_timm_vit(
            arch=arch,
            global_pool="token",
            grad_checkpointing=False,
        )
        textual = TransformerTextEncoder(
            arch="L12_W512_A8",
            vocab_size=49408,
            context_length=77,
        )
        hyco = HyCoCLIP(
            visual=visual,
            textual=textual,
            embed_dim=512,
            curv_init=1.0,
            learn_curv=True,
            entail_weight=0.0,
            use_boxes = True,
            pixel_mean=(0.485, 0.456, 0.406),
            pixel_std=(0.229, 0.224, 0.225),
        )

        # 2) retain checkpoint path so factory can load it
        checkpoint_path=r"C:\Users\xjzb2\compo_learning\model-vs-human\modelvshuman\models\hycoclip_vit_s.pth"
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        hyco.load_state_dict(state, strict=False)

        # 3) delegate to base PytorchModel (handles preprocess + to_numpy)
        super().__init__(
            hyco,
            model_name,
            hyco.embed_dim,
            hyco.pixel_mean,
            hyco.pixel_std,
        )

        hyco.to(device)
        with torch.no_grad():
            zs = self._get_zeroshot_weights(imagenet_classes, imagenet_templates)
            # bake in the model’s learned temperature
            zs = zs * hyco.logit_scale.exp().to(zs.device)
        self.zeroshot_weights = zs
        


    def _get_zeroshot_weights(
        self,
        class_names: Sequence[str],
        templates: Sequence[str]
    ) -> torch.Tensor:
        hyco = self.model
        dev  = device()

        with torch.no_grad():
            ws = []
            for cls in tqdm(class_names, desc="HyCoCLIP zeroshot"):
                prompts = [t.format(cls) for t in templates]
                toks    = clip.tokenize(prompts, context_length=77).to(dev)

                # clamp hyperbolic params
                hyco.curv.data          = torch.clamp(hyco.curv.data, **hyco._curv_minmax)
                hyco.visual_alpha.data  = torch.clamp(hyco.visual_alpha.data,  max=0.0)
                hyco.textual_alpha.data = torch.clamp(hyco.textual_alpha.data, max=0.0)

                feats = hyco.encode_text(toks, project=True)       # [T×D]
                feats = feats / feats.norm(dim=-1, keepdim=True)   # L₂‐normalize
                m     = feats.mean(dim=0)                          # mean‐pool → [D]
                m     = m / m.norm()                               # re‐normalize
                ws.append(m)

            # stack into [C×D], then transpose → [D×C]
            return torch.stack(ws, dim=1).to(dev)

    def preprocess(self):
        """Reproduce HyCoCLIP’s training transforms (size from patch_embed)."""
        vp = self.model.visual.patch_embed
        raw = vp.img_size if hasattr(vp, "img_size") else 224
        n_px = raw[0] if isinstance(raw, (tuple, list)) else raw

        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
            Normalize(self.model.pixel_mean, self.model.pixel_std),
        ])

    def forward_batch(self, images: torch.Tensor, targets) -> float:
        """
        Runs the same contrastive + hyperbolic‐entailment logic as HyCoCLIP.forward,
        but on a single GPU (i.e. skips distributed gather if not initialized).
        Returns a single scalar loss.
        """
        self.model.eval()
        with torch.no_grad():
            dev  = device()
            hyco = self.model

            # 0) undo upstream CLIP norm → apply HyCoCLIP preprocess
            imgs = undo_default_preprocessing(images)
            imgs = [ self.preprocess()(ToPILImage()(im)) for im in imgs ]
            batch = torch.stack(imgs, dim=0).to(dev)   # [B×3×H×W]

            # 1) clamp learnable params
            hyco.curv.data          = torch.clamp(hyco.curv.data, **hyco._curv_minmax)
            hyco.visual_alpha.data  = torch.clamp(hyco.visual_alpha.data,  max=0.0)
            hyco.textual_alpha.data = torch.clamp(hyco.textual_alpha.data, max=0.0)
            _curv = hyco.curv.exp()

            # 2) encode everything
            image_feats     = hyco.encode_image(batch,          project=True)  # [B×D]
            if isinstance(targets[0], str):
                idxs = [imagenet_classes.index(t) for t in targets]
                targets = torch.tensor(idxs, dtype=torch.long, device=device())
            else:
                targets = torch.as_tensor(targets, dtype=torch.long, device=device())
            text_feats      = self.class_embeddings[targets]        # [B×D]
            box_image_feats = image_feats.clone()                   # using same imgs as "boxes"
            box_text_feats  = text_feats.clone()                    # ditto

            # 3) normalize
            image_feats     = image_feats     / image_feats.norm(dim=-1, keepdim=True)
            text_feats      = text_feats      / text_feats.norm(dim=-1, keepdim=True)
            box_image_feats = box_image_feats / box_image_feats.norm(dim=-1, keepdim=True)
            box_text_feats  = box_text_feats  / box_text_feats.norm(dim=-1, keepdim=True)

            # 4) gather (only if torch.distributed is up)
            if tdist.is_available() and tdist.is_initialized():
                all_image = torch.cat(dist.gather_across_processes(image_feats),     dim=0)
                all_text  = torch.cat(dist.gather_across_processes(text_feats),      dim=0)
            else:
                all_image = image_feats
                all_text  = text_feats

            # 5) contrastive logits via hyperbolic pairwise‐distance
            image_logits     = -L.pairwise_dist(image_feats,     all_text,  _curv)
            text_logits      = -L.pairwise_dist(text_feats,      all_image, _curv)
            box_image_logits = -L.pairwise_dist(box_image_feats, all_text,  _curv)
            box_text_logits  = -L.pairwise_dist(box_text_feats,  all_image, _curv)

            # ——— TRANSPOSE the two [C×B] tensors to [B×C]:
            text_logits     = text_logits.T      # now [B×C]
            box_text_logits = box_text_logits.T  # now [B×C]
            print()

            # 6) clamp & scale
            hyco.logit_scale.data = torch.clamp(hyco.logit_scale.data, max=4.6052)
            scale = hyco.logit_scale.exp()
            image_logits     = scale * image_logits
            text_logits      = scale * text_logits
            box_image_logits = scale * box_image_logits
            box_text_logits  = scale * box_text_logits

            # 7) build targets & contrastive loss
            B    = image_feats.shape[0]
            rank = getattr(hyco, "_rank", 0)  # if no rank attr, assume 0
            targets = torch.arange(B, device=dev) + B * rank

            contrastive_loss = 0.25 * (
                torch.nn.functional.cross_entropy(image_logits,     targets)
            + torch.nn.functional.cross_entropy(text_logits,      targets)
            + torch.nn.functional.cross_entropy(box_image_logits, targets)
            + torch.nn.functional.cross_entropy(box_text_logits,  targets)
            )

            # 8) hyperbolic entailment loss
            angle_i_t   = L.oxy_angle(text_feats,      image_feats,     _curv)
            aper_i_t    = L.half_aperture(text_feats,  _curv)
            angle_bbt   = L.oxy_angle(box_text_feats,  box_image_feats, _curv)
            aper_bbt    = L.half_aperture(box_text_feats, _curv)
            angle_cib   = L.oxy_angle(box_image_feats, image_feats,     _curv)
            aper_cib    = L.half_aperture(box_image_feats, _curv)
            angle_cbt   = L.oxy_angle(box_text_feats,  text_feats,      _curv)
            aper_cbt    = L.half_aperture(box_text_feats, _curv)

            G, H = 0.7, 1.2
            te_loss  = torch.clamp(angle_i_t - G * aper_i_t,  min=0).mean()
            bte_loss = torch.clamp(angle_bbt - G * aper_bbt,  min=0).mean()
            cib_loss = torch.clamp(angle_cib - H * aper_cib,  min=0).mean()
            cbt_loss = torch.clamp(angle_cbt - H * aper_cbt,  min=0).mean()

            entailment_loss = 0.5 * (te_loss + bte_loss + cib_loss + cbt_loss)

            # 9) final (contrastive + optional entailment)
            loss = contrastive_loss
            if getattr(hyco, "entail_weight", 0) > 0:
                loss = loss + hyco.entail_weight * entailment_loss

            return loss
        
        #python -m modelvshuman --models hycoclip --datasets cue-conflict --batch-size 64 --num-workers 16