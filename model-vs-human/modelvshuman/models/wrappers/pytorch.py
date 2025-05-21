import math
import PIL
import clip
import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from tqdm import tqdm

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
            use_boxes=True,
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

        # 4) cache zero‐shot weights [D×C]
        self.zeroshot_weights = self._get_zeroshot_weights(
            imagenet_classes,
            imagenet_templates
        )

        # 5) pre‐tokenize all C class prompts for fast forward()
        dev = device()
        all_tokens = clip.tokenize(imagenet_classes, context_length=77).to(dev)
        # store as List[Tensor] for HyCoCLIP.forward signature
        self.text_tokens = [all_tokens[i] for i in range(all_tokens.size(0))]
        self.box_tokens  = self.text_tokens  # dummy, entail_weight=0

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

    def forward_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Single‐process zero‐shot inference:
         1) clamp hyperbolic params
         2) encode images & texts
         3) safe‐gather negatives
         4) compute contrastive logits
        """
        self.model.eval()
        with torch.no_grad():
            dev = device()

            # undo any upstream CLIP norm & re‐apply HyCoCLIP preprocess
            images = undo_default_preprocessing(images)
            imgs   = [self.preprocess()(ToPILImage()(im)) for im in images]
            batch  = torch.stack(imgs, dim=0).to(dev)   # [B×3×H×W]

            hyco = self.model
            # clamp parameters
            hyco.curv.data          = torch.clamp(hyco.curv.data, **hyco._curv_minmax)
            hyco.visual_alpha.data  = torch.clamp(hyco.visual_alpha.data,  max=0.0)
            hyco.textual_alpha.data = torch.clamp(hyco.textual_alpha.data, max=0.0)
            _curv = hyco.curv.exp()

            # encode
            img_feats = hyco.encode_image(batch, project=True)         # [B×D]
            txt_feats = hyco.encode_text(self.text_tokens, project=True)  # [C×D]
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            # gather negatives (no-op if not in dist)
            try:
                all_img = torch.distributed.gather_across_processes(img_feats)
                all_txt = torch.distributed.gather_across_processes(txt_feats)
            except Exception:
                all_img = [img_feats]
                all_txt = [txt_feats]
            all_img = torch.cat(all_img, dim=0)   # [B_total×D]
            all_txt = torch.cat(all_txt, dim=0)   # [C_total×D]

            # contrastive logits
            image_logits = -pairwise_dist(img_feats, all_txt, _curv)
            hyco.logit_scale.data = torch.clamp(hyco.logit_scale.data, max=4.6052)
            logits = hyco.logit_scale.exp() * image_logits  # [B×C_total]

            return self.to_numpy(logits)
        
        #python -m modelvshuman --models hycoclip --datasets cue-conflict --batch-size 64 --num-workers 16