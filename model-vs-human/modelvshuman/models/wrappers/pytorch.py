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
        self.zeroshot_weights = self._get_zeroshot_weights(imagenet_classes, imagenet_templates)

    def _get_zeroshot_weights(self, class_names, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(class_names):
                texts = [template.format(class_name) for template in templates]
                texts = clip.tokenize(texts).to(device())
                class_embeddings = self.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device())

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
        arch: str = "vit_small_patch16_224",
        checkpoint_path: str = "modelvshuman/models/hycoclip_vit_s.pth"
    ):
        # Build vision and text backbones from hycoclip codebase
        visual = build_timm_vit(arch=arch, global_pool="token", grad_checkpointing=False)
        textual = TransformerTextEncoder(arch="L12_W512_A8", vocab_size=49408, context_length=77)

        # Instantiate the HyCoCLIP model
        hyco_model = HyCoCLIP(
            visual=visual,
            textual=textual,
            embed_dim=512,
            curv_init=1.0,
            learn_curv=True,
            entail_weight=0.0,
            use_boxes=True, # Required by HyCoCLIP
        )

        # Load the pretrained weights
        base_dir = Path(__file__).resolve().parent.parent
        base_models_dir = Path(__file__).resolve().parent.parent
        ckpt_path = base_models_dir / "hycoclip_vit_s.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state = ckpt.get("model", ckpt)
        hyco_model.load_state_dict(state, strict=True)

        # Initialize the parent PytorchModel class
        super().__init__(hyco_model, model_name, *args)
        self.tokenizer = Tokenizer()

        # Precompute the zero-shot classifier weights
        self.zeroshot_weights = self._get_zeroshot_weights(imagenet_classes, imagenet_templates).to(device())


    def _get_zeroshot_weights(self, class_names: Sequence[str], templates: Sequence[str]) -> torch.Tensor:
        hyco = self.model
        dev = device()
        with torch.no_grad():
            ws = []
            for cls in tqdm(class_names, desc="HyCoCLIP zeroshot"):
                prompts = [t.format(cls) for t in templates]
                toks = self.tokenizer(prompts)
                
                # 1. Encode all text templates directly into hyperbolic space
                all_hyp_feats = hyco.encode_text(toks, project=True)
                
                # 2. Correctly average the features in hyperbolic space
                with torch.autocast(dev.type, dtype=torch.float32):
                    curv = hyco.curv.exp()
                    
                    # a. Map features to the tangent space at the origin
                    tangent_space_feats = L.log_map0(all_hyp_feats.float(), curv)
                    
                    # b. Compute the standard average in this Euclidean-like tangent space
                    mean_tangent_feat = tangent_space_feats.mean(dim=0, keepdim=True)
                    
                    # c. Map the average back to the hyperbolic manifold
                    class_embedding = L.exp_map0(mean_tangent_feat, curv).squeeze(0)

                ws.append(class_embedding)
            zeroshot_weights = torch.stack(ws, dim=1).to(dev)
        return zeroshot_weights

    def forward_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encodes a batch of images and computes logits against the zero-shot classifier
        by mirroring the logic in the model's original training objective.
        """
        self.model.eval()
        
        # Note: This preprocessing pattern remains fragile. A more robust solution
        # would be to integrate the model's normalization directly into the
        # data loader's transformation pipeline.
        images = undo_default_preprocessing(images)
        images = images.to(device())

        with torch.no_grad():
            # Encode images and project them onto the hyperbolic manifold
            image_feats = self.model.encode_image(images, project=True)

            # Get model curvature and the learned logit scale
            curv = self.model.curv.exp()
            scale = self.model.logit_scale.exp()

            # 1. Compute scores using hyperbolic distance, the same metric used in training.
            # A smaller distance means higher similarity, so we negate the distance.
            logits = -L.pairwise_dist(image_feats, self.zeroshot_weights.T, curv)

            # 2. Apply the learned temperature scaling
            scaled_logits = scale * logits

        return self.to_numpy(scaled_logits)
        
    def preprocess(self):
        """
        Creates a preprocessing pipeline that aligns with CLIP's methodology.
        It resizes, crops, and converts the image to a tensor in the [0, 1] range.
        Normalization is NOT applied here, as it is handled internally by the
        model's encode_image method.
        """
        n_px = self.model.visual.input_resolution
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
        ])




        
        #python -m modelvshuman --models hycoclip --datasets cue-conflict --batch-size 64 --num-workers 16