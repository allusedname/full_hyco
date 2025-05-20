#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

from hycoclip.config import LazyCall as L
from hycoclip.encoders.image_encoders import build_timm_vit
from hycoclip.encoders.text_encoders import TransformerTextEncoder
from hycoclip.models import MERU

from .train_hycoclip_vit_l import dataset, optim, train


model = L(MERU)(
    visual=L(build_timm_vit)(
        arch="vit_large_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77 # originally context_length=77
    ),
    embed_dim=512,
    curv_init=1.0,
    learn_curv=True,
    entail_weight=0.2,
    use_boxes=False,
)
