# configs/train_partimagenet.py

# ─────────────────────────────────────────────────────────────────────────────
# 1) Bring in your base model/optimizer/train loop from your ViT-S config
# ─────────────────────────────────────────────────────────────────────────────
from configs.train_hycoclip_vit_s import model, optim, train

# ─────────────────────────────────────────────────────────────────────────────
# 2) Glue in our PartImageNetDataset + transforms + tokenizer
# ─────────────────────────────────────────────────────────────────────────────
from hycoclip.data.partimagenet_dataset import PartImageNetDataset
from torchvision import transforms as T

# -- training split --
dataset = PartImageNetDataset(
    root="C:\\Users\\xjzb2\\compo_learning\\PartImageNet_Seg\\PartImageNet",  # e.g. C:/.../sample/PartImageNet
    split="train",
    transform=T.Compose([
        T.RandomResizedCrop(224, scale=(0.5,1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ]),
    tokenizer=model.tokenizer,    # reuse the same CLIP tokenizer :contentReference[oaicite:0]{index=0}
)

# -- validation split --
dataset_val = PartImageNetDataset(
    root=dataset.root if hasattr(dataset, "root") else "/absolute/path/to/sample_unzipped/PartImageNet",
    split="val",
    transform=dataset.transform,
    tokenizer=dataset.tokenizer,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3) No other symbols need touching—`train.py` will see `dataset`/`dataset_val`
#    alongside the imported `model`, `optim`, and `train`.
# ─────────────────────────────────────────────────────────────────────────────
