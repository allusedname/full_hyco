import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# exactly 41 part‐names from the Compositor repo, with Background last
# -----------------------------------------------------------------------------
PART_CLASSES = [
    'Quadruped-Head','Quadruped-Torso','Quadruped-Leg','Quadruped-Foot','Quadruped-Tail',
    'Biped-Head','Biped-Torso','Biped-Arm','Biped-Hand','Biped-Foot','Biped-Tail',
    'Fish-Head','Fish-Torso','Fish-Tail',
    'Bird-Head','Bird-Torso','Bird-Wing','Bird-Foot','Bird-Tail',
    'Snake-Head','Snake-Torso',
    'Reptile-Head','Reptile-Torso',
    'Car-Window','Car-Wheel','Car-Headlight','Car-Bumper','Car-Engine','Car-Mirror','Car-Seat',
    'Bicycle-Frame','Bicycle-Wheel','Bicycle-Handlebar','Bicycle-Seat',
    'Aeroplane-Body','Aeroplane-Wing','Aeroplane-Engine','Aeroplane-Tail',
    'Bottle-Body','Bottle-Mouth',
    'Background'
]
BACKGROUND_PART_ID = len(PART_CLASSES) - 1

class PartImageNetDataset(Dataset):
    """
    Returns a dict with:
      * image           – Tensor[3×224×224]   full image
      * text_inputs     – LongTensor[L]        tokenized synset
      * box             – Tensor[3×h×w]        per‐part crop
      * box_text_inputs – LongTensor[L]        tokenized part‐name
    """
    def __init__(self, root, split, transform=None, tokenizer=None):
        """
        root/PartImageNet/
          images/{split}        – .JPEG
          annotations/{split}    – .png per‐part mask + .json dims
          annotations/{split}_whole – .png full‐object masks
        """
        self.img_dir   = Path(root) / "images"      / split
        self.part_dir  = Path(root) / "annotations" / split
        self.obj_dir   = Path(root) / "annotations" / f"{split}_whole"
        self.transform = transform
        self.tokenizer = tokenizer

        # build (image_path, part_mask_path, part_id) list
        self.samples = []
        for img_path in sorted(self.img_dir.glob("*.JPEG")):
            name      = img_path.stem
            mask_path = self.part_dir / f"{name}.png"
            if not mask_path.exists():
                # no part‐mask for this split (e.g. train/val may be empty)
                continue

            mask = np.array(Image.open(mask_path))
            # find all part‐IDs except background
            pids = [int(pid) for pid in np.unique(mask) if pid != BACKGROUND_PART_ID]
            for pid in pids:
                self.samples.append((img_path, mask_path, pid))

        assert self.samples, f"No samples found in {self.img_dir}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, pid = self.samples[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        # crop box around this part
        ys, xs = np.where(mask == pid)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        box = img.crop((x1, y1, x2, y2))

        # apply same transform to both
        if self.transform:
            img = self.transform(img)
            box = self.transform(box)

        # to [0,1] TensorCHW
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float().div(255)
        box_tensor = torch.from_numpy(np.array(box)).permute(2,0,1).float().div(255)

        # text: object = synset prefix, part = PART_CLASSES[pid]
        synset   = img_path.stem.split("_")[0]
        part_txt = PART_CLASSES[pid]

        # tokenize and squeeze
        txt_obj = self.tokenizer(
          synset, return_tensors="pt", padding=True, truncation=True
        )["input_ids"].squeeze(0)
        txt_box = self.tokenizer(
          part_txt, return_tensors="pt", padding=True, truncation=True
        )["input_ids"].squeeze(0)

        return {
          "image":           img_tensor,
          "text_inputs":     txt_obj,
          "box":             box_tensor,
          "box_text_inputs": txt_box,
        }
