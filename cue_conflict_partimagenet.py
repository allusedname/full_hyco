"""
Generate a cue‑conflict variant of a PartImageNet style dataset using
Adaptive Instance Normalisation (AdaIN).

This script traverses a directory of PartImageNet images (organised into
subdirectories such as ``train``, ``val`` and ``test``) and, for every
content image, randomly samples a style image from a different class.  A
class is inferred from the ImageNet synset prefix of the filename (for
example ``n01440764`` from ``n01440764_8240.JPEG``).  The content and
style images are then combined using the AdaIN model provided in the
``AdaIN_master`` folder.  The resulting stylised image retains the
geometric structure of the content image but adopts the textural
statistics of the style image, yielding a texture–shape cue conflict.

The output file names follow the convention used by the original
texture‐versus‐shape dataset: ``<contentClass><contentIndex>-<styleClass><styleIndex>.png``.
For example ``n01440764_8240.JPEG`` stylised with ``n01443537_2249.JPEG``
would produce ``n01440764_8240-n01443537_2249.png``.

The AdaIN encoder/decoder weights are not shipped with this repository
for licencing reasons.  You can download pre‑trained weights from the
official AdaIN implementation (see the README in ``AdaIN_master``) and
place ``decoder.pth`` and ``vgg_normalised.pth`` inside the
``AdaIN_master`` directory.  If the weights are absent the model will
still run but the stylisation quality will be poor.

Example usage:

.. code-block:: bash

    # Assume you have downloaded the weights into AdaIN_master
    python cue_conflict_partimagenet.py \
        --input_root sample/PartImageNet/images \
        --output_root sample/PartImageNet_cue_conflict/images \
        --alpha 1.0

This will process all splits found under ``input_root``.  You can
restrict processing to a single split (e.g. ``train``) using the
``--split`` argument.
"""

import argparse
import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# AdaIN setup

def _load_adain_model(adain_dir: str) -> Tuple[nn.Module, nn.Module]:
    """Load the AdaIN encoder (VGG) and decoder.

    The ``AdaIN_master`` directory contains ``net.py`` defining the model
    architecture and (optionally) ``decoder.pth`` and
    ``vgg_normalised.pth`` containing trained weights.  If the weight
    files are present they are loaded; otherwise the randomly initialised
    model is returned.

    Parameters
    ----------
    adain_dir: str
        Path to the ``AdaIN_master`` directory within this repository.

    Returns
    -------
    encoder: nn.Module
        The VGG19 encoder truncated at relu4_1.
    decoder: nn.Module
        The decoder network mapping AdaIN features back to image space.
    """
    import importlib.util
    import sys

    # Dynamically import the net module from AdaIN_master
    net_path = os.path.join(adain_dir, "net.py")
    spec = importlib.util.spec_from_file_location("adain_net", net_path)
    adain_net = importlib.util.module_from_spec(spec)
    sys.modules["adain_net"] = adain_net
    spec.loader.exec_module(adain_net)  # type: ignore

    # Load encoder and decoder architectures
    encoder: nn.Module = adain_net.vgg
    decoder: nn.Module = adain_net.decoder

    # Optionally load trained weights if available
    decoder_weights = os.path.join(adain_dir, "decoder.pth")
    vgg_weights = os.path.join(adain_dir, "vgg_normalised.pth")

    if os.path.isfile(decoder_weights):
        try:
            decoder.load_state_dict(torch.load(decoder_weights, map_location="cpu"))
            print(f"Loaded decoder weights from {decoder_weights}")
        except Exception as e:
            print(f"Warning: failed to load decoder weights from {decoder_weights}: {e}")
    else:
        print(
            f"Warning: decoder weights not found at {decoder_weights}; "
            "using random initialisation."
        )

    if os.path.isfile(vgg_weights):
        try:
            encoder.load_state_dict(torch.load(vgg_weights, map_location="cpu"))
            print(f"Loaded VGG weights from {vgg_weights}")
        except Exception as e:
            print(f"Warning: failed to load VGG weights from {vgg_weights}: {e}")
    else:
        print(
            f"Warning: VGG weights not found at {vgg_weights}; "
            "using random initialisation."
        )

    # Disable gradients for encoder
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder, decoder


@torch.no_grad()
def _adain_style_transfer(
    encoder: nn.Module,
    decoder: nn.Module,
    content: torch.Tensor,
    style: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Perform AdaIN style transfer on a pair of images.

    Parameters
    ----------
    encoder: nn.Module
        The VGG encoder truncated at relu4_1.
    decoder: nn.Module
        The decoder network.
    content: torch.Tensor
        Content image batch of shape (N, 3, H, W) normalized to [0,1].
    style: torch.Tensor
        Style image batch of shape (N, 3, H, W) normalized to [0,1].
    alpha: float, default=1.0
        Blend factor between the stylised features and content features. A
        value of 1.0 applies full style and 0.0 keeps the original
        content.  Intermediate values interpolate between the two.

    Returns
    -------
    torch.Tensor
        Stylised image batch of shape (N, 3, H, W).  Pixel values are
        clamped to [0,1].
    """
    # Import functions from AdaIN_master.function lazily to avoid global import
    from AdaIN_master.function import adaptive_instance_normalization

    # Extract features
    content_feat = encoder(content)
    style_feat = encoder(style)
    # Adaptive Instance Normalisation
    t = adaptive_instance_normalization(content_feat, style_feat)
    # Interpolate with content features
    t = alpha * t + (1.0 - alpha) * content_feat
    # Decode back to image space
    output = decoder(t)
    # Clamp to valid range
    return output.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Utility functions

def _gather_image_paths(root_dir: str) -> Dict[str, List[str]]:
    """Group image file paths by their inferred class.

    Given a directory containing JPEG images whose filenames start with
    an ImageNet synset (e.g. ``n01440764_8240.JPEG``), this function
    returns a dictionary mapping each synset to a list of absolute
    filenames belonging to that class.

    Parameters
    ----------
    root_dir: str
        Directory containing images.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from class name (synset) to list of file paths.
    """
    classes: Dict[str, List[str]] = {}
    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if not os.path.isfile(path):
            continue
        if not entry.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # Extract class prefix (characters before the first underscore)
        parts = entry.split("_")
        if not parts:
            continue
        cls = parts[0]
        classes.setdefault(cls, []).append(path)
    return classes


def _build_content_style_pairs(classes: Dict[str, List[str]]) -> List[Tuple[str, str, str, str]]:
    """Create a list of content–style pairs ensuring cue conflict.

    For each content image in each class the function randomly selects a
    style image from a *different* class.  The returned list contains
    tuples of the form ``(content_path, style_path, content_cls, style_cls)``.

    Parameters
    ----------
    classes: Dict[str, List[str]]
        Mapping from class name to list of image paths.

    Returns
    -------
    List[Tuple[str, str, str, str]]
        A list of assignments for processing.
    """
    pairs: List[Tuple[str, str, str, str]] = []
    class_names = list(classes.keys())
    for content_cls, content_files in classes.items():
        # Candidate style classes exclude the content class
        style_candidates = [c for c in class_names if c != content_cls]
        if not style_candidates:
            # Only one class present – nothing to do
            continue
        for content_path in content_files:
            style_cls = random.choice(style_candidates)
            style_path = random.choice(classes[style_cls])
            pairs.append((content_path, style_path, content_cls, style_cls))
    return pairs


def _prepare_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Load an image file and convert it into a normalised tensor.

    Images are resized to have their smallest side equal to 512 pixels
    while preserving the aspect ratio, then centre‐cropped to 512×512.
    Finally, pixel values are scaled to [0,1].  Adjust these
    transformations as required for your dataset.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(512),  # maintain aspect ratio
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return tensor.to(device)


def _save_output_image(
    tensor: torch.Tensor,
    output_path: str,
) -> None:
    """Save a stylised tensor as a PNG image."""
    # Remove batch dimension
    tensor = tensor.squeeze(0).cpu()
    # Convert to PIL image (transforms.ToPILImage expects [0,1] range)
    img = transforms.ToPILImage()(tensor)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, format="PNG")


# ---------------------------------------------------------------------------
# Main entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a cue‑conflict version of PartImageNet using the "
            "AdaIN model from the AdaIN_master folder."
        )
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help=(
            "Root directory containing PartImageNet images.  Expected to "
            "contain subdirectories such as 'train', 'val' and 'test'."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=(
            "Root directory for saving stylised images.  The same "
            "subdirectory structure as the input will be created inside "
            "this folder."
        ),
    )
    parser.add_argument(
        "--adain_dir",
        type=str,
        default="AdaIN_master",
        help=(
            "Path to the AdaIN_master directory.  If relative, it is "
            "interpreted with respect to this script's location."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help=(
            "Optionally process only a single split (train/val/test).  "
            "If omitted all splits under input_root are processed."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Stylisation strength (0.0 retains content, 1.0 fully stylises).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible style pairings.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve AdaIN directory relative to this script if necessary
    if not os.path.isabs(args.adain_dir):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        adain_dir = os.path.join(script_dir, args.adain_dir)
    else:
        adain_dir = args.adain_dir

    # Load encoder/decoder
    encoder, decoder = _load_adain_model(adain_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()
    decoder.to(device).eval()

    # Determine which splits to process
    splits: List[str]
    if args.split is None:
        splits = [d for d in os.listdir(args.input_root) if os.path.isdir(os.path.join(args.input_root, d))]
    else:
        splits = [args.split]

    for split in splits:
        input_split_dir = os.path.join(args.input_root, split)
        if not os.path.isdir(input_split_dir):
            print(f"Skipping non‑existent split directory: {input_split_dir}")
            continue
        print(f"Processing split '{split}'…")
        # Group images by class
        classes = _gather_image_paths(input_split_dir)
        if len(classes) < 2:
            print(
                f"Warning: fewer than two classes detected in {input_split_dir}; "
                "unable to create cue‑conflict pairs."
            )
            continue
        pairs = _build_content_style_pairs(classes)
        # For reproducibility we can shuffle the pair list
        random.shuffle(pairs)
        # Process each pair
        for content_path, style_path, content_cls, style_cls in pairs:
            # Load tensors
            content_tensor = _prepare_image_tensor(content_path, device)
            style_tensor = _prepare_image_tensor(style_path, device)
            # Stylise
            output_tensor = _adain_style_transfer(
                encoder, decoder, content_tensor, style_tensor, alpha=args.alpha
            )
            # Determine output filename
            content_name = os.path.splitext(os.path.basename(content_path))[0]
            style_name = os.path.splitext(os.path.basename(style_path))[0]
            output_name = f"{content_name}-{style_name}.png"
            # Save
            output_dir = os.path.join(args.output_root, split)
            output_path = os.path.join(output_dir, output_name)
            _save_output_image(output_tensor, output_path)
        print(f"Finished processing split '{split}'.")


if __name__ == "__main__":
    main()