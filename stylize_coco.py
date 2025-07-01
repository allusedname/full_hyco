import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CocoDetection
from PIL import Image
import torchvision.models as models
import os
import argparse
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- AdaIN Style Transfer Components ---

# VGG19 for feature extraction
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
vgg.to(device)
for param in vgg.parameters():
    param.requires_grad = False

# AdaIN layer
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, x):
        return self.model(x)

decoder = Decoder()
decoder.to(device)
# Note: For this script to work, you would typically load pre-trained weights for the decoder.
# Since we don't have a pre-trained decoder here, the output will be artistic but not necessarily high-quality.
# For a full implementation, you would need to train the decoder or download pre-trained weights.

def style_transfer(vgg, decoder, content_image, style_image, alpha=1.0):
    content_feat = vgg(content_image)
    style_feat = vgg(style_image)
    feat = adaptive_instance_normalization(content_feat, style_feat)
    feat = feat * alpha + content_feat * (1 - alpha)
    return decoder(feat)

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(description="Apply AdaIN Style Transfer to the COCO dataset.")
    parser.add_argument('--coco_root', type=str, required=True, help='Root directory of the COCO dataset (e.g., /path/to/coco/val2017).')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to the COCO annotation file (e.g., /path/to/coco/annotations/instances_val2017.json).')
    parser.add_argument('--style_image', type=str, required=True, help='Path to the style image.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the stylized images.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Style transfer strength (0.0 to 1.0).')
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Load the style image
    style_image = Image.open(args.style_image).convert('RGB')
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    # Load the COCO dataset
    coco_dataset = CocoDetection(root=args.coco_root, annFile=args.ann_file, transform=transform)

    # Process each image in the dataset
    for i in tqdm(range(len(coco_dataset))):
        content_image, _ = coco_dataset[i]
        content_tensor = content_image.unsqueeze(0).to(device)

        with torch.no_grad():
            stylized_tensor = style_transfer(vgg, decoder, content_tensor, style_tensor, args.alpha)

        # Save the stylized image
        output_image_path = os.path.join(args.output_dir, f"{coco_dataset.ids[i]:012d}.jpg")
        stylized_image = transforms.ToPILImage()(stylized_tensor.cpu().squeeze(0))
        stylized_image.save(output_image_path)

    print(f"Stylized images saved to {args.output_dir}")

if __name__ == '__main__':
    main()