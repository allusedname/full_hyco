# Model Zoo

We trained HyCoCLIP along with MERU and CLIP as baselines with Vision Transformers small and base. These were trained on the [GRIT dataset](https://huggingface.co/datasets/zzliang/GRIT) having 20.5 million grounded image-text pairs for 500,000 iterations with batch size of 768. The model configs can be found in [`./configs`](./configs) directory. Checkpoints are available at [huggingface/avik-pal/hycoclip](https://huggingface.co/avik-pal/hycoclip). These can be downloaded with `huggingface-cli` using the following command:

```
huggingface-cli download avik-pal/hycoclip hycoclip_vit_s.pth --local-dir ./checkpoints
```

|     model     | ImageNet ZS Classification (Acc) | Caltech-101 ZS Classification (Acc) | Food-101 ZS Classification (Acc) | COCO ZS Text Retrieval (R@10) | COCO ZS Image Retrieval (R@10) | huggingface link                                                                            |
| :-----------: | :------------------------------: | :---------------------------------: | :------------------------------: | :---------------------------: | :----------------------------: | ------------------------------------------------------------------------------------------- |
|   CLIP-S/16   |               36.7               |                73.6                 |               44.7               |             79.1              |              65.2              | [clip_vit_s.pth](https://huggingface.co/avik-pal/hycoclip/blob/main/clip_vit_s.pth)         |
|   MERU-S/16   |               35.4               |                73.0                 |               48.8               |             78.8              |              65.3              | [meru_vit_s.pth](https://huggingface.co/avik-pal/hycoclip/blob/main/meru_vit_s.pth)         |
| HyCoCLIP-S/16 |             **41.7**             |              **75.7**               |             **50.2**             |           **79.5**            |            **66.6**            | [hycoclip_vit_s.pth](https://huggingface.co/avik-pal/hycoclip/blob/main/hycoclip_vit_s.pth) |
|               |                                  |                                     |                                  |                               |
|   CLIP-B/16   |               40.6               |                76.7                 |               48.6               |             81.5              |              68.5              | [clip_vit_b.pth](https://huggingface.co/avik-pal/hycoclip/blob/main/clip_vit_b.pth)         |
|   MERU-B/16   |               40.1               |                72.8                 |               51.5               |             82.0              |              68.6              | [meru_vit_b.pth](https://huggingface.co/avik-pal/hycoclip/blob/main/meru_vit_b.pth)         |
| HyCoCLIP-B/16 |             **45.8**             |              **81.3**               |             **59.2**             |           **82.0**            |            **69.3**            | [hycoclip_vit_b.pth](https://huggingface.co/avik-pal/hycoclip/blob/main/hycoclip_vit_b.pth) |
