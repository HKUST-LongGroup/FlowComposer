# FlowComposer: Composable Flows for Compositional Zero-Shot Learning

## Setup

```bash
conda create --name flowcomposer python=3.7
conda activate flowcomposer
pip install -r requirements.txt
```

### CLIP Download

Download the CLIP ViT-L/14 checkpoint:

```bash
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

## Training

Please refer to `run.sh`.

## Acknowledgement

This repository is developed based on *Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning* and *Exploring Cross-Modal Flows for Few-Shot Learning (FMA)*.
