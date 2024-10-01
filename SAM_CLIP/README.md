# SAM-CLIP Metric

This Metric is used to evaluate images against a given prompt using SAM (Segment Anything Model) and CLIP (Contrastive Language-Image Pretraining) models. It segments the image, extracts keywords from the prompt, and computes scores based on the relevance of the image segments to the prompt.

## Overview

The Metric performs the following steps:
1. **Image Segmentation**: Segments the input image using the SAM model.
2. **Keyword Extraction**: Extracts keywords from the input prompt.
3. **Image-Prompt Matching**: Computes similarity scores between the image segments and the extracted keywords using the CLIP model.
4. **Score Adjustment**: Adjusts the scores based on predefined thresholds and criteria.

## Requirements

- Python 3.10+
- CUDA-enabled GPU (for optimal performance)
- PyTorch with FlashAttention support

## Setup and Installation

1. **Clone the Repository**:
2. **Download Model Weights**:
    Download and place the pre-trained SAM model weights (`sam_vit_h_4b8939.pth`) from [here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and place in the appropriate directory.

### Command Line Interface

3. To process an image and a prompt using SAM and CLIP models:
```bash
python Sam_Clip.py <image_path> <prompt>
```

