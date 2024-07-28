# Image Quality and Aesthetics Scoring with Q-Align

This Metric is to evaluate images based on quality and aesthetics using the Q-Align model.

## Overview

The framework performs the following steps:
1. **Load the Model**: Initializes the Q-Align model.
2. **Image Evaluation**: Evaluates the input image to compute quality and aesthetics scores.

## Requirements

- Python 3.10+
- CUDA-enabled GPU (for optimal performance)
- PyTorch with FlashAttention support
- Transformers library

## Setup and Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.ecodesamsung.com/SRIB-PRISM/CITB_23RSG25_Develop_a_Quality_Assurance_framework_for_Image_Generation_Models
    cd "CITB_23RSG25_Develop_a_Quality_Assurance_framework_for_Image_Generation_Models"
    cd Q_ALING
    ```

3. **Download Model Weights**:
    The Q-Align model weights will be downloaded automatically when running the code.

## Usage

### Command Line Interface

To process an image and get quality and aesthetics scores:
```bash
python q_align_hf.py <image_path>
```

**_NOTE:_** 

1. To run the evaluation, the GPU must support CUDA, and PyTorch must be compiled with FlashAttention.
2. The evaluation requires a sufficient amount of GPU memory to load and run the model.
3. The Q-Align model can be used to score images for both quality and aesthetics

