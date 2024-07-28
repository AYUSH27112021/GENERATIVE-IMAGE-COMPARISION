# FID Metric Evaluation

This metric is used to evaluate images using the FID (Fréchet Inception Distance) metric. The FID metric is commonly used to assess the quality of images generated by generative models by comparing the generated images to real images.

## Overview

The framework performs the following steps:
1. **Load the Model**: Initializes the Inception v3 model for FID computation.
2. **Image Preprocessing**: Preprocesses the input images using the appropriate pipeline.
3. **Feature Extraction**: Extracts features for both the real and generated images using the Inception v3 model.
4. **FID Computation**: Computes the FID score between the real and generated images.

## Requirements

- Python 3.10+
- CUDA-enabled GPU (for optimal performance)
- PyTorch
- torchvision

## Setup and Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.ecodesamsung.com/SRIB-PRISM/CITB_23RSG25_Develop_a_Quality_Assurance_framework_for_Image_Generation_Models
    cd "CITB_23RSG25_Develop_a_Quality_Assurance_framework_for_Image_Generation_Models"
    cd FID
    ```

2. **Custom Model Training**:
    The weights from online image recognition models were found insufficient for our study. Therefore, training a custom model on a custom dataset is necessary.