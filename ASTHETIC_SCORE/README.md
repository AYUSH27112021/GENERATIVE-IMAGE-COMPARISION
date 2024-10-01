# Aesthetic Score Prediction using CLIP and MLP

This Metric is used to predict the aesthetic score of images using a pre-trained CLIP model and a Multi-Layer Perceptron (MLP) for regression.

## Overview

The framework performs the following steps:
1. **Load the Model**: Initializes the CLIP model and the MLP for aesthetic score prediction.
2. **Image Preprocessing**: Preprocesses the input image using CLIP's preprocessing pipeline.
3. **Feature Extraction**: Extracts image features using the CLIP model.
4. **Score Prediction**: Uses the MLP to predict the aesthetic score from the extracted features.

## Requirements

- Python 3.10+
- CUDA-enabled GPU (for optimal performance)
- PyTorch
- PyTorch Lightning
- Transformers library
- CLIP library

## Setup and Installation

1. **Clone the Repository**:
2. **Install Dependencies**:
    Ensure you have Python installed. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. **Download Model Weights**:
    Download the pre-trained weights for the MLP model (`sac+logos+ava1-l14-linearMSE.pth`) and place them in the appropriate directory.

## Usage

### Command Line Interface

To process an image and get the aesthetic score:
```bash
python aesthetic_score.py <image_path>
```

**_NOTE:_** 

1. An example script `aesthetic_score_train.py` is provided to train your custom model and obtain weights.
