# GENERATIVE-IMAGE-COMPARISION Models

This framework utilizes various evaluation metrics to provide a combined and normalized score that assesses the quality and relevance of generated images based on input prompts.

## Project Overview

The framework evaluates images using the following metrics:
- **Aesthetic Score**: Evaluates the visual appeal of the image.
- **CLIP**: Measures the similarity between the image and the input text prompt.
- **QAlign**: Provides quality and aesthetics alignment scores for the image.
- **SAM CLIP**: Segments the image and evaluates the relevance of segmented parts to the prompt.
- **IQA PyTorch**: (Optional) Provides an Image Quality Assessment if weights are available.

![Diagram](https://github.com/AYUSH27112021/GENERATIVE-IMAGE-COMPARISION/results/Framework/Quality_Assurance_framework.png")

## Components

- **Input Image**: The image to be evaluated.
- **Input Prompt**: The text description to which the image is compared.
- **Model Weights**: Pre-trained weights for the models used in the evaluation.

## Evaluation Process

1. **Input Image and Prompt**: Provide the image and the corresponding text prompt.
2. **Evaluation by Models**: Various models evaluate the image based on different metrics.
3. **Aesthetic Score**: Calculates the aesthetic appeal of the image.
4. **CLIP**: Computes the similarity between the image and the text prompt.
5. **QAlign**: Provides quality alignment scores.
6. **SAM CLIP**: Segments the image and evaluates the relevance of each segment.
7. **IQA PyTorch**: (Optional) Provides an Image Quality Assessment if weights are available.
8. **Combined and Normalized Score**: The scores from all evaluations are combined and normalized to provide a final score between 0 and 1.

## Setup and Installation

1. **Clone the Repository**:
    ```bash
    git clone GENERATIVE-IMAGE-COMPARISION
    cd "GENERATIVE-IMAGE-COMPARISION"
    ```

2. **Install Dependencies**:
    Ensure you have Python installed. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Model Weights**:
    Download and place the pre-trained model weights in the appropriate directory as needed.

## Usage

### Command Line Interface

To compute the similarity between an image and a text prompt using all metrics at once:
```bash
python All_metrics.py <image_path> <text_input>
```
### Example
```bash
python compute_similarity.py "assets\areoplane_prompt.jpg" "A picture of an airplane in a thunderstorm.
```

**_NOTE:_** 

1. To run the `ALL_metric.py` file, the GPU must support CUDA, and PyTorch must be compiled with FlashAttention.
2. The `ALL_metric.py` file requires a sufficient amount of GPU memory to load and run the models.
3. Some models also support `torch.run` and can be executed in parallel with others.
4. Models can also be run separately. See the `metric` folder README for more details.

