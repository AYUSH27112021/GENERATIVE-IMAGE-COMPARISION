# Quality Assurance Framework for Image Generation Models Result

## Overview

This folder contains the results of testing a few generated images against specific prompts. The evaluation includes scores from various metrics and a combined average score.

## Contents

- **assets/**: Contains the images and prompts used for testing.
- **results.csv**: A CSV file with the scores of all metrics for each image and prompt, including a combined average score.
- **framework/**: Contains a conceptual diagram illustrating the workflow.

## Description

The evaluation framework assesses generated images based on several quality metrics. Each metric provides a different perspective on the image quality and relevance to the given prompt. The final results include both individual metric scores and a combined average score for a comprehensive evaluation.

## Metrics Used

1. **Aesthetic Score**: Evaluates the visual appeal of the image.
2. **CLIP**: Measures the similarity between the image and the input text prompt.
3. **QAlign**: Provides quality alignment scores for the image.
4. **SAM CLIP**: Segments the image and evaluates the relevance of each segment to the prompt.
5. **IQA PyTorch**: (Optional) Provides an Image Quality Assessment if weights are available.

## Concept Diagram

The **framework** folder contains a flowchart that visually represents the evaluation process. This diagram helps in understanding the workflow and the interactions between different components of the framework.

## Notes
