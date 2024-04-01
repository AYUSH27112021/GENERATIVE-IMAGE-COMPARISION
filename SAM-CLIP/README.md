# CLIP-SAM

Combining the CLIP with SAM.

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [Contrastive Language-Image Pre-Training (CLIP)](https://github.com/openai/CLIP)
- [keyBERT](https://maartengr.github.io/KeyBERT/#about-the-project)

Steps Outlined
* User Input: We take the user's question or prompt as input.
* Keyword Extraction: We analyze the user's input and identify the most important words and phrases (keywords). These keywords represent what the user is interested in within the image.
* Image Segmentation: We use a technique called SAM (Semantic Annotation Model) to break down the image. This process essentially divides the image into its different parts, like objects, backgrounds, and textures.
* Matching Keywords to Image Parts: We use CLIP (Contrastive Language-Image Pre-training) to compare each keyword with each segmented part of the image. CLIP helps us understand how well each part of the image relates to the keywords.

## Usage

1. Download [weights](https://github.com/facebookresearch/segment-anything#model-checkpoints) and place them in this repos root.

2. Install dependencies:
```python
    pip install keybert
    pip install torch opencv-python Pillow
    pip install git+https://github.com/openai/CLIP.git
    pip install git+https://github.com/facebookresearch/segment-anything.git
```
3. Run Notebook `main.ipynb`


## Example

Example output for prompt "kiwi"

![Image with segmentation](assets/example-segmented.png)


[Example Image Source](https://unsplash.com/photos/zeFy-oCUhV8?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink)
