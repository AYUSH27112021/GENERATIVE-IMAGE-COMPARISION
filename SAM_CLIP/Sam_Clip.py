from os.path import dirname, abspath
import sys
sys.path.append(abspath(dirname(__file__)) + r"\SAM_CLIP")
import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import clip
import torch
import numpy as np
from KeyBertTest import extract_only_keywords
from Metric_score import adjust_score
import os
import warnings
warnings.filterwarnings("ignore")


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = r"sam_vit_h_4b8939.pth"

# Initialize SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=os.path.join(os.path.dirname(__file__),CHECKPOINT_PATH))
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

# Load CLIP model
model, preprocess = clip.load("ViT-L/14", device=DEVICE)

@torch.no_grad()
def retriev(elements, search_text):
    preprocessed_images = [preprocess(image).to(DEVICE) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(DEVICE)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

def visualize(cropped_boxes, text):
    scores = retriev(cropped_boxes, text)
    images_scores = sorted(scores.cpu().numpy(), reverse=True)
    return adjust_score(images_scores, threshold=0.3, initial_score=0.5)

def Sam_Clip(image_path, prompt):
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("Valid image path must be provided.")
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Valid prompt string must be provided.")
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    
    image_pil = Image.open(image_path)
    total_area = image_pil.size[0] * image_pil.size[1]
    min_area_threshold = total_area * 0.05
    cropped_boxes = []

    for mask in masks:
        mask_area = np.sum(mask["segmentation"])
        if mask_area >= min_area_threshold:
            cropped_boxes.append(segment_image(image_pil, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))

    key_prompts = extract_only_keywords(prompt)
    final_scores = [visualize(cropped_boxes, kp) for kp in key_prompts]
    return final_scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process an image and a prompt using SAM and CLIP models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image_path", 
        type=str, 
        help="Path to the image file to be processed."
    )
    parser.add_argument(
        "prompt", 
        type=str, 
        help="Prompt string to process with the image."
    )
    
    args = parser.parse_args()
    
    try:
        final_scores = Sam_Clip(args.image_path, args.prompt)
        print("Final Scores:", final_scores)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
