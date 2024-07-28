import requests
import torch
from transformers import AutoModelForCausalLM
import warnings
from PIL import Image
import sys
import os

warnings.filterwarnings("ignore")

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "q-future/one-align", 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

def get_image_scores(image_path):
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("Valid image path must be provided.")
    """
    This function takes an image path as input, calculates the quality and aesthetics scores using the model,
    and returns these scores.
    """
    image = Image.open(image_path)
    
    # Get quality score
    q_align_score_quality = model.score([image], task_="quality", input_="image")
    
    # Get aesthetics score
    q_align_score_aesthetics = model.score([image], task_="aesthetics", input_="image")
    
    return q_align_score_quality.item(), q_align_score_aesthetics.item()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process an image and a string using various metrics and return either individual results or a final score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image_path", 
        type=str, 
        help="Path to the image file to be processed."
    )

    args = parser.parse_args()
    
    try:
        quality_score,aesthetics_score = get_image_scores(args.image_path)
        print("Quality score :",quality_score)
        print("Aesthetic score :",aesthetics_score)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
