import torch
import clip
from PIL import Image
import os
import sys

def compute_image_text_similarity(image_path, text_input):
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("Valid image path must be provided.")
    if not text_input or not isinstance(text_input, str):
        raise ValueError("Valid text input must be provided.")

    # Determine the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and preprocessor
    model, preprocess = clip.load("ViT-L/14", device=device)

    # Preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Generate the tokenizer for the text input
    text = clip.tokenize([text_input]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # Compute score between the image and the text
        logits_per_image, logits_per_text = model(image, text)
        
        # Get score with softmax
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    inner_list_probs = probs[0]
    formatted_probs = [float(f'{i:e}') for i in inner_list_probs]
    
    return  logits_per_image.cpu().numpy().tolist()[0][0]/100

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute the similarity between an image and a text input using CLIP model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image_path", 
        type=str, 
        help="Path to the image file to be processed."
    )
    parser.add_argument(
        "text_input", 
        type=str, 
        help="Text input to compare with the image."
    )
    
    args = parser.parse_args()
    
    try:
        result = compute_image_text_similarity(args.image_path, args.text_input)
        print("Image - text Similarity :", result)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
