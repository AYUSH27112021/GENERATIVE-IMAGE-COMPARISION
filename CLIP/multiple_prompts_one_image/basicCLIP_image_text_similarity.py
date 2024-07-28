#importing required modules
import torch
import clip
from PIL import Image
#gpu or cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
#Load the model and preprocessor
model, preprocess = clip.load("ViT-B/32", device=device)
print(device)
#Preprocess the image
image = preprocess(Image.open(r"multiple_prompts_one_image\dog.jpg")).unsqueeze(0).to(device)
#Generate the tokenizer for 1 class.
# text = clip.tokenize(["a dog"]).to(device)
text = clip.tokenize(["a dog in a green background"]).to(device)
with torch.no_grad():
  image_features = model.encode_image(image)
  text_features = model.encode_text(text)
  #Compute score between the image
  logits_per_image, logits_per_text = model(image, text)
  #get score with softmax
  probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#Print scores
print("Image-Text-Similarity:",logits_per_image)
print("Label probs:", probs)
inner_list_probes=probs[0]
for i in inner_list_probes:
 print({'%.8f'},format(float(i)))