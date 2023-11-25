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
#Generate the tokenizer for our 5 classes
text = clip.tokenize(["a dog", "a cat","a man","a tree", "food"]).to(device)

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
 print(float('{:e}'.format(i)))


# Output:
# Image-Text-Similarity: tensor([[25.5781, 20.1719, 19.7812, 17.2812, 20.9531]], device='cuda:0',
#        dtype=torch.float16)
# Label probs: [[9.829e-01 4.410e-03 2.985e-03 2.451e-04 9.636e-03]]
# {'%.8f'} 0.98291015625
# {'%.8f'} 0.0044097900390625
# {'%.8f'} 0.0029850006103515625
# {'%.8f'} 0.00024509429931640625
# {'%.8f'} 0.00963592529296875