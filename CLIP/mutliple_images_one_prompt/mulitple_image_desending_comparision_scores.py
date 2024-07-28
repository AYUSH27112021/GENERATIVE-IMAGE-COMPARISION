import torch
import clip
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
#my batch of images
images = [r'mutliple_images_one_prompt\car.jpg', r'mutliple_images_one_prompt\dinosaure.jpg', r'mutliple_images_one_prompt\dog.jpg', r'mutliple_images_one_prompt\plane.jpg',
           r'mutliple_images_one_prompt\tree.jpg', r'mutliple_images_one_prompt\woman.jpg']
#what I am searching
text = clip.tokenize(['a man']).to(device)
#create an array of preprocessed images
imgs = [preprocess(Image.open(img)) for img in images]
with torch.no_grad():
  #prediction with one prompt, several images. We need to stack the images together
 logits_per_image, logits_per_text = model(torch.stack(imgs).to(device),text)
 probs = logits_per_text.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs)
print(logits_per_image)
inner_list_probes=probs[0]
for i in inner_list_probes:
 print({'%.8f'},format(float(i)))


#  output:
#  Label probs: [[2.359e-04 4.861e-02 9.463e-01 1.262e-04 4.246e-03 6.022e-04]]
# tensor([[17.2812],
#         [22.6094],
#         [25.5781],
#         [16.6562],
#         [20.1719],
#         [18.2188]], device='cuda:0', dtype=torch.float16)
# tensor([[17.2812, 22.6094, 25.5781, 16.6562, 20.1719, 18.2188]],
#        device='cuda:0', dtype=torch.float16)
# {'%.8f'} 0.0002359151840209961
# {'%.8f'} 0.048614501953125
# {'%.8f'} 0.9462890625
# {'%.8f'} 0.00012624263763427734
# {'%.8f'} 0.004245758056640625
# {'%.8f'} 0.0006022453308105469