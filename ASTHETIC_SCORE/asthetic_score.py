from PIL import Image
import torch
import pytorch_lightning as pl
import torch.nn as nn
import clip
import torch.nn.functional as F
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def predict_aesthetic_score(image_path):
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("Valid image path must be provided.")
    
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load(os.path.join(os.path.dirname(__file__),"sac+logos+ava1-l14-linearMSE.pth"))
    model.load_state_dict(s)
    model.to("cuda")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   
    pil_image = Image.open(image_path)
    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model2.encode_image(image)
    im_emb_arr = normalized(image_features.cpu().detach().numpy())

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    return prediction.item()

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
        score = predict_aesthetic_score(args.image_path)
        print("Aesthetic score predicted by the model:")
        print(score)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
