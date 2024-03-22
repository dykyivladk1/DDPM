from model import UNet, Diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from PIL import Image

import os
from tqdm import tqdm
from polip import *

import argparse

parser = argparse.ArgumentParser(description = "Train a simple DDPM")
parser.add_argument("--batch_size", type = str, default = 12)
parser.add_argument("--lr", type = float, default = 3e-4)
parser.add_argument("--num_epochs", type = int, default = 50)
parser.add_argument("--num_workers", type = int, default = 8)
parser.add_argument("--device", type = str, default = "mps")
parser.add_argument("--image_dir", type = str, default = "./data")
parser.add_argument("--img_size", type = int, default = 64)
args = parser.parse_args()


device = torch.device(args.device)

model = UNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
mse = nn.MSELoss()
diffusion = Diffusion()
num_epochs = args.num_epochs


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
    
transform = get_rgb_transform(resize = (args.img_size, args.img_size))
ds = CustomImageDataset(image_dir = args.image_dir,
                        transform = transform)

dl = torch.utils.data.DataLoader(ds, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = args.num_workers)


if __name__ == "__main__":
    for epoch in range(num_epochs):
        pbar = tqdm(dl, total = len(dl))
        for images in pbar:
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(MSE = loss.item())
        
        sampled_images = diffusion.sample(model, n = images.shape[0])
        image_save_path = os.path.join("results", "ddpm_conditional")
        os.makedirs(image_save_path, exist_ok=True)
        save_image(sampled_images.float() / 255, os.path.join(image_save_path, f"{epoch}.jpg"))

        model_save_path = os.path.join("models", "ddpm_conditional")
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_save_path, f"ckpt_{epoch}.pt"))

