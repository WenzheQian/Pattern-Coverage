import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from model import VAE

def visualize_random_samples(model, latent_dim, device, num_images=100, save_path="random_samples", weight_path=None):
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        z = F.normalize(z, dim=-1)
        samples = model.decoder(z)
        samples = samples.cpu().numpy()
        fig, axs = plt.subplots(10, 10, figsize=(20, 20))
        for i in range(100):
            row, col = divmod(i, 10)
            axs[row, col].imshow(np.transpose(samples[i], (1, 2, 0)))
            axs[row, col].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "random_samples_epoch_g.png"))
        plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim=128).to(device)
    visualize_random_samples(vae, latent_dim=128, device=device, weight_path="checkpoints/best_vae_qiu.pth")