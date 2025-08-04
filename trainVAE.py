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
import pandas as pd  
from medmnist import OCTMNIST
from torchvision import transforms
from torch.utils.data import DataLoader


class RGBImageDataset(Dataset):
    def __init__(self, image_dir, csv_path, mode='train', transform=None, train_ratio=1.0):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(csv_path, header=0)
        self.transform = transform

        total = len(self.labels_df)
        split_idx = int(total * train_ratio)

        if mode == 'train':
            self.indices = list(range(0, split_idx))
        elif mode == 'test':
            self.indices = list(range(split_idx, total))
        else:
            raise ValueError("mode must be 'train' or 'test'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image_id = f"{real_idx:05d}.png"
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        row = self.labels_df.iloc[real_idx]
        R, G, B = int(row.iloc[0]), int(row.iloc[1]), int(row.iloc[2])
        label = R * 100 + G * 10 + B

        return image, label

class RGBImageDataset1(Dataset):
    def __init__(self, image_dir, csv_path, mode='train', transform=None, train_ratio=1.0):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(csv_path, header=0)
        self.transform = transform

        total = len(self.labels_df)
        split_idx = int(total * train_ratio)

        if mode == 'train':
            self.indices = list(range(0, split_idx))
        elif mode == 'test':
            self.indices = list(range(split_idx, total))
        else:
            raise ValueError("mode must be 'train' or 'test'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image_id = f"{real_idx:06d}.png"
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        row = self.labels_df.iloc[real_idx]
        R, G, B = int(row.iloc[0]), int(row.iloc[1]), int(row.iloc[2])
        label = R * 100 + G * 10 + B

        return image, label

class RGBImageDataset2(Dataset):
    def __init__(self, image_dir, csv_path, mode='train', transform=None, train_ratio=0.8):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(csv_path, header=0)
        self.transform = transform

        total = len(self.labels_df)
        split_idx = int(total * train_ratio)

        if mode == 'train':
            self.indices = list(range(0, split_idx))
        elif mode == 'test':
            self.indices = list(range(split_idx, total))
        else:
            raise ValueError("mode must be 'train' or 'test'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image_id = f"{real_idx:06d}.png"
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        row = self.labels_df.iloc[real_idx]
        R, G, B = int(row.iloc[0]), int(row.iloc[1]), int(row.iloc[2])
        label = R * 100 + G * 10 + B

        return image, label


def spherical_center_loss(z, labels, num_classes=1000, margin=0.5):
   
    device = z.device
    feature_dim = z.size(1)

    class_centers = torch.zeros((num_classes, feature_dim), device=device)
    class_counts = torch.zeros(num_classes, device=device)

    for i in range(z.size(0)):
        label = labels[i]
        class_centers[label] += z[i]
        class_counts[label] += 1

    class_counts[class_counts == 0] = 1  
    class_centers = F.normalize(class_centers / class_counts.unsqueeze(1), p=2, dim=1)

    center_z = class_centers[labels]  
    cos_sim = F.cosine_similarity(z, center_z, dim=1)
    intra_loss = 1 - cos_sim.mean()  

    center_sim = torch.matmul(class_centers, class_centers.T)  
    eye = torch.eye(num_classes, device=device)
    inter_mask = 1 - eye
    inter_loss = F.relu(center_sim * inter_mask - margin).mean()

    return intra_loss + 0.1 * inter_loss


def visualize_reconstruction(model, data_loader, device, num_images=8, save_path="reconstructions5"):
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        x,_ = next(iter(data_loader))
        x = x.to(device)
        x_recon, _, _, _ = model(x)
        print("x min:", x.min().item(), "max:", x.max().item(), "mean:", x.mean().item())
        print("x_recon min:", x_recon.min().item(), "max:", x_recon.max().item(), "mean:", x_recon.mean().item())
        x = x.cpu().numpy()
        x_recon = x_recon.cpu().numpy()

        fig, axs = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
        for i in range(num_images):
            axs[0, i].imshow(np.transpose(x[i], (1, 2, 0)))
            axs[0, i].axis("off")
            axs[1, i].imshow(np.transpose(x_recon[i], (1, 2, 0)))
            axs[1, i].axis("off")
        axs[0, 0].set_title("Original")
        axs[1, 0].set_title("Reconstruction")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "reconstruction_epoch.png"))
        plt.close()

def visualize_random_samples(model, latent_dim, device, num_images=100, save_path="random_samples", weight_path=None):
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        samples = model.decoder(z)
        samples = samples.cpu().numpy()
        fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
        for i in range(num_images):
            axs[i].imshow(np.transpose(samples[i], (1, 2, 0)))
            axs[i].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "random_samples_epoch.png"))
        plt.close()

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    traindataset0 = RGBImageDataset('../VAE-Q/images', '../VAE-Q/labels.csv', mode='train', transform=transform)
    traindataset1 = RGBImageDataset1("../VAE-Q/train_stacked_mnist_1_3_9/images",  '../VAE-Q/train_stacked_mnist_1_3_9/labels.csv' , transform=transform)
    traindataset2 = RGBImageDataset2("../VAE-Q/train_stacked_mnist_pro/images", '../VAE-Q/train_stacked_mnist_pro/labels.csv' , transform=transform)
    traindataset = traindataset0 + traindataset1 + traindataset2
    train_loader = DataLoader(traindataset, batch_size=64, shuffle=True, num_workers=4)

    testdataset = RGBImageDataset2("../VAE-Q/train_stacked_mnist_pro/images", '../VAE-Q/train_stacked_mnist_pro/labels.csv' , mode='test' , transform=transform)
    test_loader = DataLoader(testdataset, batch_size=64, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim=128).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    kl_lamda = 0.2
    best_loss = float('inf')

    for epoch in range(1, 101):
        vae.train()
        total_loss = 0
        for x,labels in train_loader:
            x = x.to(device)
            x_recon, mu, logvar, z = vae(x)
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            center_loss = spherical_center_loss(z, labels, num_classes=1000, margin=0.5)
            loss = recon_loss + kl_lamda * kl_loss + 0.2 * center_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(vae.state_dict(), "checkpoints/best_vae_qiuqiu.pth")
            print(f"Best model saved at epoch {epoch} with loss {avg_loss:.4f}")

        visualize_reconstruction(vae, test_loader, device, num_images=8, save_path="reconstructions5")
