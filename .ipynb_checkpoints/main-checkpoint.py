import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from reflected_flow import ReflectedFlowModel , SphericalFlowModel
from trainer import FlowTrainer
from model import VAE 


class TrainImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, f"{i:05d}.png") for i in range(00000, 59999)
            if os.path.exists(os.path.join(root_dir, f"{i:05d}.png"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def setup_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    traindataset = TrainImageDataset(root_dir="../VAE-Q/images", transform=transform)
    train_loader = DataLoader(traindataset, batch_size=64, shuffle=True, num_workers=4)


    return train_loader

def main():

    parser = argparse.ArgumentParser(description='Reflected Flow Training in VAE Latent Space')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--flow_timesteps', type=int, default=200, help='Flow timesteps')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for flow training')
    parser.add_argument('--device', type=str, default='cuda', help='Device (auto, cpu, cuda)')

    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    os.makedirs("results", exist_ok=True)
    
    train_loader = setup_data()
    

    vae = VAE(latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load("./checkpoints/best_vae_qiu.pth", map_location=device))
    print("VAE loaded successfully!")
    
    flow_model = SphericalFlowModel(latent_dim=args.latent_dim).to(device)

    print(f"Flow model created with {sum(p.numel() for p in flow_model.parameters()):,} parameters")
    
    trainer = FlowTrainer(vae, flow_model, device)
    
    print("=" * 60)
    print("Starting Reflected Flow Training in VAE Latent Space")
    print("=" * 60)
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Flow timesteps: {args.flow_timesteps}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    trainer.train(data_loader=train_loader,epochs=args.epochs,batch_size=args.batch_size)
    
    print("=" * 60)
    print("Final Evaluation and Visualization")
    print("=" * 60)
    
    trainer.visualize_generation(num_samples=64, save_path="results/final_generated_samples.png", save_separately=False)
    trainer.plot_losses(save_path="results/training_losses.png")
    trainer.save_model("final_flow_q.pth") 
    
    print("\nTraining completed successfully!")
    print("Generated samples saved in 'results/final_generated_samples.png'")
    print("Training loss curve saved in 'results/training_losses.png'")
    print("Final model saved in 'checkpoints/final_flow_q.pth'")

if __name__ == "__main__":
   
    main()