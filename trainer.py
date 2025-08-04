import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from reflected_flow import ReflectedFlowModel
from sklearn.manifold import TSNE

class FlowTrainer:
    def __init__(self, vae, flow_model, device, save_dir="checkpoints"):
        self.vae = vae
        self.flow_model = flow_model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        
        self.flow_optimizer = optim.AdamW(
            flow_model.parameters(), 
            lr=2e-4, 
            weight_decay=1e-6
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.flow_optimizer, 
            T_max=100, 
            eta_min=1e-6
        )
        
        self.losses = []
    
    def extract_latent_codes(self, data_loader):

        print("Extracting latent codes from VAE...")
        latent_codes = []
        
        with torch.no_grad():
            for x in tqdm(data_loader, desc="Extracting latents"):
                x = x.to(self.device)
                _, mu, _, z = self.vae(x)  
                latent_codes.append(z.cpu())
        
        latent_codes = torch.cat(latent_codes, dim=0)
        z_min = latent_codes.min(dim=0)[0]
        z_max = latent_codes.max(dim=0)[0]
        print(f"Per-dimension min:\n{z_min}")
        print(f"Per-dimension max:\n{z_max}")
        norms = latent_codes.norm(dim=1)
        print(f"Norm stats — min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}, std: {norms.std():.4f}")
        print(f"Extracted {latent_codes.shape[0]} latent codes with dimension {latent_codes.shape[1]}")
        return latent_codes
    
    def train_epoch(self, latent_loader):
        
        self.flow_model.train()
        total_loss = 0
        num_batches = len(latent_loader)
        
        for batch_idx, (z,) in enumerate(tqdm(latent_loader, desc="Training Flow")):
            z = z.to(self.device)
            
            
            loss = self.flow_model(z)
            
            self.flow_optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
            
            self.flow_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        return avg_loss
    
    def train(self, data_loader, epochs=100, batch_size=256):
        print("Starting Reflected Flow training...")
        
        # 提取潜在编码
        latent_codes = self.extract_latent_codes(data_loader)
        
        # 创建潜在空间数据集
        latent_dataset = TensorDataset(latent_codes)
        latent_loader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        best_loss = float('inf')
        patience = 20
        no_improve = 0
        
        for epoch in range(1, epochs + 1):
            # 训练一个epoch
            avg_loss = self.train_epoch(latent_loader)
            self.scheduler.step()
            
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model("best_flow_oushi.pth")
                no_improve = 0
                print(f"Best model saved with loss: {avg_loss:.6f}")
            else:
                no_improve += 1
            
            # 早停
            if no_improve >= patience:
                print(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break
            
            # 定期可视化和保存
            if epoch % 20 == 0:
                self.visualize_generation(save_path=f"results/flow_epoch_{epoch}.png")
                self.save_model(f"flow_epoch_{epoch}_oushi.pth")
        
        # 加载最佳模型
        self.load_model("best_flow_oushi.pth")
        print("Training completed!")
    
    @torch.no_grad()
    def generate_samples(self, num_samples=16):
        """生成新样本"""
        self.flow_model.eval()
        
        z_samples = self.flow_model.sample(num_samples, self.device)
        
        x_samples = self.vae.decoder(z_samples)
        
        return x_samples, z_samples
    
    @torch.no_grad()
    
    
    @torch.no_grad()
    def visualize_generation(self, num_samples=16, save_path="generated_samples.png", save_separately=False):
        """可视化生成的样本"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        x_samples, _ = self.generate_samples(num_samples)
        # x_samples = (x_samples + 1) / 2
        x_samples = x_samples.cpu().numpy()
        #################
        if save_separately:
            separate_dir = "generated_samples"
            os.makedirs(separate_dir, exist_ok=True)
            for i in range(num_samples):
                sample = x_samples[i]
                if sample.shape[0] == 1:  # 单通道
                    plt.imshow(sample[0], cmap="gray")
                else:  # 多通道
                    plt.imshow(np.transpose(sample, (1, 2, 0)))
                plt.axis("off")
                plt.savefig(os.path.join(separate_dir, f"sample_{i:04d}.png"), dpi=150, bbox_inches='tight')
                plt.close()
            print(f"Individual samples saved to {separate_dir}")
        ############
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i in range(num_samples):
            sample = x_samples[i]
            if sample.shape[0] == 1:  # 单通道
                axes[i].imshow(sample[0], cmap="gray")
            else:  # 多通道
                axes[i].imshow(np.transpose(sample, (1, 2, 0)))
            axes[i].axis("off")
        
        # 隐藏多余的子图
        for i in range(num_samples, len(axes)):
            axes[i].axis("off")
        
        plt.suptitle("Generated Samples", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Generated samples saved to {save_path}")
    
    def visualize_generation_separately(self, num_samples=16, save_path="generated_samples.png", save_separately=False):
        """可视化生成的样本"""
        x_samples, _ = self.generate_samples(num_samples)
        x_samples = x_samples.cpu().numpy()
        #################
        if save_separately:
            separate_dir = "generated_samples9"
            os.makedirs(separate_dir, exist_ok=True)
            for i in tqdm(range(num_samples), desc="Saving samples"):
                sample = x_samples[i]
                # sample = (sample + 1) / 2
                if sample.shape[0] == 1:  # 单通道
                    plt.imshow(sample[0], cmap="gray")
                else:  # 多通道
                    plt.imshow(np.transpose(sample, (1, 2, 0)))
                plt.axis("off")
                plt.savefig(os.path.join(separate_dir, f"samples_{i:05d}.png"), dpi=150, bbox_inches='tight')
                plt.close()
            print(f"Individual samples saved to {separate_dir}")
    
    
    
    def plot_losses(self, save_path="training_losses.png"):
        """绘制训练损失曲线"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, linewidth=2)
        plt.title("Reflected Flow Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved to {save_path}")
    
    def save_model(self, filename):
        """保存Flow模型"""
        path = os.path.join(self.save_dir, filename)
        torch.save(self.flow_model.state_dict(), path)
    
    def load_model(self, filename):
        """加载Flow模型"""
        path = os.path.join(self.save_dir, filename)
        self.flow_model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Flow model loaded from {path}")