from reflected_flow import ReflectedFlowModel
from trainer import FlowTrainer
import torch
from model import VAE


device = torch.device('cuda')
flow_model = ReflectedFlowModel(latent_dim=128).to(device)
flow_model.load_state_dict(torch.load("./checkpoints/best_flow_qiu.pth", map_location=device))
print("Reflected Flow Model loaded successfully!")
flow_model.eval()

vae = VAE(latent_dim=128).to(device)
vae.load_state_dict(torch.load("./checkpoints/best_vae_qiu.pth", map_location=device))
print("VAE loaded successfully!")
vae.eval()

trainer = FlowTrainer(vae, flow_model, device)
trainer.visualize_generation_separately(num_samples=150000, save_path="results/final_generated_samples.png", save_separately=True)
# trainer.visualize_generation(num_samples=64, save_path="random_samples/generated_samples.png", save_separately=False)