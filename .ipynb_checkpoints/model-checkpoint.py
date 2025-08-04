import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        mid_channels = channels // 2
        self.conv1 = nn.Conv2d(channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 通道注意力（更轻量）
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResidualBlock(32)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResidualBlock(64)
        self.attention = ChannelAttention(64)
        
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        feature_dim = 64 * 8 * 8
        hidden_dim = 512
        
        self.fc_hidden = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.attention(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc_hidden(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        feature_dim = 64 * 8 * 8
        hidden_dim = 512
        
        self.fc_expand = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU()
        )
        
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.attention = ChannelAttention(64)
        self.res1 = ResidualBlock(64)
        
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResidualBlock(32)
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        x = self.fc_expand(z)
        x = x.view(-1, 64, 8, 8)
        
        x = self.conv1(x)
        x = self.attention(x)
        x = self.res1(x)
        x = self.deconv1(x)
        x = self.res2(x)
        x = self.deconv2(x)
        
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z = F.normalize(z, p=2, dim=1, eps=1e-8)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z
    


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18Classifier, self).__init__()
        self.model = resnet18(pretrained=False)  
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet18Classifier1(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18Classifier1, self).__init__()
        self.convnet = resnet18(pretrained=False)
        self.convnet.fc = nn.Linear(self.convnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.convnet(x)

