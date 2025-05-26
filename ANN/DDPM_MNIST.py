# -*- coding: utf-8 -*-
'''
DDPM(Denoising Diffusion Probabilistic Models) for handwriting generation based on MNIST database
Author: Dr QIU Lu
Commented & modified by: LUO Chensheng 
Time: 25 May 2025
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Parameter
batch_size = 128
image_size = 28
channels = 1
timesteps = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build Unet Network
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Define DDPM
class DDPM(nn.Module):
    def __init__(self, unet, timesteps=timesteps):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps

        # For each step, we add noise of std beta
        self.beta = torch.linspace(0.0001, 0.02, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    # Forward diffusion (add noise)
    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        xt = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise
        return xt, noise

    # Loss between U-NET predicted noise and real noise
    def loss(self, x0):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,)).to(device)
        xt, noise = self.forward_diffusion(x0, t) #x0 add noise to get xt
        predicted_noise = self.unet(xt, t) # use xt to predict the added noise
        return F.mse_loss(predicted_noise, noise)

    # Backwork diffusion(denoise)
    def generate(self, num_samples):
        with torch.no_grad():
            # Beginning is random with num_samples
            x = torch.randn(num_samples, channels, image_size, image_size).to(device)
            for t in range(self.timesteps-1, -1, -1):
                t_batch = torch.full((num_samples,), t, dtype=torch.long).to(device)
                predicted_noise = self.unet(x, t_batch) # The predicted noise of x(t-1) to xt
                alpha_t = self.alpha[t].view(-1, 1, 1, 1)
                alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                x = (x - (1 - alpha_t)/torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)
                if t > 0:
                    x += torch.sqrt(beta_t) * torch.randn_like(x)
        return x.cpu()

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./ANN/data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

unet = UNet().to(device)
ddpm = DDPM(unet)
optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-3)

# Model training
train_flag = 0  # 设置为True训练模型，设置为False从保存的模型中加载
num_epochs = 50
path = './ANN/model/ddpm.pth'
if train_flag:
    for epoch in range(num_epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            loss = ddpm.loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
                
        torch.save(ddpm.state_dict(), path)
        print("Model is saved.")
else:
    # Use pretrained model
    ddpm.load_state_dict(torch.load(path))
    print("The model is loaded.")

# Generate samples
samples = ddpm.generate(16)
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(samples[i].permute(1, 2, 0).squeeze().numpy(), cmap='gray')
    plt.axis('off')
plt.savefig('./ANN/result/DDPM_generated_samples.png')