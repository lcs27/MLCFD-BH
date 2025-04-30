# -*- coding: utf-8 -*-
'''
Autoencoder for image denoising
Copied from https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/ for autoencoder without denoising process

Denoising added by Chensheng Luo
Time: 30 April 2025
'''
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

tensor_transform = transforms.ToTensor()
dataset = datasets.MNIST(root="./ANN/data", train=True, download=False, transform=tensor_transform)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 9)
        )
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AE()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

epochs = 20
outputs = []
losses = []

print("Cuda availiable :", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    for images, _ in loader:
        noise = torch.randn_like(images) * 0.2  # Adding Gaussian noise with standard deviation 0.2
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0., 1.)  # Clamping to ensure pixel values are in [0, 1]
        noisy_images = noisy_images.view(-1, 28 * 28).to(device)
        images = images.view(-1, 28 * 28).to(device)
        
        reconstructed = model(noisy_images)
        loss = loss_function(reconstructed, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    outputs.append((epoch, images, reconstructed))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./ANN/result/AutoEncoder_train.jpg')

model.eval()
dataiter = iter(loader)
images, _ = next(dataiter)

images = images.view(-1, 28 * 28).to(device)
noise = torch.randn_like(images) * 0.2  # Adding Gaussian noise with standard deviation 0.2
noisy_images = images + noise
noisy_images = torch.clamp(noisy_images, 0., 1.)  # Clamping to ensure pixel values are in [0, 1]
reconstructed = model(noisy_images)

fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(10, 6))
for i in range(10):
    axes[0, i].imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(noisy_images[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
    axes[2, i].imshow(reconstructed[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    axes[2, i].axis('off')
fig.savefig('./ANN/result/AutoEncoder_result.jpg')