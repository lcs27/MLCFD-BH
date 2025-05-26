# -*- coding: utf-8 -*-
'''
VAE(Variational Autoencoder) for handwriting generation based on MNIST database
Author: Dr QIU Lu
Commented & modified by: LUO Chensheng 
Time: 25 May 2025
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
batch_size = 128
latent_dim = 2
epochs = 50
learning_rate = 1e-3

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 将图像展平为向量
])

train_dataset = datasets.MNIST(root='./ANN/data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义变分自编码器
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # 均值
        self.fc22 = nn.Linear(400, latent_dim)  # 方差
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):  #VAE的关键部分，在得到的均值和方差上引入随机性，从而生成潜在空间的采样点
        std = torch.exp(0.5 * logvar) #计算方差的平方根（方差是以对数形式存储的）
        eps = torch.randn_like(std)   #生成一个与均值和方差同形状的随机噪声
        return mu + eps * std         #均值和噪声相加并乘以标准差，得到采样点
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL散度
    KLD = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar)
    
    return BCE + KLD

# 初始化模型和优化器
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# 定义生成函数
def generate_images(n_samples=64):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        samples = model.decode(z)
    return samples.cpu().numpy().reshape(-1, 1, 28, 28)

# 可视化生成的图像
def plot_generated_images(samples):
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.savefig(f'./ANN/result/VAE_generated_images.png')

# 设置开关
train_switch = 0  # 设置为True进行训练，设置为False加载已保存的模型
path = './ANN/model/vae_model.pth'
if train_switch:
    # 训练模型
    for epoch in range(1, epochs + 1):
        train(epoch)
    
    # 保存模型
    torch.save(model.state_dict(), path)
else:
    # 加载已保存的模型
    model.load_state_dict(torch.load(path))

# 生成并显示图像
generated_samples = generate_images()
plot_generated_images(generated_samples)