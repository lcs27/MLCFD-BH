# -*- coding: utf-8 -*-
'''
GAN(Generative Adversarial Network) for handwriting generation based on MNIST database
Author: Dr QIU Lu
Commented & modified by: LUO Chensheng 
Time: 25 May 2025
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置超参数
batch_size = 128
learning_rate = 0.0002
num_epochs = 100
latent_dim = 100  # latent dimension for generator
img_size = 28
channels = 1

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalisation
])

train_dataset = datasets.MNIST(root='./ANN/data', train=True, transform=transform, download=True)
print("Download sucessful!")

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
print("Data loaded.")

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)  # 用于条件生成，将标签嵌入到高维空间
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),  # Input: random noise on latent_dim + feature 
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_size * img_size),  # 输出是28×28的图像
            nn.Tanh()  # 输出范围在[-1,1]
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        z = torch.cat([z, c], 1)  # 将噪声和标签嵌入拼接
        img = self.model(z)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size + 10, 512),  # 输入是图像 + 标签嵌入
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        img_flat = img.view(img.size(0), -1)  # 将图像展平
        img_flat = torch.cat([img_flat, c], 1)  # 拼接图像和标签嵌入
        prob = self.model(img_flat)
        return prob

# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()
# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# 训练过程
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):

        # Generator of real image and fake image, labeling
        batch_size = imgs.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Discriminator traning
        optimizer_d.zero_grad()
        ## True images
        outputs = discriminator(imgs, labels)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        ## Fake images
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z, labels)
        outputs = discriminator(fake_imgs.detach(), labels) 
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        ## Total loss and optimisation
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Generator training
        optimizer_g.zero_grad()
        ## Produce fake images
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z, labels)
        outputs = discriminator(fake_imgs, labels)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

    # 可视化生成的图像
    if (epoch + 1) % 10 == 0:
        z = torch.randn(10, latent_dim)  # 生成10个图像
        labels = torch.arange(10)  # 生成0 -9的标签
        generated_imgs = generator(z, labels)
        plt.figure(figsize=(10, 1))
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(generated_imgs[i, 0].detach().numpy(), cmap='gray')
            plt.axis('off')
        plt.savefig(f'./ANN/result/GAN_generated_images_epoch_{epoch + 1}.png')