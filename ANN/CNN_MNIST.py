# -*- coding: utf-8 -*-
'''
CNN(Convolutional Neural Network) for handwriting recognition based on MNIST database detect
by Chensheng Luo
Time: 26 May 2025
'''
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Parameters
batch_size = 128
learning_rate = 0.0002
epochs = 20
img_size = 28 
channels = 1 

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 标准化到[-1,1]
])
dataset = datasets.MNIST(root="./ANN/data", train=True, download=False, transform=tensor_transform)
print("Download sucessful!")

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(10,10,kernel_size=5,stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) #2x2 maxpool
        self.fc1 = nn.Linear(4*4*10,100)
        self.fc2 = nn.Linear(100,10)
  
    def forward(self,x):
        x = F.relu(self.conv1(x)) #24x24x10
        x = self.pool(x) #12x12x10
        x = F.relu(self.conv2(x)) #8x8x10
        x = self.pool(x) #4x4x10    
        x = x.view(-1, 4*4*10) #flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    # def __init__(self):
    #     super().__init__()
    #     self.network = nn.Sequential(
    #         nn.Conv2d(1,10,kernel_size=5,stride=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2,stride=2),

    #         nn.Conv2d(10,10,kernel_size=5,stride=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2,stride=2),

    #         nn.Flatten(),
    #         nn.Linear(4*4*10,100),
    #         nn.Linear(100,10),
    #     )
  
    # def forward(self,x):
    #     return self.network(x)

model = CNN()

# Loss and optimisation
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 20
losses = []

print("Cuda availiable :", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        labels_one_hot = F.one_hot(labels, num_classes=10).float()
        loss = loss_function(outputs, labels_one_hot)
        # It is also OK to use: 
        #   loss = loss_function(outputs, labels)
        #   loss = loss_function(labels_one_hot,outputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./ANN/result/CNN_train.jpg')

model.eval()
dataiter = iter(loader)
images, labels  = next(dataiter)
images = images.to(device)
labels = labels.to(device)
results = model(images)

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 6))
for n in range(20):
    i = n//5
    j = n%5
    axes[i, j].imshow(images[n].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    axes[i, j].set_title(f'Judged: {str(results[n].argmax().item())} + True: {labels[n]}')
    axes[i, j].axis('off')
    axes[i, j].title.set_fontsize(10)
fig.savefig('./ANN/result/CNN_result.jpg')