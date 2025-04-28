# -*- coding: utf-8 -*-
'''
Homework code - solution of heat equation using explicit finite difference method
Author: Dr QIU Lu
Commented & modified by: LUO Chensheng 
Time: 28 April 2025
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Random set
torch.manual_seed(42)
np.random.seed(42)

# Generate training data for cylinder flow
def generate_data(num_samples = 200):
    '''
    Input:
        num_samples: number of samples to generate
    Output:
        x, y, t: coordinates and time, random distribution, shape (num_samples)
        u, v, p: velocity and pressure fields, shape (num_samples)
    '''
    x = np.random.uniform(-2, 2, num_samples).astype(np.float32)
    y = np.random.uniform(-2, 2, num_samples).astype(np.float32)
    t = np.random.uniform(0, 10, num_samples).astype(np.float32)
    
    u = np.sin(x + y + t).astype(np.float32)
    v = np.cos(x - y + t).astype(np.float32)
    p = (np.sin(x) * np.cos(y) * np.sin(t)).astype(np.float32)
    
    return x, y, t, u, v, p

# PINN model 
class PINN(nn.Module):
    '''
    3 - tanh - 20 - tanh - 20 - tanh - 20 - tanh - 20 - 3
    '''
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(3, 20),  # 输入层：x, y, t
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 3)  # 输出层：u, v, p
        )

    def forward(self, x, y, t):
        input = torch.cat([x, y, t], dim=1)
        output = self.hidden_layers(input)
        return output[:, 0:1], output[:, 1:2], output[:, 2:3]  # u, v, p

# PDE Loss function
def compute_physics_loss(model, x, y, t, nu):
    '''
    Input:
        model: PINN model
        x, y, t: coordinates and time, shape (num_samples)
        nu: PDE parameter(kinematic viscosity)
    Output:
        physics_loss: PDE loss, real
    '''
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    u, v, p = model(x, y, t)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    # Momentum equations
    momentum_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    # Continuity equation
    continuity = u_x + v_y

    # PDE Loss should be sum of these three equations
    physics_loss = torch.mean(momentum_u**2 + momentum_v**2 + continuity**2)
    return physics_loss

# Training function
def train_pinn(model):
    # Random measured data
    x, y, t, u_true, v_true, p_true = generate_data()
    x = torch.tensor(x.reshape(-1, 1), requires_grad=True)
    y = torch.tensor(y.reshape(-1, 1), requires_grad=True)
    t = torch.tensor(t.reshape(-1, 1), requires_grad=True)
    u_true = torch.tensor(u_true.reshape(-1, 1))
    v_true = torch.tensor(v_true.reshape(-1, 1))
    p_true = torch.tensor(p_true.reshape(-1, 1))

    # Initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    nu = 0.01  # 流体的运动粘度

    # Training
    for epoch in range(1000):
        optimizer.zero_grad()

        # Forward and Losses
        ## PDE loss
        physics_loss = compute_physics_loss(model, x, y, t, nu) # PDE loss is computed randomly
        ## Data loss
        u_pred, v_pred, p_pred = model(x, y, t)
        data_loss = torch.mean((u_pred - u_true)**2 + (v_pred - v_true)**2 + (p_pred - p_true)**2)
        
        total_loss = physics_loss + data_loss

        # Backward and optimisation
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Total Loss: {total_loss.item():.4f}, '
                  f'Physics Loss: {physics_loss.item():.4f}, Data Loss: {data_loss.item():.4f}')

    return model

if __name__ == "__main__":
    # PINN training
    model = PINN()
    train_pinn(model)
    model.eval()

    # Visualize the results
    x = np.linspace(-2, 2, 100).astype(np.float32)
    y = np.linspace(-2, 2, 100).astype(np.float32)
    t = np.linspace(0, 10, 100).astype(np.float32)
    X, Y, T = np.meshgrid(x, y, t)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    T = T.reshape(-1, 1)

    # Prediction
    with torch.no_grad():
        u_pred, v_pred, p_pred = model(
            torch.tensor(X, requires_grad=True),
            torch.tensor(Y, requires_grad=True),
            torch.tensor(T, requires_grad=True)
        )

    # 将预测结果转换为numpy数组
    u_pred = u_pred.numpy()
    v_pred = v_pred.numpy()
    p_pred = p_pred.numpy()

    print("Data is calculated.")
    
    # Visualization
    feature_times = [0.0, 2.5, 5.0, 7.5, 10.0]
    x = np.linspace(-2, 2, 100).astype(np.float32)
    y = np.linspace(-2, 2, 100).astype(np.float32)
    X, Y = np.meshgrid(x, y)

    fig, axs = plt.subplots( 2, len(feature_times), figsize=( 4 * len(feature_times), 6.5))
    for i, t_val in enumerate(feature_times):
        t = np.full((100, 100), t_val).astype(np.float32)
        T = t.reshape(-1, 1)
        X_flat = X.reshape(-1, 1)
        Y_flat = Y.reshape(-1, 1)

        with torch.no_grad():
            u_pred, v_pred, _ = model(
                torch.tensor(X_flat, requires_grad=True),
                torch.tensor(Y_flat, requires_grad=True),
                torch.tensor(T, requires_grad=True)
            )

        u_pred = u_pred.numpy().reshape(100, 100)
        v_pred = v_pred.numpy().reshape(100, 100)

        ax = axs[0,i]
        contour = ax.contourf(X, Y, u_pred, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax)
        ax.set_title(f'Velocity u at t = {t_val}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax = axs[1,i]
        contour = ax.contourf(X, Y, v_pred, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax)
        ax.set_title(f'Velocity v at t = {t_val}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.tight_layout()
    plt.savefig('./PINN/result/cylinder_flow_pinn.png', dpi=300)