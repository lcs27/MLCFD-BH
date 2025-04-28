# -*- coding: utf-8 -*-
'''
Homework code - solution of heat equation using PINN
Author: Dr QIU Lu
Commented & modified by: LUO Chensheng 
Time: 28 April 2025
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


## PINN model
class HeatPINN(nn.Module):
    '''
    2 - tanh - 64 - tanh - 64 - 1
    '''
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

## PDE loss calculation
def heat_residual(u_pred, x, t, alpha):
    '''
    Input:
        u_pred: predicted temperature distribution, array of shape (nt, nx)
        x: spatial coordinates, array of shape (nx)
        t: time coordinates, array of shape (nt)
        alpha: equation parameter
    Output:
        residual: PDE residual, real
    '''
    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    residual = u_t - alpha * u_xx
    return torch.mean(residual**2)


## Initial and boundary conditions definition
def ICBC_define(a, b, c, d, e):
    '''
    Input:
        a, b, c, d, e: parameters
    Output:
        u_IC: initial temperature distribution, array of shape (nx)
        u_BC_left: left boundary temperature, array of shape (nt)
        u_BC_right: right boundary temperature, array of shape (nt)
    '''
    #初始条件及边界条件
    
    #u_IC = torch.sin(torch.pi * x)  # 初始时刻温度分布
    u_IC = torch.full((Data_Density,1), a)  # 初始时刻温度分布
    u_BC_left = torch.sin(torch.pi * t_BC * d) #torch.full((Data_Density,1), 1.0)  # 左边界温度随时间变化
    #u_BC_right = torch.sin(-torch.pi * t_BC * e) #torch.full((Data_Density,1), 1.0)  #右边界温度随时间变化
    #u_BC_left = torch.full((Data_Density,1), b)  # 左边界温度随
    u_BC_right = torch.full((Data_Density,1), c)  #右边界温度随
    
    return u_IC, u_BC_left, u_BC_right
    
    
## Training function
def train_pinn(num_epochs, u_initial, u_bounadary_left, u_bounadary_right):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):  

        ## Step 1: forward and Loss functions
        # PDE Loss    
        u_pred = model(x_in, t_in)
        loss_pde = heat_residual(u_pred, x_in, t_in, alpha)

        # IC Loss
        u_initial_pred = model(x_IC, torch.zeros_like(x_IC))
        loss_ic = torch.mean((u_initial_pred - u_initial)**2)

        # BC Loss
        u_boundary_left_pred = model(torch.full_like(t_BC, x_min), t_BC)
        u_boundary_right_pred = model(torch.full_like(t_BC, x_max), t_BC)     
        loss_bc = torch.mean((u_boundary_left_pred - u_bounadary_left)**2 + (u_boundary_right_pred - u_bounadary_right)**2)

        total_loss = loss_pde + loss_ic + loss_bc
        
        # Data Loss (if using cut window)
        if Cut_window_flag == True:
            u_wd_pred = model(x_wd, t_wd)
            loss_data = torch.mean((u_wd_pred - u_wd)**2)
            total_loss = total_loss + loss_data
        

        ## Step 2: Backward and optimization
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            #print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}, PDE Loss: {loss_pde:.4f}, IC Loss: {loss_ic:.4f}, BC Loss: {loss_bc:.4f}, Data Loss: {loss_data:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}, "
                  f"PDE Loss: {loss_pde:.4f}, IC Loss: {loss_ic:.4f}, BC Loss: {loss_bc:.4f}")
        
    return model

## Predict the whole field using PINN
def Pred_FullField(X_p, T_p):
    ''' 
    Input:
        X_p: spatial coordinates, array of shape (nx, nt)
        T_p: time coordinates, array of shape (nx, nt)
        X_p, T_p should usually be a meshgrid of spatial and time coordinates
    Output:
        U_pred: predicted temperature distribution, array of shape (nx, nt)
    '''
    x_flat = X_p.flatten()
    t_flat = T_p.flatten()
    x_tensor = torch.tensor(x_flat, dtype=torch.float32).reshape(-1, 1)
    t_tensor = torch.tensor(t_flat, dtype=torch.float32).reshape(-1, 1)
    u_pred = model(x_tensor, t_tensor).detach().numpy()
    U_pred = u_pred.reshape(X_p.shape)
    return U_pred

    
# Visualize the results
def visualize_results(model, ShowMarkers):
    x_p = np.linspace(x_min, x_max, 100)
    t_p = np.linspace(t_min, t_max, 100)
    X_p, T_p = np.meshgrid(x_p, t_p)
    
    U_pred = Pred_FullField(X_p, T_p)
    
    plt.figure(figsize=(3, 2.5*(t_max-t_min)/(x_max-x_min) ))
    plt.contourf(X_p, T_p, U_pred, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Temperature')
    plt.title('Temperature Distribution')
    plt.xlabel('Position (x)')
    plt.ylabel('Time (t)')
    
    if ShowMarkers == True:
        plt.scatter(x_in.detach().numpy(),t_in.detach().numpy(),color='red',marker='+')
        plt.scatter(torch.full_like(t_BC, x_min),t_BC,color='black',marker='>')
        plt.scatter(torch.full_like(t_BC, x_max),t_BC,color='black',marker='<')
        plt.scatter(x_IC,torch.zeros_like(x_IC),color='yellow',marker='^')
        
        if Cut_window_flag == True:
            plt.scatter(x_wd.detach().numpy(), t_wd.detach().numpy(), color='white',marker='D')
        
    plt.savefig('./PINN/result/PINN_heat_equation.png', dpi=600)
    return U_pred



def Sampling_timespace(D_Dens):
    '''
    Sampling points for PDE loss, random sampling
    '''
    Data_Density_in = D_Dens*D_Dens
    x_in = torch.rand(Data_Density_in, 1, requires_grad=True) * (x_max - x_min) + x_min
    t_in = torch.rand(Data_Density_in, 1, requires_grad=True) * (t_max - t_min) + t_min
    return x_in, t_in

def Cut_window(D_Dens, x_cut, y_cut, u_val):
    '''
    Measured data (here given as random)
    '''
    Data_Density_in = D_Dens*D_Dens
    x_min, x_max, t_min, t_max = x_cut, x_cut+0.1, y_cut, y_cut+0.1
    x_wd = torch.rand(Data_Density_in, 1, requires_grad=True) * (x_max - x_min) + x_min
    t_wd = torch.rand(Data_Density_in, 1, requires_grad=True) * (t_max - t_min) + t_min
    u_wd = torch.full_like(x_wd, u_val)
    return x_wd, t_wd, u_wd

def Sampling_ICBC(D_Dens):
    '''
    Sampling points for initial and boundary conditions, uniform sampling
    '''
    x = torch.linspace(x_min, x_max, D_Dens).unsqueeze(1)
    t = torch.linspace(t_min, t_max, D_Dens).unsqueeze(1)
    return x, t
    
if __name__ == "__main__":
    
    ## Parameters
    # PINN parameters
    Data_Density = 10   # 
    num_epochs = 2000    # 
    Train_model = 1      
    Cut_window_flag = 0

    # PDE parameters
    x_min, x_max = 0, 1
    t_min, t_max = 0, 3
    alpha = 0.05
        
    ## Sampling points
    x_IC, t_BC = Sampling_ICBC(Data_Density)  # IC BC loss sampling points
    x_in, t_in = Sampling_timespace(Data_Density) # PDE loss sampling points

    ## ICBC condistion set
    u_IC, u_BC_left, u_BC_right = ICBC_define(0, 1, 1, 1, 1)
    
    ## Measure data
    x_wd, t_wd, u_wd = Cut_window(10, 0.2, 1.5, 1.2)
   
    ## PINN model and training
    model = HeatPINN()
    print('Model details:',model)
    if Train_model == True:      
        model = train_pinn(num_epochs, u_IC, u_BC_left, u_BC_right)
        torch.save(model, './PINN/result/my_model.pth')
        print('Model is saved.')
    else:
        model = torch.load('./PINN/result/my_model.pth')
        print('Model is loaded.')
        
    
    ## Visualization
    U_pred = visualize_results(model, 1)