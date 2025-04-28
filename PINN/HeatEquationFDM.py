# -*- coding: utf-8 -*-
'''
Homework code - solution of heat equation using explicit finite difference method
Author: Dr QIU Lu
Commented & modified by: LUO Chensheng 
Time: 28 April 2025
'''


import numpy as np
import math
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee', 'std-colors'])


def Compute_T_field(L, total_time, T_left, T_right, T_initial):
    '''
    Solve the heat equation using explicit finite difference method.
    Input:
        L: length of the rod (m)
        total_time: total simulation time (s)
        T_left: left boundary temperature (°C), constant or array of shape (nt)
        T_right: right boundary temperature (°C), constant or array of shape (nt)
        T_initial: initial temperature (°C), constant or array of shape (nx)
    Output:
        T_history: temperature distribution over time, np array of shape (nt, nx)
    '''

    ## Parameters
    dt = total_time / 5000.0 
    dx = L / 100.0
    print(f"dt = {dt}, dx = {dx}")
    nx = int(L / dx) + 1 # Number of grid points
    nt = int(total_time / dt) + 1 # Number of time steps
    
    ## Initialization of arrays
    T = np.zeros(nx)
    T_initial_dist = np.linspace(0, T_initial, nx)  # 初始温度分布
    T = T_initial_dist.copy()
    T_history = [T.copy()]
    
    ## Explicit finite difference method
    for j in range(nt):
        T_new = T.copy()
        
        for i in range(1, nx-1):
            T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])
        
        # Boundary conditions
        T_new[0] = math.sin(T_left * math.pi * j * dt) #T_left #
        #T_new[0] = Q_right / alpha * dx + T_new[1] #math.sin(T_left * math.pi * j * dt) #T_left #
        #T_new[0] = T_left 
        T_new[-1] = T_right #-math.sin(T_right * pi * j * dt)
        #T_new[-1] = Q_right / alpha * dx + T_new[nx-2]
        
        ## Update temperature distribution
        T = T_new.copy()
        T_history.append(T.copy())
        
    return T_history
        
    
    
def visualize_results(T_history):
    '''
    Visualize the temperature distribution over time.
    '''
    x_p = np.linspace(x_min, x_max, 101)
    t_p = np.linspace(t_min, t_max, 5002)
    X_p, T_p = np.meshgrid(x_p, t_p)

    plt.figure(figsize=(3, 8))
    plt.contourf(X_p, T_p, T_history, cmap='viridis')
    plt.colorbar(label='Temperature')
    plt.title('Temperature Distribution')
    plt.xlabel('Position (x)')
    plt.ylabel('Time (t)')
            
    plt.savefig('./PINN/result/heat_equation.png', dpi=600)

if __name__ == "__main__":
    ## Initialization
    x_min, x_max = 0, 1
    t_min, t_max = 0, 3
    alpha = 0.08  # heat conduction coefficient (m²/s)
    L = x_max - x_min                # Total length of the rod (m)
    total_time = t_max - t_min       # Total simulation time (s)
    
    # Initial condition
    T_initial = 0.0

    # Boundary conditions
    T_left = 1
    T_right = 1

    ## Solution
    T_history = Compute_T_field(L, total_time, T_left, T_right, T_initial)
    
    ## Visualize results
    visualize_results(T_history)