# -*- coding: utf-8 -*-
'''
Homework1 code
Author: LUO Chensheng
Time: 17 Mar 2025
'''
import numpy as np
import matplotlib.pyplot as plt
from SVMv2 import SVM_brute_force_v2 as SVM, GetXorDS, GetSpiralDS, GetCircularDS

def GetDisDS(N=20,dis=0.3):
    X = np.random.rand(N*2,2)
    X[:N,:] = X[:N,:]-dis
    X[N:,:] = X[N:,:]+dis
    L = np.ones((N*2,),dtype='float')
    L[N:]=-1
    return X,L,N*2
## Homework 1

# Random data generation and visualization
X,L,N = GetXorDS(N=40)
paradic = {1:{'marker':'s','color':'blue'},
            -1:{'marker':'o','color':'red'}}

ax,fig=plt.subplots()
for i in range(N):
    fig.scatter(X[i,0],X[i,1],marker=paradic[L[i]]['marker'],color=paradic[L[i]]['color'])
fig.set_xlim(-1.2,1.2)
fig.set_ylim(-1.2,1.2)
plt.savefig('./Classification-KNN-SVM/result/homework1/dataset.png')

# Preparative work
kk = 0
N_samples=20
xsamples = np.linspace(-1.2,1.2,N_samples)
ysamples = np.linspace(-1.2,1.2,N_samples)
Xbac,Ybac = np.meshgrid(xsamples,ysamples)
Zbac = np.zeros_like(Xbac)

# Traversing...
for sigma in [2e-1,1,5]:
    for c in [1e1,1e2,1e3]:
        hyper_plane,b,sv_ind = SVM(X,L,c=c,kernel='polynomial',kargs=[sigma])
        plt.figure()
        for i in range(N):
            plt.scatter(X[i,0],X[i,1],marker=paradic[L[i]]['marker'],color=paradic[L[i]]['color'])
        plt.scatter(X[sv_ind,0],X[sv_ind,1],facecolors='none',edgecolors='r',s=100)

        for i in range(0,N_samples):
            for j in range(0,N_samples):
                newmovie = np.array([Xbac[i,j],Ybac[i,j]])
                Zbac[i,j] = hyper_plane(newmovie)+b

        plt.contour(Xbac,Ybac,Zbac,levels=[-1,0,1],colors='k',linestyles=[':','-',':'])
        plt.title('sigma = %.1f, c = %.0f'%(sigma,c))
        plt.savefig('./Classification-KNN-SVM/result/homework1/sigma_c_%d.png'%(kk))
        
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1, projection='3d')
        surf = ax.plot_surface(Xbac,Ybac,Zbac,cmap='rainbow')
        fig.colorbar(surf)
        plt.title('sigma = %.1f, c = %.0f'%(sigma,c))
        plt.savefig('./Classification-KNN-SVM/result/homework1/sigma_c_%d_3D.png'%(kk))

        kk += 1


