# -*- coding: utf-8 -*-
'''
Homework code - selfmade Artificial Neutral Network test code
Author: LUO Chensheng
Time: 12 April 2025
'''

import numpy as np
from ANN_selfmade import ANN
import matplotlib.pyplot as plt
from testpackage import GetCircularDS, GetXorDS, GetSpiralDS
import scienceplots
import time
plt.style.use(['science', 'ieee', 'std-colors'])

# Generate dataset
X,L,N = GetXorDS(N=100)
paradic = {1:{'marker':'s','color':'blue'},
            0:{'marker':'o','color':'red'}}
X = np.array(X)
L = np.array(L).reshape(len(L),1)
fig,ax=plt.subplots()
for i in range(N):
    ax.scatter(X[i,0],X[i,1],marker=paradic[L[i,0]]['marker'],color=paradic[L[i,0]]['color'])
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_title('Training Dataset')
fig.savefig('./ANN/result/selfmade_dataset.png',dpi=600)


# Training ann
start_time = time.time()
print('Training ANN...')
ann = ANN(input_size=2, hidden_layers=[4,4], output_size=1,learning_rate=1)
losses = ann.training(X,L, round = 6000)
end_time = time.time()
print('Training time: {:.2f}s'.format(end_time-start_time))

fig,ax=plt.subplots()
ax.semilogy(losses)
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
ax.set_title('Training Loss')
ax.autoscale(tight=True)
fig.savefig('./ANN/result/selfmade_losses_ann.png',dpi=600)



# Predict
N_samples=20
xsamples = np.linspace(0,1,N_samples)
ysamples = np.linspace(0,1,N_samples)
Xbac,Ybac = np.meshgrid(xsamples,ysamples)
Zbac = np.zeros_like(Xbac)

for i in range(0,N_samples):
    for j in range(0,N_samples):
        Zbac[i,j] = ann.predict(np.array([Xbac[i,j],Ybac[i,j]]))

fig,ax=plt.subplots()
for i in range(N):
    ax.scatter(X[i,0],X[i,1],marker=paradic[L[i,0]]['marker'],color=paradic[L[i,0]]['color'])
ax.contour(Xbac,Ybac,Zbac,levels=[0.5],colors='k',linestyles=['--'])
ax.contourf(Xbac,Ybac,Zbac,levels=10,alpha=0.2)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_title('ANN Prediction')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
fig.savefig('./ANN/result/selfmade_predict_ann.png',dpi=600)
