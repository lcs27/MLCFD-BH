# -*- coding: utf-8 -*-
'''
Homework code - selfmade Artificial Neutral Network test code
Author: LUO Chensheng
Time: 13 April 2025
'''

import numpy as np
import matplotlib.pyplot as plt
from testpackage import GetCircularDS, GetXorDS, GetSpiralDS
import scienceplots
import time
import tensorflow as tf
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
fig.savefig('./ANN/result/tensorflow_dataset.png',dpi=600)

# Training ann
start_time = time.time()
print('Training ANN...')
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(4, activation='sigmoid'),
    tf.keras.layers.Dense(4, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
            metrics=['mse', 'binary_accuracy'])
model.fit(X,L, epochs=4000,batch_size=30, verbose=0)
history = model.history.history
end_time = time.time()
print('Training time: {:.2f}s'.format(end_time-start_time))


# Predict
N_samples=20
xsamples = np.linspace(0,1,N_samples)
ysamples = np.linspace(0,1,N_samples)
Xbac,Ybac = np.meshgrid(xsamples,ysamples)
Zbac = np.zeros_like(Xbac)
ll = np.size(Xbac,0)*np.size(Xbac,1)
Xbac = Xbac.reshape(ll,1)
Ybac = Ybac.reshape(ll,1)
xx = np.concatenate((Xbac,Ybac),axis=1)
xx.transpose()
Zbac = model.predict(xx)

Xbac = Xbac.reshape(N_samples,N_samples)
Ybac = Ybac.reshape(N_samples,N_samples)
Zbac = Zbac.reshape(N_samples,N_samples)
fig,ax=plt.subplots()
for i in range(N):
    ax.scatter(X[i,0],X[i,1],marker=paradic[L[i,0]]['marker'],color=paradic[L[i,0]]['color'])
ax.contour(Xbac,Ybac,Zbac,levels=[0.5],colors='k',linestyles=['--'])
cs = ax.contourf(Xbac,Ybac,Zbac,levels=10,alpha=0.2)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_title('ANN Prediction')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
fig.colorbar(cs)
fig.savefig('./ANN/result/tensorflow_predict_ann.png',dpi=600)

