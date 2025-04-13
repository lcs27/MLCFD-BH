# -*- coding: utf-8 -*-
'''
Homework code - selfmade Artificial Neutral Network
Author: LUO Chensheng
Time: 12 April 2025
'''

import numpy as np

def GetCircularDS(N = 30):
    '''
    Generate a circular data set.
    '''
    X = (np.random.rand(N,2)-0.5)*2
    L = np.ones((N,),dtype='float')
    for i in range(0,N):
        d = np.power(X[i,0],2) + np.power(X[i,1],2)
        if d<=0.64:
            L[i] = 1
        else:
            L[i] = 0
    return X,L,N

def GetXorDS(N = 60):
    '''
    Generate a XOR data set.
    '''
    X = np.random.rand(N,2)
    L = np.ones((N,),dtype='float')
    for i in range(0,N):
        # if (X[i,0]-0.5)<0:
        #     X[i,0] /= 1.25
        # else:
        #     X[i,0] = 0.6 + (X[i,0]-0.5)/1.25

        # if (X[i,1]-0.5)<0:
        #     X[i,1] /= 1.25
        # else:
        #     X[i,1] = 0.6 + (X[i,1]-0.5)/1.25

        if (X[i,0]-0.5)*(X[i,1]-0.5)>=0:
            L[i] = 1
        else:
            L[i] = 0
    return X,L,N

def GetSpiralDS():
    '''
    Generate a spiral data set.
    '''
    N = 20 # 每类样本点的个数
    D = 2 # 维度（每个样本点有2个特征,即横坐标和纵坐标）
    K = 4 # 旋臂数
    X = np.zeros((N*K,D)) # 数据矩阵 (每一行是一个样本)
    y = np.zeros(N*K, dtype='int') # 类标签
    for j in range(K):
        ix = list(range(N*j,N*(j+1)))
        r = np.linspace(0.0,1,N) 
        t = np.linspace(j*5,(j+1)*5,N) #+ np.random.randn(N)*0.001
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        if j in [0,2]:
            y[ix] = 1
        else:
            y[ix] = 0
    return X,y,N*K