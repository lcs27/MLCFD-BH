# -*- coding: utf-8 -*-
"""
@author: Dr FAN Yu
@Commented & modified by: LUO Chensheng @ 17 Mar 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from KNN import KNN

''''
Original comment by FAN Yu
# v1 notes
# bf for Brute-Force
# The optimization problem concerning alphas is better solved by SMO
# But it is way too complex so here we use a standard optimization routine
# The speed may be affected but the main idea of svm is clearly illustrated
# and most importantly the code is much shorter
# v2 notes
# The way of returning the trained model is updated
# Only the support vectors are kept to compute the model output
# Allow the use of kernel functions 
# A better way of computing b is implemented'
'''

def SVM_brute_force_v2(X,L,c=None,kernel='linear',kargs=[]):
    '''
    This function realises the (linear & non-linear) SVM algorithm of classification problem.
    Input:
        X: list of elements, [x1,x2,...,xn]
        L: list of labels, [y1,y2,...,yn]
        c: penalty parameter, default is None(infinity, solid margin).
        kernel: kernel function, default is linear, 
        other options are polynomial and RBF.
        kargs: kernel function parameters. Polynomial: [g,c,n], RBF: [g]
    Output:
        hyper_plane: the hyperplane function to apply on the new data.
        b: the bias of the hyperplane
        sv_ind: the indices of the support vectors
    '''

    # Initialization
    N_data = X.shape[0]
    G = np.zeros((N_data,N_data),dtype='float')
    bnds = []
    kfuncdic = {'linear':lambda x1,x2:np.dot(x1,x2),
                'polynomial':lambda x1,x2,g,c,n: np.power(g*np.dot(x1,x2)+c,n),
                'RBF':lambda x1,x2,g:np.exp(-np.power(np.linalg.norm(x1-x2),2)/(2*g*g))}
    kfunc = kfuncdic[kernel]

    # G[i,j] = y_i y_j K(x_i,x_j)
    for i in range(0,N_data):
        bnds.append((0,c)) # bnds: 0<= a_i <= c
        for j in range(0,N_data):
            G[i,j]=L[i]*L[j]*kfunc(X[i,:],X[j,:],*kargs)
    
    # SVM_func = L(a) to minimize
    def SVM_func(alpha):
        t = np.zeros_like(G)
        for i in range(0,N_data):
            for j in range(0,N_data):
                t[i,j] = alpha[i]*alpha[j]*G[i,j]
        return -np.sum(alpha) + 0.5*np.sum(t)
    
    # cons: sum a_i y_i = 0
    cons = ({'type': 'eq', 'fun': lambda x: np.dot(x,L)})
    alpha_init = np.zeros((N_data,),dtype='float')
    res = minimize(SVM_func, alpha_init, method='SLSQP', bounds=bnds,constraints=cons)

    # support vectors are i with a_i non zero
    alpha = res.x
    sv_ind = np.array(range(0,N_data))
    sv_ind = sv_ind[alpha>1e-5]

    # hyperplane g(x) = sum a_i y_i K(x_i,x)
    def hyper_plane(new_x):
        re = 0
        for i in sv_ind:
            re += alpha[i]*L[i]*kfunc(new_x,X[i],*kargs)
        return re
    

    # b = average of (y_i - g(x_i)) for support vectors
    b=0
    for ind in sv_ind:
        b += L[ind]-hyper_plane(X[ind,:])
    b = b/sv_ind.shape[0]
    return hyper_plane,b,sv_ind

def test_A():
    '''
    Test function, classify the movie data set.
    '''

    # Data set
    DataSet=[['California Man',[3,104],'Romantic'],
             ['He\'s Not Rally into Dudes',[2,100],'Romantic'],
             ['Beautiful Woman',[1,81],'Romantic'],
             ['Kevin longblade',[101,10],'Action'],
             ['Robo Slayer 3000',[99,5],'Action'],
             ['Amped 2',[98,2],'Action']]
    
    Valuedic={'Romantic':1,'Action':-1}
    
    X = np.array([np.array(data[1]) for data in DataSet],dtype='float')
    L = np.array([Valuedic[data[2]] for data in DataSet],dtype='float')
        
    hyper_plane,b,sv_ind = SVM_brute_force_v2(X,L,kernel='RBF',kargs=[1e2])
    
    # Data visualization
    plt.figure()
    paradic = {'Romantic':{'marker':'s','color':'red'},
               'Action':{'marker':'o','color':'green'}}
    for Data in DataSet:
        #plt.text(*Data[1],Data[0],ha='center')
        plt.plot(*Data[1],**paradic[Data[2]])
    plt.xlabel('Count of action scenes')
    plt.ylabel('Count of kissing scenes')
    plt.plot(X[sv_ind,0],X[sv_ind,1],marker='*',color='white',lw=0,fillstyle='none',markersize=14) # support vectors
    
    Clean_DS = [[np.array(data[1]),data[2]]for data in DataSet]
    labels = ['Romantic','Action']
    
    # SVM for y by traversing x
    N_samples=20
    xsamples = np.linspace(0,120,N_samples)
    ysamples = np.linspace(0,120,N_samples)
    X,Y = np.meshgrid(xsamples,ysamples)
    Z = np.zeros_like(X)
    Z_KNN = np.zeros_like(X)
    Valuedic={'Romantic':1,'Action':-1}
    for i in range(0,N_samples):
        for j in range(0,N_samples):
            newmovie = np.array([X[i,j],Y[i,j]])
            Z[i,j] = hyper_plane(newmovie)+b
            newmovie = [np.array([X[i,j],Y[i,j]]),'?']
            newlabel = KNN(newmovie,Clean_DS,labels,N=1)
            Z_KNN[i,j] = Valuedic[newlabel]
    ZL =np.sign(Z)
    plt.pcolor(X,Y,Z_KNN,alpha=0.4,cmap='rainbow',shading='auto')
    cbar=plt.colorbar(ticks=[1,-1])
    cbar.ax.set_yticklabels(['Romantic', 'Action'])
    plt.contour(X,Y,Z,levels=[-1,0,1],colors='k',linestyles=[':','-',':'])
    plt.ylim(0,120)
    plt.xlim(0,120)
    plt.savefig('./Classification-KNN-SVM/result/Movie_SVMv2.png')
    
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X,Y,Z,cmap='rainbow')
    fig.colorbar(surf)
    plt.savefig('./Classification-KNN-SVM/result/Movie_SVMv2_3D.png')
    
    
def test_B():
    '''
    Test function, classify the a random set.
    '''
    # Random data generation and visualization
    N = 20
    X = np.random.rand(N*2,2)
    dis = 0.5
    X[:N,:] = X[:N,:]-dis
    X[N:,:] = X[N:,:]+dis
    L = np.ones((N*2,),dtype='float')
    L[N:]=-1
    paradic = {1:{'marker':'s','color':'blue'},
               -1:{'marker':'o','color':'black'}}
    plt.figure()
    for i in range(0,2*N):
        plt.plot(X[i,0],X[i,1],**paradic[L[i]])
    plt.savefig('./Classification-KNN-SVM/result/Random_data.png')

    # SVM classification
    cl = ['purple','red','green']
    ml = ['*','o','^']
    #hyper_plane,b,sv_ind = SVM_brute_force_v2(X,L,c=10)
    hyper_plane,b,sv_ind = SVM_brute_force_v2(X,L,c=1,kernel='RBF',kargs=[1e0])
    
    plt.plot(X[sv_ind,0],X[sv_ind,1],marker=ml[0],color=cl[0],lw=0,fillstyle='none',markersize=14)
        
    Clean_DS = []
    for i in range(0,2*N):
        Clean_DS.append([X[i,:],L[i]])
    labels = [1,-1]
    
    # SVM for y by traversing x
    N_samples=20
    xsamples = np.linspace(-0.5,1.5,N_samples)
    ysamples = np.linspace(-0.5,1.5,N_samples)
    X,Y = np.meshgrid(xsamples,ysamples)
    Z = np.zeros_like(X)
    Z_KNN = np.zeros_like(X)
    for i in range(0,N_samples):
        for j in range(0,N_samples):
            newmovie = np.array([X[i,j],Y[i,j]])
            Z[i,j] = hyper_plane(newmovie)+b
            
            newmovie = [np.array([X[i,j],Y[i,j]]),'?']
            newlabel = KNN(newmovie,Clean_DS,labels,N=3)
            Z_KNN[i,j] = newlabel
            
    ZL = np.sign(Z)
    plt.pcolor(X,Y,Z_KNN,alpha=0.4,cmap='rainbow',shading='auto')
    cbar=plt.colorbar(ticks=[1,-1])
    cbar.ax.set_yticklabels(['Group 1', 'Group 2'])
    plt.ylim(-0.5,1.5)
    plt.xlim(-0.5,1.5)
    plt.contour(X,Y,Z,levels=[-1,0,1],colors='k',linestyles=[':','-',':'])
    plt.savefig('./Classification-KNN-SVM/result/Random_SVMv2.png')
    
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X,Y,Z,cmap='rainbow')
    fig.colorbar(surf)
    plt.savefig('./Classification-KNN-SVM/result/Random_SVMv2_3D.png')
    
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
            L[i] = -1
    return X,L,N

def GetXorDS(N = 60):
    '''
    Generate a XOR data set.
    '''
    X = (np.random.rand(N,2)-0.5)*2
    L = np.ones((N,),dtype='float')
    for i in range(0,N):
        if X[i,0]*X[i,1]>=0:
            L[i] = 1
        else:
            L[i] = -1
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
            y[ix] = -1
    return X,y,N*K

def test_C():
    '''
    Test function, classify the self-generated data set.
    '''
    # Data generation and visualization
    X,L,N = GetCircularDS()
    #X,L,N = GetXorDS()
    #X,L,N = GetSpiralDS()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=L, s=40, cmap='Spectral')
    plt.savefig('./Classification-KNN-SVM/result/self_data.png')
    
    paradic = {1:{'marker':'s','color':'blue'},
               -1:{'marker':'o','color':'black'}}
    plt.figure()
    for i in range(0,N):
        plt.plot(X[i,0],X[i,1],**paradic[L[i]])
    plt.savefig('./Classification-KNN-SVM/result/self_data_labeled.png')
    

    # Classification
    cl = ['purple','red','green']
    ml = ['*','o','^']
    c=None
    #hyper_plane,b,sv_ind = SVM_brute_force_v2(X,L,c,kernel='RBF',kargs=[1])
    hyper_plane,b,sv_ind = SVM_brute_force_v2(X,L,c=1e4,kernel='RBF',kargs=[1])
    #hyper_plane,b,sv_ind = SVM_brute_force_v2(X,L,c,kernel='polynomial',kargs=[1,0,5])
    plt.plot(X[sv_ind,0],X[sv_ind,1],marker=ml[0],color=cl[0],lw=0,fillstyle='none',markersize=14) # support vectors
    
    Clean_DS = []
    for i in range(0,N):
        Clean_DS.append([X[i,:],L[i]])
    labels = [1,-1]
    
    # SVM result for y by traversing x
    N_samples=30
    xsamples = np.linspace(-1.2,1.2,N_samples)
    ysamples = np.linspace(-1.2,1.2,N_samples)
    X,Y = np.meshgrid(xsamples,ysamples)
    Z = np.zeros_like(X)
    Z_KNN = np.zeros_like(X)
    for i in range(0,N_samples):
        for j in range(0,N_samples):
            newmovie = np.array([X[i,j],Y[i,j]])
            Z[i,j] = hyper_plane(newmovie)+b
            
            newmovie = [np.array([X[i,j],Y[i,j]]),'?']
            newlabel = KNN(newmovie,Clean_DS,labels,N=1)
            Z_KNN[i,j] = newlabel
    ZL = np.sign(Z)
    
    #plt.pcolor(X,Y,ZL,alpha=0.4,cmap='jet',shading='auto')
    plt.pcolor(X,Y,Z_KNN,alpha=0.4,cmap='rainbow',shading='auto')
    cbar=plt.colorbar(ticks=[1,-1])
    cbar.ax.set_yticklabels(['Group 1', 'Group 2'])
    plt.contour(X,Y,Z,levels=[-1,0,1],colors='k',linestyles=[':','-',':'])
    plt.ylim(-1.2,1.2)
    plt.xlim(-1.2,1.2)
    plt.savefig('./Classification-KNN-SVM/result/self_SVMv2.png')

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X,Y,Z,cmap='rainbow')
    fig.colorbar(surf)
    plt.savefig('./Classification-KNN-SVM/result/self_SVMv2_3D.png')
    
if __name__ == '__main__':
    test_A()
    test_B()
    test_C()
