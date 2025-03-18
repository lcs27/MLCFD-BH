# -*- coding: utf-8 -*-
"""
@author: Dr FAN Yu
@Commented & modified by: LUO Chensheng @ 17 Mar 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from KNN import KNN

'''
V1 original comment by FAN Yu
# bf for Brute-Force
# The optimization problem concerning alphas is better solved by SMO
# But it is way too complex so here we use a standard optimization routine
# The speed may be affected but the main idea of svm is clearly illustrated
# and most importantly the code is much shorter'
'''

def SVM_brute_force(X,L,c=None):
    '''
    This function realises the (linear) SVM algorithm of classification problem.
    Input:
        X: list of elements, [x1,x2,...,xn]
        L: list of labels, [y1,y2,...,yn]
        c: penalty parameter, default is None(infinity, solid margin).
    Output:
        alpha: the Lagrange multipliers(a_i)
        weights: the weights of the hyperplane, w
        b: the bias of the hyperplane
    '''
    # Initialization
    N_data = X.shape[0] # number of data points
    N_dim = X.shape[1] # dimension of data points
    G = np.zeros((N_data,N_data),dtype='float')
    bnds = []

    # G[i,j] = y_i y_j x_i^T x_j
    for i in range(0,N_data):
        bnds.append((0,c)) # bnds: 0<= a_i <= c
        for j in range(0,N_data):
            G[i,j]=L[i]*L[j]*np.dot(X[i,:],X[j,:])

    # SVM_func = L(a) to minimize       
    def SVM_func(alpha):
        # t[i,j] = a_i a_j y_i y_j x_i^T x_j
        t = np.zeros_like(G)
        for i in range(0,N_data):
            for j in range(0,N_data):
                t[i,j] = alpha[i]*alpha[j]*G[i,j]
        return -np.sum(alpha) + 0.5*np.sum(t)
    
    # cons: sum a_i y_i = 0
    cons = ({'type': 'eq', 'fun': lambda x: np.dot(x,L)})
    alpha_init = np.zeros((N_data,),dtype='float')
    res = minimize(SVM_func, alpha_init, method='SLSQP', bounds=bnds,constraints=cons)
    alpha = res.x

    # w = sum a_i y_i x_i
    weights = np.zeros((N_dim,),dtype='float')
    for i in range(0,N_data):
        weights += alpha[i]*L[i]*X[i,:]

    # b = y_i - w^T x_i for a_i != 0
    ind = np.argmax(alpha)
    b = L[ind]-np.dot(weights,X[ind,:])
    return alpha,weights,b

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
    
    # SVM calculation
    alphas,weights,b = SVM_brute_force(X,L)
    # support vector is a_i != 0, here we use margin 1e-5
    sv_filter = alphas>1e-5
    
    # Data visualization
    plt.figure()
    paradic = {'Romantic':{'marker':'s','color':'red'},
               'Action':{'marker':'o','color':'green'}}
    for Data in DataSet:
        #plt.text(*Data[1],Data[0],ha='center')
        plt.plot(*Data[1],**paradic[Data[2]])
    plt.xlabel('Count of action scenes')
    plt.ylabel('Count of kissing scenes')
    plt.savefig('./Classification-KNN-SVM/result/Movie_data.png')
    
    # Margin and support vector visualization
    xlist = np.linspace(0,120)
    ylist = (-b - weights[0]*xlist)/weights[1]
    ylist_1 = (1-b - weights[0]*xlist)/weights[1]
    ylist_2 = (-1-b - weights[0]*xlist)/weights[1]
    plt.plot(xlist,ylist,color='white',lw=2.0)
    plt.plot(xlist,ylist_1,color='white',lw=1.0,ls='-.')
    plt.plot(xlist,ylist_2,color='white',lw=1.0,ls='-.')
    plt.plot(X[sv_filter,0],X[sv_filter,1],marker='*',color='white',lw=0,fillstyle='none',markersize=14)

    
    Clean_DS = [[np.array(data[1]),data[2]]for data in DataSet]
    labels = ['Romantic','Action']
    
    # SVM result for y by traversing x
    N_samples=20
    xsamples = np.linspace(0,120,N_samples)
    ysamples = np.linspace(0,120,N_samples)
    X,Y = np.meshgrid(xsamples,ysamples)
    Z = np.zeros_like(X)
    Valuedic={'Romantic':1,'Action':0}
    for i in range(0,N_samples):
        for j in range(0,N_samples):
            newmovie = [np.array([X[i,j],Y[i,j]]),'?']
            newlabel = KNN(newmovie,Clean_DS,labels,N=3)
            Z[i,j] = Valuedic[newlabel]
    
    # Result visualization
    plt.pcolor(X,Y,Z,alpha=0.4,cmap='rainbow',shading='auto')
    cbar=plt.colorbar(ticks=[1,0])
    cbar.ax.set_yticklabels(['Romantic', 'Action'])
    plt.ylim(0,120)
    plt.xlim(0,120)
    plt.savefig('./Classification-KNN-SVM/result/Movie_SVMv1.png')
    
    
def test_B():
    '''
    Test function, classify the a random set.
    '''
    # Random data generation and visualization
    N = 15
    X = np.random.rand(N*2,2)
    dis = 0.5
    X[:N,:] = X[:N,:]-dis
    X[N:,:] = X[N:,:]+dis
    L = np.ones((N*2,),dtype='float')
    L[N:]=-1
    paradic = {1:{'marker':'s','color':'orange'},
               -1:{'marker':'o','color':'black'}}
    plt.figure()
    for i in range(0,2*N):
        plt.plot(X[i,0],X[i,1],**paradic[L[i]])
    plt.savefig('./Classification-KNN-SVM/result/Random_data.png')
    
    # SVM classification for different c
    cl = ['purple','red','green']
    ml = ['*','o','^']
    for i,c in enumerate([10,None]):
        alphas,weights,b = SVM_brute_force(X,L,c=c)
        sv_filter = alphas>1e-5
        
        # Margin and support vector visualization
        xlist = np.linspace(-0.5,1.5)
        ylist = (-b - weights[0]*xlist)/weights[1]
        ylist_1 = (1-b - weights[0]*xlist)/weights[1]
        ylist_2 = (-1-b - weights[0]*xlist)/weights[1]
        plt.plot(xlist,ylist,color=cl[i],lw=2.0)
        plt.plot(xlist,ylist_1,color=cl[i],lw=1.0,ls='-.')
        plt.plot(xlist,ylist_2,color=cl[i],lw=1.0,ls='-.')
        plt.plot(X[sv_filter,0],X[sv_filter,1],marker=ml[i],color=cl[i],lw=0,fillstyle='none',markersize=14)
    
    plt.ylim(-0.5,1.5)
    plt.xlim(-0.5,1.5)
    plt.savefig('./Classification-KNN-SVM/result/Random_SVMv1.png')
    
if __name__ == '__main__':
    test_A()
    test_B()

