# -*- coding: utf-8 -*-
"""
@author: Dr FAN Yu
@Commented & modified by: LUO Chensheng @ 17 Mar 2025
"""

import numpy as np
import matplotlib.pyplot as plt

def KNN(newdata,DS,labels,N=5):
    '''
    This function realises the KNN algorithm of classification problem.
    Input:
        newdata: list, data to classify, [element(x), label(y)], y is a placeholder.
        DS: list of elements, [x1,x2,...,xn]
        labels: list of labels, [y1,y2,...,yn]
        N: the number of nearest neighbors, default is 5.
    Output:
        newdata: the new data with the label.
        Final_Name: the label of the new data y
    '''

    # Step 0: initialization
    Distance = np.zeros(len(DS))
    Labelhits = {}
    for name in labels:
        Labelhits[name] = 0
    
    # Step 1: distance calculation
    for i,Data in enumerate(DS):
        Distance[i] = np.linalg.norm(Data[0]-newdata[0])
    
    # Step 2: ordering & find the N-nearest neighbors
    pos = np.argpartition(Distance,N)

    # Step 3: count the labels of the N-nearest neighbors
    for p in pos[:N]:
        Labelhits[DS[p][1]]+=1 
    keys=[]
    hits=[]
    for key,value in Labelhits.items():
        keys.append(key)
        hits.append(value)
    
    # Step 4: find the label with the most hits
    max_id = np.argmax(hits)
    Final_Name = keys[max_id]
    newdata[1]=Final_Name
    return Final_Name


def AIDating():
    '''
    Test function, classify the dating data set.
    AI Dating set is composed of 4 parameters:
        Input parameters:
            1. Flight miles (x1)
            2. Gaming hours (x2)
        Output/Predicted parameters:
            3. Eating icecream (y1)
            4. Feeling (y2)
    '''
    DataSet = np.loadtxt('./Classification-KNN-SVM/datingDataSet.txt')
    
    # Normalisation of first 3 coefficients
    for i in range(0,3):
        Dmax = np.max(DataSet[:,i])
        Dmin = np.min(DataSet[:,i])
        DataSet[:,i] = (DataSet[:,i]-Dmin)/(Dmax-Dmin)
    
    # Category division of 4th coefficient
    LikeSet = np.array([DataSet[i,:] for i in range(0,len(DataSet)) if DataSet[i,3]==3])
    SosoSet = np.array([DataSet[i,:] for i in range(0,len(DataSet)) if DataSet[i,3]==2])
    HateSet = np.array([DataSet[i,:] for i in range(0,len(DataSet)) if DataSet[i,3]==1])
    
    # Data visualization
    plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.scatter(LikeSet[:,0],LikeSet[:,1],LikeSet[:,2],marker='o',label='Like')
    ax.scatter(SosoSet[:,0],SosoSet[:,1],SosoSet[:,2],marker='s',label='Soso')
    ax.scatter(HateSet[:,0],HateSet[:,1],HateSet[:,2],marker='^',label='Hate')
    ax.legend()
    ax.set_xlabel('Flight Miles')
    ax.set_ylabel('Gaming Hours')
    ax.set_zlabel('Eating Icecream')
    plt.savefig('./Classification-KNN-SVM/result/AIDating_data.png')
    
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(LikeSet[:,0],LikeSet[:,1],marker='o',fillstyle='none',lw=0,color='red',label='Like')
    ax.plot(SosoSet[:,0],SosoSet[:,1],marker='s',fillstyle='none',lw=0,color='green',label='Soso')
    ax.plot(HateSet[:,0],HateSet[:,1],marker='^',fillstyle='none',lw=0,color='purple',label='Hate')
    ax.legend()
    ax.set_xlabel('Flight Miles')
    ax.set_ylabel('Gaming Hours')
    plt.savefig('./Classification-KNN-SVM/result/AIDating_data2.png')
    
    # KNN for y2 by traversing x1 and x2
    Clean_DS = [[DataSet[i,:2],DataSet[i,3]] for i in range(0,len(DataSet))]
    N_samples=20
    xsamples = np.linspace(0,1,N_samples)
    ysamples = np.linspace(0,1,N_samples)
    X,Y = np.meshgrid(xsamples,ysamples)
    Z = np.zeros_like(X)
    labels=[1,2,3]
    for i in range(0,N_samples):
        for j in range(0,N_samples):
            newguy = [np.array([X[i,j],Y[i,j]]),-1]
            newfeel = KNN(newguy,Clean_DS,labels,N=7)
            Z[i,j] = newfeel

    # Result visualization
    plt.pcolormesh(X,Y,Z,alpha=0.5,cmap='rainbow',shading='auto')
    cbar=plt.colorbar(ticks=[1,2,3])
    cbar.ax.set_yticklabels(['Hate', 'Soso', 'Like'])
    plt.savefig('./Classification-KNN-SVM/result/AIDating_KNN.png')
    

def TestKNN():
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
    
    # Data visualization
    plt.figure()
    paradic = {'Romantic':{'marker':'s','color':'red'},
               'Action':{'marker':'o','color':'green'}}
    for Data in DataSet:
        plt.text(*Data[1],Data[0],ha='center')
        plt.plot(*Data[1],**paradic[Data[2]])
    plt.xlabel('Count of action scenes')
    plt.ylabel('Count of kissing scenes')
    plt.savefig('./Classification-KNN-SVM/result/Movie_data.png')
    
    Clean_DS = [[np.array(data[1]),data[2]]for data in DataSet]
    labels = ['Romantic','Action']

    # KNN for new movie
    newmovie = [np.array([25,28]),'?']
    plt.plot(*newmovie[0],marker='x')
    name = KNN(newmovie,Clean_DS,labels,N=3)
    print('newmovie is a %s movie'%(name,))
    
    # KNN for y by traversing x
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
    plt.pcolor(X,Y,Z,alpha=0.5,cmap='rainbow',shading='auto')
    cbar=plt.colorbar(ticks=[1,0])
    cbar.ax.set_yticklabels(['Romantic', 'Action'])
    plt.savefig('./Classification-KNN-SVM/result/Movie_KNN.png')

if __name__ == '__main__':
    TestKNN()
    AIDating()