# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:43:56 2021
Revised on Thu Mar 21 12:38:00 2022

@author: Dr FAN Yu
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

#####通用设置#####
plt.rc('font',size=14)
plt.rc('savefig',dpi=600)
plt.rc('savefig',format='png')
#################

plt.close('all')

Sample_Size = 5000
xlist = np.random.rand(Sample_Size)
ylist = np.random.rand(Sample_Size)


#%%
#范雨：这里不需要你现在掌握
#主要功能是让你随时退出计算
global STOP 
STOP= False

def on_press(event):
    print(event.key)
    if event.key == 'x':
        global STOP
        STOP = True
fig = plt.figure(1)
fig.canvas.mpl_connect('key_press_event', on_press)
fig = plt.figure(2)
fig.canvas.mpl_connect('key_press_event', on_press)

#%%
count_hit = 0
xlist_hit = []
ylist_hit = []
xlist_miss = []
ylist_miss = []
PI_hist = []
plt.figure(1)
plt.axis('equal')
theta = np.linspace(0,np.pi/2)
Plot_Every = 100
for i in range(0,Sample_Size):
    x = xlist[i]
    y = ylist[i]
    if x*x+y*y<=1.0:
        count_hit += 1
        xlist_hit.append(x)
        ylist_hit.append(y)
    else:
        xlist_miss.append(x)
        ylist_miss.append(y)
    if STOP:
       break 
    if (i+1) % Plot_Every == 0:
        plt.figure(1)
        plt.cla()
        plt.plot(xlist_hit,ylist_hit,lw=0,color='red',marker='.')
        plt.plot(xlist_miss,ylist_miss,lw=0,color='gray',marker='s')
        plt.fill_between(np.cos(theta),np.sin(theta),color='green',alpha=0.4)
        PI = count_hit/(i+1)*4
        plt.title(r'Hits={},Points={},$\pi\approx${:.5f}'.format(count_hit,Sample_Size,PI))
        PI_hist.append(PI)
        plt.pause(0.01)
        plt.figure(2)
        plt.cla()
        plt.xlabel(f'Iterations * {Plot_Every}')
        plt.ylabel('Approx. PI')
        plt.plot(PI_hist)
        plt.axhline(np.pi,color='red',linestyle=':')
        plt.pause(0.01)
        
        
        
        
        
        
        
        
        
        
