# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 19:54:25 2019

@author: Prashant Bharti
"""
import numpy as np
import random
import math

alpha = 0.7  #learning rate
lamda = 0.6  #discount factor

def policy(state,action):
    if state > 0:
        s = state
    else:
        s = -1
    if (state > 0) and (action == 1):
        reward = 95
    if (state > 0) and (action == -1):
        reward = 5
    else:
        reward = 0
    return s, reward

#Q-factor update
def Q_update(Qold, Qnext, reward):
    return (1-alpha)*Qold + alpha*(reward + (lamda*Qnext))

#neural network with Q-learning
def neuralnetwork(state, action):
    w = np.random.random((len(state),len(action)))
    Qold = Qnxt1 = Qnxt2 = Qnext = Qnew = 0
    kmax = 100000
    M_max = 10000
    miu = 1/M_max #step size
    i = 0
    while kmax!=0:
        if i==len(state)-1:
            i = 0
        j = i+1
        a = random.choice(list(enumerate(action)))[0]
        if(i==0):
            Qold = w[i][a]
            Qnxt1 = w[j][0]
            Qnxt2 = w[j][1] 
        Qold += w[i][a]*state[i]
        Qnxt1 += w[j][0]*state[j]
        Qnxt2 += w[j][1]*state[j]
        Qnext = max(Qnxt1, Qnxt2)
        [st,reward] = policy(state[i],action[a])
        if st==-1:
            Qnew = 0
        else:
            Qnew += Q_update(Qold,Qnext,reward)
        while M_max!=0:
            for s in range(len(state)):
                w[s][a] += miu*(Qnew-Qold)*state[s]
            M_max -= 1
        i += 1
        kmax -= 1
    Q_factor = w.max(1)
    Q_matrix = Q_factor.reshape(int(math.sqrt(len(state))),int(math.sqrt(len(state))))
    Q_matrix = Q_matrix.astype(int)
    return Q_matrix

# driver function 
def main():
    while True:
        n = int(input("Enter the number of channels: "))    
        if(math.sqrt(n)==int(math.sqrt(n))):
            break
        else:
            print("\n------->Please enter number in square only!<---------")
    channels = np.random.randint(-300,300,(n))
    dim = int(math.sqrt(len(channels)))
    mat_channels = channels.reshape(dim,dim)
    print("\nChannels are divided into "+str(dim)+"x"+str(dim)+" square matrix:\n" + str(mat_channels) + "\n")
    options = [-1,1]    
    Whitespaces = neuralnetwork(channels, options)
    print("\nQ(optimized from the neural network):\n" + str(Whitespaces))

if __name__ == '__main__': 
    main()
