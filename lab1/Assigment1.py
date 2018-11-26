# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:12:49 2017

@author: eri
"""

import pickle
import numpy as np
from pdb import set_trace
from timeit import default_timer
import matplotlib.pyplot as plt



def LoadBatch (file):
    X = np.array;
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data'].T/255;
    y = dict[b'labels'];# list of labels as int
    Y = np.zeros((10, X.shape[1]));
    for i in range(len(y)):
        Y[y[i], i ] = 1;
    return X, Y, y
    
def EvaluateClassifier(X, W, b):
    s = W.dot(X) + b;
    denom = np.dot(np.ones((1,W.shape[0])), np.exp(s));
    p = np.exp(s)/denom; # devide each column of s with a vector element of denom
    return p
        
def ComputeCost(X, Y, W, b, lamda):
    W2 = np.power(W, 2);
    W2Sum = W2.sum();
    P = EvaluateClassifier(X, W, b);
    a = np.diag(np.dot(Y.T, P)).reshape((1,X.shape[1])) #create a vector 1xd of probs
    logP = -np.log(a);
    logPSum = logP.sum();
    loss = (1/X.shape[1])*logPSum;
    cost = (1/X.shape[1])*logPSum + lamda*W2Sum;
    return loss, cost
    
def ComputeAccuracy(X, y, W, b):
    P = EvaluateClassifier(X, W, b);
    PArgMax = np.argmax(P, axis=0)
    MisClas = PArgMax - y; # array (10000,). The correctly classified returns 0 the others nonzero
    sumMisClas = np.count_nonzero(MisClas)
    accuracy = 1-sumMisClas/X.shape[1]
    return accuracy
    

def ComputeGradients(X, Y, P, W, lamda):
    DLb = np.zeros((W.shape[0],1));
    DLw = np.zeros((W.shape[0],W.shape[1]));
    g = P-Y;
    for i in range(P.shape[1]):
        DLb += g[:,i].reshape((g.shape[0],1)); 
        DLw += np.dot(g[:,i].reshape((g.shape[0],1)), X.T[i,:].reshape((1,X.shape[0])));
    DLb = DLb/X.shape[1];
    DLw = DLw/X.shape[1];
    DJw = DLw + 2*lamda*W;
    DJb = DLb;
    return DJw, DJb; 
    
    
    
def ComputeGradsNumSlow(X, Y, W, b, lamda, h = 1e-6):

    DWnum = np.zeros((W.shape[0], W.shape[1]))
    Dbnum = np.zeros((W.shape[0], 1))

    for i in range(b.shape[0]):
        b_try = np.copy(b)      
        b_try[i] -= h       
        l1, c1 = ComputeCost(X, Y, W, b_try, lamda)
        b_try = np.copy(b)
        b_try[i] += h
        l2, c2 = ComputeCost(X, Y, W, b_try, lamda)
        Dbnum[i] = (c2-c1) / (2 * h)
        
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.copy(W)
            W_try[i][j] -= h
            l1, c1 = ComputeCost(X, Y, W_try, b, lamda)
    
            W_try = np.copy(W)
            W_try[i][j] += h
            l2, c2 = ComputeCost(X, Y, W_try, b, lamda)
    
            DWnum[i][j] = (c2-c1) / (2*h)

    return  DWnum, Dbnum
    
    
def VisWeights(W):
    s_im=[]
    for i in W:
        Im=(i.reshape(3, 32, 32)).T        
        image=((Im-np.min(Im))/(np.max(Im)-np.min(Im)))
        image =  np.transpose(image, (1, 0, 2))
        s_im.append(image)

    fig = plt.figure()
    
    for i in range(W.shape[0]):
        ax = fig.add_subplot(1, 10, i+1)
        ax.imshow(s_im[i])
        ax.axis('off')
    plt.show()
    
    
def MiniBatchGD(X, Y, y, X_val, Y_val, y_val, W, b, n_batch, eta, n_epochs, lamda, eta_decaying):
    loss = [];
    cost = [];
    loss_val = [];
    cost_val = [];
    acc = [];
    acc_val = [];
    listW = [];
    for k in range(n_epochs):
        for i in range(1,X.shape[1]//n_batch):
            i_start = (i-1)*n_batch + 1;
            i_end = i*n_batch;
            Xbatch = X[:, i_start:i_end];
            Ybatch = Y[:, i_start:i_end];
            P = EvaluateClassifier(Xbatch, W, b);
            Dw , Db = ComputeGradients(Xbatch, Ybatch, P, W, lamda) 
            W = W - eta* Dw;
            b = b- eta * Db;
        l, c = ComputeCost(X, Y, W, b, lamda);
        l_val, c_val = ComputeCost(X_val, Y_val, W, b, lamda);
        loss.append(l);
        cost.append(c);
        loss_val.append(l_val);
        cost_val.append (c_val);
        acc.append(ComputeAccuracy(X, y, W, b));
        acc_val.append(ComputeAccuracy(X_val, y_val, W, b))
        listW.append(W)
        if eta_decaying == 'on':
            eta -=eta-eta*0.9;
            print (eta)
    figure1, ax = plt.subplots(2, sharex=True)
    ax[0].plot(loss,label = 'training set')
    ax[0].plot(loss_val, label='validation set')
    ax[1].plot(cost,label = 'training set')
    ax[1].plot(cost_val,label = 'validation set')
    figure2 =plt.figure()
    plt.plot(acc,label = 'training set')
    plt.plot(acc_val, label='validation set')
    plt.legend()
    plt.show()
    print(np.max(acc_val))
    idxMaxW = np.argmax(acc_val);
    W = listW[idxMaxW]
    
    return W
        
     
    
if __name__ == "__main__":

#==============================================================================
# call minibatch
#==============================================================================
    np.random.seed(400)
    W = np.random.normal(0, 0.01, (10, 3072));
    b = np.random.normal(0, 0.01, (10, 1));

# 10000 as training, 10000 as validation
    X, Y , y  = LoadBatch ('Datasets/data_batch_1')
#    X_val, Y_val, y_val = LoadBatch ('Datasets/data_batch_2')
#    start= default_timer();
#    W = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, W, b, n_batch=100, eta=0.01, n_epochs=40, lamda=0, eta_decaying="on")
#    print(default_timer() - start)
    
# 19000 training set 1000 validation 
#    X1, Y1 , y1  = LoadBatch ('Datasets/data_batch_1')
#    X2, Y2, y2 = LoadBatch ('Datasets/data_batch_2')
#    X  = np.concatenate((X1,X2[:,:9000] ), axis=1)
#    Y  = np.concatenate((Y1,Y2[:,:9000] ), axis =1)
#    y  = np.concatenate((y1,y2[:9000] ))
#    X_val, Y_val, y_val = LoadBatch ('Datasets/data_batch_3')
#    start= default_timer();
#    W = MiniBatchGD(X, Y, y.tolist(), X_val[:,:1000], Y_val[:,:1000], y_val[:1000], W, b, n_batch=100, eta=0.01, n_epochs=40, lamda=0,eta_decaying="off")
#    print(default_timer() - start)
    
#    VisWeights(W)


#==============================================================================
#  compare gradients calculated numerical and analytical
#==============================================================================
    P = EvaluateClassifier(X, W, b);
    Wslow, bslow = ComputeGradsNumSlow(X[:,:10], Y, W, b, 0, h = 1e-6);
    W_analyt, b_analyt = ComputeGradients(X[:,:10], Y[:,:10], P[:,:10], W, 0)
    WDiff = np.absolute(Wslow - W_analyt)
    WAverage = np.average(WDiff)
    AnalytWAbs = np.absolute(W_analyt)
    NumWAbs = np.absolute(Wslow)
    sumW = AnalytWAbs + NumWAbs
    res = WAverage / np.amax(sumW) 
    print("res =", res)
#    



